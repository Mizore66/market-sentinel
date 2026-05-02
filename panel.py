"""LangGraph "expert panel" multi-agent for Market Sentinel.

The panel splits one user question into specialized sub-investigations that
run in parallel, then synthesizes the findings:

    +-------------+        +-----------------------+
    |             |  -->   | Technical Analyst     |  --+
    |   START     |        | (price + history)     |    |
    |             |  -->   +-----------------------+    |    +-------------+
    +-------------+                                     +--> | Synthesizer | --> END
                  |        +-----------------------+    |    +-------------+
                  +----->  | Fundamental Researcher|  --+
                           | (news + 10-K + buzz)  |
                           +-----------------------+

Each worker is a small ReAct agent (LangChain ``create_tool_calling_agent``)
with a focused tool subset - this is the "Expert Panel" pattern from the plan.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any, TypedDict

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

try:
    from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
except ImportError:  # pragma: no cover
    from langchain.agents import AgentExecutor, create_tool_calling_agent

from langgraph.graph import END, START, StateGraph

from agent import _build_llm
from tools import get_stock_price, search_market_news
from tools_extra import (
    FUNDAMENTAL_TOOLS,
    TECHNICAL_TOOLS,
    get_market_sentiment,
    score_sentiment,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Worker prompts                                                              #
# --------------------------------------------------------------------------- #


TECHNICAL_PROMPT = (
    "You are the **Technical Analyst** on the Market Sentinel expert panel.\n"
    "Your job: produce a tight, numbers-first read on price action.\n"
    "Always call `get_stock_price` first for the current quote, then "
    "`get_price_history` for trend / volatility context.\n"
    "Output 3-6 short bullet points covering: current price, recent return, "
    "volatility, drawdown, and any notable level (52-week high/low). "
    "Never invent numbers - if a tool fails, say so."
)

FUNDAMENTAL_PROMPT = (
    "You are the **Fundamental Researcher** on the Market Sentinel expert panel.\n"
    "Your job: explain the *why* behind the price - news catalysts, sentiment, "
    "and disclosed risks.\n"
    "Decide which of these tools to call (you may call several):\n"
    "  - `search_market_news` for breaking headlines\n"
    "  - `get_reddit_buzz` for retail-investor sentiment\n"
    "  - `get_market_sentiment` for an aggregate news-sentiment score\n"
    "  - `query_sec_10k` if the user asks about risks, business segments, or "
    "    anything an annual report would address\n"
    "  - `query_uploaded_document` if the user uploaded a doc and the question "
    "    plausibly targets it\n"
    "Output 3-6 short bullet points. Cite source URLs inline as Markdown links "
    "wherever possible. Never invent facts."
)

SYNTHESIZER_PROMPT = (
    "You are the **Lead Editor** of the Market Sentinel expert panel.\n"
    "Two specialists handed in their notes. Combine them into ONE crisp answer "
    "for the user. Rules:\n"
    "1. Lead with a 1-2 sentence direct answer.\n"
    "2. Then a short '## Technical' section (price / trend) and a "
    "   '## Fundamental' section (news / sentiment / filings).\n"
    "3. Preserve every source link from the specialists.\n"
    "4. If a specialist reported missing data, surface that limitation; do "
    "   not paper over it.\n"
    "5. End with a single 'Bottom line:' sentence."
)


# --------------------------------------------------------------------------- #
# State                                                                       #
# --------------------------------------------------------------------------- #


def _merge_steps(
    left: list[tuple[str, Any]] | None,
    right: list[tuple[str, Any]] | None,
) -> list[tuple[str, Any]]:
    return (left or []) + (right or [])


class PanelState(TypedDict, total=False):
    """Shared state across the panel graph."""

    question: str
    technical_brief: str
    fundamental_brief: str
    sentiment: dict[str, Any]
    final_answer: str
    intermediate_steps: Annotated[list[tuple[str, Any]], _merge_steps]


# --------------------------------------------------------------------------- #
# Worker construction                                                         #
# --------------------------------------------------------------------------- #


def _make_worker(name: str, system_prompt: str, tools: list) -> AgentExecutor:
    """Build a focused ReAct worker with a fixed system prompt + tool subset."""
    llm = _build_llm()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )
    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
        max_iterations=4,
    )
    executor.name = name  # type: ignore[attr-defined]
    return executor


def _label_steps(label: str, steps: list[Any]) -> list[tuple[str, Any]]:
    return [(label, s) for s in steps]


# --------------------------------------------------------------------------- #
# Nodes                                                                       #
# --------------------------------------------------------------------------- #


def _technical_node(state: PanelState) -> dict[str, Any]:
    worker = _make_worker(
        name="technical_analyst",
        system_prompt=TECHNICAL_PROMPT,
        tools=[get_stock_price, *TECHNICAL_TOOLS],
    )
    try:
        result = worker.invoke({"input": state["question"]})
        return {
            "technical_brief": result.get("output", "").strip(),
            "intermediate_steps": _label_steps(
                "Technical Analyst", result.get("intermediate_steps", [])
            ),
        }
    except Exception as exc:
        logger.exception("Technical worker failed")
        return {
            "technical_brief": f"_Technical Analyst failed: {exc}_",
            "intermediate_steps": [],
        }


def _fundamental_node(state: PanelState) -> dict[str, Any]:
    worker = _make_worker(
        name="fundamental_researcher",
        system_prompt=FUNDAMENTAL_PROMPT,
        tools=[search_market_news, *FUNDAMENTAL_TOOLS],
    )
    try:
        result = worker.invoke({"input": state["question"]})
        return {
            "fundamental_brief": result.get("output", "").strip(),
            "intermediate_steps": _label_steps(
                "Fundamental Researcher", result.get("intermediate_steps", [])
            ),
        }
    except Exception as exc:
        logger.exception("Fundamental worker failed")
        return {
            "fundamental_brief": f"_Fundamental Researcher failed: {exc}_",
            "intermediate_steps": [],
        }


def _sentiment_node(state: PanelState) -> dict[str, Any]:
    """Score the fundamental brief itself for an at-a-glance sentiment chip."""
    brief = state.get("fundamental_brief") or ""
    return {"sentiment": score_sentiment(brief)}


def _synthesizer_node(state: PanelState) -> dict[str, Any]:
    llm = _build_llm()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYNTHESIZER_PROMPT),
            (
                "human",
                "User question:\n{question}\n\n"
                "Technical Analyst notes:\n{technical_brief}\n\n"
                "Fundamental Researcher notes:\n{fundamental_brief}\n\n"
                "Aggregate news sentiment: {sentiment_label} "
                "(compound={sentiment_score})",
            ),
        ]
    )
    chain = prompt | llm
    sentiment = state.get("sentiment") or {"label": "Neutral", "compound": 0.0}
    msg = chain.invoke(
        {
            "question": state["question"],
            "technical_brief": state.get("technical_brief", "(no input)"),
            "fundamental_brief": state.get("fundamental_brief", "(no input)"),
            "sentiment_label": sentiment.get("label", "Neutral"),
            "sentiment_score": sentiment.get("compound", 0.0),
        }
    )
    answer = getattr(msg, "content", str(msg))
    return {"final_answer": answer.strip() if isinstance(answer, str) else str(answer)}


# --------------------------------------------------------------------------- #
# Graph                                                                       #
# --------------------------------------------------------------------------- #


def build_panel_graph():
    """Compile the LangGraph state machine for the expert panel."""
    g = StateGraph(PanelState)
    g.add_node("technical_analyst", _technical_node)
    g.add_node("fundamental_researcher", _fundamental_node)
    g.add_node("sentiment_scorer", _sentiment_node)
    g.add_node("synthesizer", _synthesizer_node)

    g.add_edge(START, "technical_analyst")
    g.add_edge(START, "fundamental_researcher")
    # Sentiment chip needs the fundamental brief, so it runs after that worker.
    g.add_edge("fundamental_researcher", "sentiment_scorer")
    # Synthesizer waits for both research tracks AND the sentiment node.
    g.add_edge("technical_analyst", "synthesizer")
    g.add_edge("sentiment_scorer", "synthesizer")
    g.add_edge("synthesizer", END)
    return g.compile()


def run_panel(question: str) -> dict[str, Any]:
    """Convenience helper: run the panel synchronously and return its full state."""
    graph = build_panel_graph()
    return graph.invoke({"question": question})


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "Should I buy NVDA right now?"
    out = run_panel(q)
    print("\n=== Final Answer ===\n")
    print(out.get("final_answer"))
    print("\n=== Sentiment ===\n", out.get("sentiment"))
