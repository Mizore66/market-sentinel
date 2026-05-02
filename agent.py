"""LangChain tool-calling agent for Market Sentinel.

Defaults to Groq (free tier) for cost reasons. Set ``LLM_PROVIDER=openai``
in the environment to switch to OpenAI's ``gpt-4o-mini``.
"""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv

# In LangChain 1.x the legacy ``AgentExecutor`` / ``create_tool_calling_agent``
# API lives in the ``langchain_classic`` package; in 0.3.x it lives in
# ``langchain.agents``. Try the modern location first, then fall back so the
# project works on both.
try:
    from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
except ImportError:  # pragma: no cover - exercised on langchain<1.0
    from langchain.agents import AgentExecutor, create_tool_calling_agent

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tools import ALL_TOOLS as BASE_TOOLS

load_dotenv()


def _all_tools() -> list:
    """Combine base tools (price + news) with the extension tools.

    Extras (SEC 10-K RAG, sentiment, Reddit buzz, uploaded-doc RAG, technical
    history) are imported lazily so the base agent still works if optional
    deps are missing.
    """
    try:
        from tools_extra import EXTRA_TOOLS

        return [*BASE_TOOLS, *EXTRA_TOOLS]
    except Exception:  # pragma: no cover
        return list(BASE_TOOLS)


SYSTEM_PROMPT = (
    "You are Market Sentinel, an expert financial AI analyst.\n"
    "You MUST use the provided tools to fetch real-time data before answering "
    "user queries about prices, fundamentals, or news.\n"
    "Tool selection rules:\n"
    "1. Numeric quotes -> `get_stock_price`. Never invent numbers.\n"
    "2. Trend / volatility / drawdown questions -> `get_price_history`.\n"
    "3. Breaking headlines -> `search_market_news`.\n"
    "4. Aggregate news vibe (positive / neutral / negative) -> `get_market_sentiment`.\n"
    "5. Retail / Reddit chatter -> `get_reddit_buzz`.\n"
    "6. Risks, business segments, MD&A, anything an annual report covers -> `query_sec_10k`.\n"
    "7. The user uploaded a document -> `query_uploaded_document`.\n"
    "If a tool returns an error or empty result, say so explicitly. Do NOT "
    "fabricate data. Keep answers direct, use short bullets, cite tickers, and "
    "preserve any source URLs that the tools returned."
)


def _build_llm() -> BaseChatModel:
    """Instantiate the chat model based on env vars.

    Default = Groq (free tier, cheapest). Set ``LLM_PROVIDER=openai`` to
    switch to ``gpt-4o-mini``.
    """
    provider = os.getenv("LLM_PROVIDER", "groq").strip().lower()

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0,
        )

    from langchain_groq import ChatGroq

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Get a free key at "
            "https://console.groq.com/keys and add it to your .env file."
        )

    return ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        temperature=0,
        api_key=api_key,
    )


def build_agent_executor(llm: BaseChatModel | None = None) -> AgentExecutor:
    """Compose the prompt, LLM, and tools into an ``AgentExecutor``."""
    chat_model = llm or _build_llm()
    tools = _all_tools()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )

    agent = create_tool_calling_agent(chat_model, tools, prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
        max_iterations=6,
    )


def run_query(question: str, **kwargs: Any) -> dict[str, Any]:
    """Convenience helper for scripts and tests."""
    executor = build_agent_executor()
    return executor.invoke({"input": question, **kwargs})


if __name__ == "__main__":
    import sys

    user_question = " ".join(sys.argv[1:]) or "What is the current price of NVDA?"
    result = run_query(user_question)
    print("\n=== Final Answer ===")
    print(result.get("output"))
    print("\n=== Tool Calls ===")
    for action, observation in result.get("intermediate_steps", []):
        print(f"- {action.tool}({action.tool_input}) -> {str(observation)[:160]}...")
