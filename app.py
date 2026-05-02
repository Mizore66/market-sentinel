"""Streamlit chat UI for Market Sentinel.

Renders a chat interface with two modes:

- **Single Agent**: classic ReAct agent calling all tools.
- **Expert Panel**: LangGraph multi-agent (Technical Analyst + Fundamental
  Researcher + Synthesizer) running in parallel.

Sidebar features:
- LLM info, tools, and clear-history.
- PDF uploader: indexes the doc with TF-IDF so the agent can RAG over it.
- Sentiment chip on the latest panel answer.
- Executive brief download (Markdown + PDF) generated from the last answer.
"""

from __future__ import annotations

import os
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

from agent import build_agent_executor
from panel import build_panel_graph
from rag import UPLOADED_DOC, extract_text_from_pdf_bytes
from report import build_markdown_brief, build_pdf_brief

load_dotenv()

st.set_page_config(
    page_title="Market Sentinel",
    page_icon="*",
    layout="wide",
)


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _provider_label() -> str:
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    if provider == "openai":
        return f"OpenAI - {os.getenv('OPENAI_MODEL', 'gpt-4o-mini')}"
    return f"Groq - {os.getenv('GROQ_MODEL', 'llama-3.3-70b-versatile')} (free tier)"


def _api_key_present() -> bool:
    if os.getenv("LLM_PROVIDER", "groq").lower() == "openai":
        return bool(os.getenv("OPENAI_API_KEY"))
    return bool(os.getenv("GROQ_API_KEY"))


@st.cache_resource(show_spinner="Spinning up Market Sentinel agent...")
def _get_single_agent_executor() -> Any:
    return build_agent_executor()


@st.cache_resource(show_spinner="Compiling expert-panel graph...")
def _get_panel_graph() -> Any:
    return build_panel_graph()


def _sentiment_emoji(label: str) -> str:
    return {"Positive": "+", "Negative": "-", "Neutral": "="}.get(label, "?")


# --------------------------------------------------------------------------- #
# Session state                                                               #
# --------------------------------------------------------------------------- #


if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_run" not in st.session_state:
    st.session_state.last_run = None  # holds the brief payload after each turn
if "uploaded_doc_meta" not in st.session_state:
    st.session_state.uploaded_doc_meta = None


# --------------------------------------------------------------------------- #
# Sidebar                                                                     #
# --------------------------------------------------------------------------- #


with st.sidebar:
    st.markdown("## Market Sentinel")
    st.caption("Autonomous financial intelligence agent")
    st.markdown("---")

    mode = st.radio(
        "Agent mode",
        options=["Single Agent", "Expert Panel"],
        index=0,
        help=(
            "**Single Agent** uses one ReAct loop over all tools (faster, "
            "cheaper).\n\n"
            "**Expert Panel** uses a LangGraph multi-agent: a Technical "
            "Analyst and a Fundamental Researcher work in parallel and a "
            "Synthesizer combines their notes."
        ),
    )

    st.markdown("**LLM**")
    st.code(_provider_label(), language="text")

    st.markdown("**Available tools**")
    st.markdown(
        "- `get_stock_price` (yfinance)\n"
        "- `get_price_history` (yfinance)\n"
        "- `search_market_news` (DuckDuckGo)\n"
        "- `get_market_sentiment` (VADER + DDG news)\n"
        "- `get_reddit_buzz` (public Reddit JSON)\n"
        "- `query_sec_10k` (SEC EDGAR + TF-IDF RAG)\n"
        "- `query_uploaded_document` (TF-IDF RAG)\n"
    )

    st.markdown("---")
    st.markdown("### Document RAG")
    uploaded = st.file_uploader(
        "Upload an annual report or any PDF",
        type=["pdf"],
        help="The agent can search inside this document via the "
             "`query_uploaded_document` tool.",
    )
    if uploaded is not None:
        if (
            st.session_state.uploaded_doc_meta is None
            or st.session_state.uploaded_doc_meta.get("name") != uploaded.name
        ):
            with st.spinner(f"Indexing {uploaded.name}..."):
                try:
                    text = extract_text_from_pdf_bytes(uploaded.getvalue())
                    n_chunks = UPLOADED_DOC.set_from_text(text, label=uploaded.name)
                    st.session_state.uploaded_doc_meta = {
                        "name": uploaded.name,
                        "chunks": n_chunks,
                        "chars": len(text),
                    }
                except Exception as exc:
                    st.error(f"Could not parse PDF: {exc}")
                    UPLOADED_DOC.clear()
                    st.session_state.uploaded_doc_meta = None
    if st.session_state.uploaded_doc_meta:
        meta = st.session_state.uploaded_doc_meta
        st.success(
            f"Indexed **{meta['name']}** "
            f"({meta['chunks']} chunks, {meta['chars']:,} chars)"
        )
        if st.button("Clear uploaded doc", use_container_width=True):
            UPLOADED_DOC.clear()
            st.session_state.uploaded_doc_meta = None
            st.rerun()

    st.markdown("---")
    if st.button("Clear chat history", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_run = None
        st.rerun()


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #


st.title("Market Sentinel")
st.caption(
    "A real-time financial intelligence agent. Ask about prices, news, "
    "filings, or social sentiment - watch it think, then export the brief."
)

if not _api_key_present():
    st.error(
        "No LLM API key detected. Copy `.env.example` to `.env` and add your "
        "free Groq API key from https://console.groq.com/keys, then refresh."
    )
    st.stop()


# Replay history.
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input(
    "Ask Market Sentinel about a stock, ETF, filing, or market event..."
)

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        thought_container = st.container()
        callback = StreamlitCallbackHandler(
            thought_container,
            expand_new_thoughts=True,
            collapse_completed_thoughts=False,
        )

        run_payload: dict[str, Any] = {
            "question": prompt,
            "mode": mode,
        }

        try:
            if mode == "Single Agent":
                executor = _get_single_agent_executor()
                result = executor.invoke(
                    {"input": prompt},
                    config={"callbacks": [callback]},
                )
                answer = result.get("output", "(no response)")
                run_payload.update(
                    {
                        "answer": answer,
                        "intermediate_steps": result.get("intermediate_steps", []),
                    }
                )
            else:
                graph = _get_panel_graph()
                with thought_container.status(
                    "Expert panel deliberating...", expanded=True
                ) as status:
                    status.write("Spawning Technical Analyst and Fundamental "
                                 "Researcher in parallel...")
                    state = graph.invoke(
                        {"question": prompt},
                        config={"callbacks": [callback]},
                    )
                    status.update(label="Synthesis complete", state="complete")

                answer = state.get("final_answer", "(no response)")
                sentiment = state.get("sentiment") or {}
                if sentiment:
                    st.info(
                        f"**News sentiment:** {_sentiment_emoji(sentiment.get('label','Neutral'))} "
                        f"{sentiment.get('label','Neutral')} "
                        f"(compound={sentiment.get('compound',0)})"
                    )
                with st.expander("Technical Analyst notes"):
                    st.markdown(state.get("technical_brief", "_no notes_"))
                with st.expander("Fundamental Researcher notes"):
                    st.markdown(state.get("fundamental_brief", "_no notes_"))

                run_payload.update(
                    {
                        "answer": answer,
                        "sentiment": sentiment,
                        "technical_brief": state.get("technical_brief"),
                        "fundamental_brief": state.get("fundamental_brief"),
                        "intermediate_steps": state.get("intermediate_steps", []),
                    }
                )
        except Exception as exc:
            answer = f"Sorry, the agent failed: `{exc}`"
            run_payload["answer"] = answer

        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.last_run = run_payload


# --------------------------------------------------------------------------- #
# Executive brief export                                                      #
# --------------------------------------------------------------------------- #


if st.session_state.last_run:
    st.markdown("---")
    st.subheader("Executive Brief Export")
    col_md, col_pdf, _ = st.columns([1, 1, 3])

    payload = st.session_state.last_run
    md_text = build_markdown_brief(
        question=payload.get("question", ""),
        answer=payload.get("answer", ""),
        mode=payload.get("mode", "Single Agent"),
        sentiment=payload.get("sentiment"),
        technical_brief=payload.get("technical_brief"),
        fundamental_brief=payload.get("fundamental_brief"),
        intermediate_steps=payload.get("intermediate_steps"),
    )

    with col_md:
        st.download_button(
            label="Download Markdown",
            data=md_text.encode("utf-8"),
            file_name="market-sentinel-brief.md",
            mime="text/markdown",
            use_container_width=True,
        )

    with col_pdf:
        try:
            pdf_bytes = build_pdf_brief(
                question=payload.get("question", ""),
                answer=payload.get("answer", ""),
                mode=payload.get("mode", "Single Agent"),
                sentiment=payload.get("sentiment"),
                technical_brief=payload.get("technical_brief"),
                fundamental_brief=payload.get("fundamental_brief"),
                intermediate_steps=payload.get("intermediate_steps"),
            )
            st.download_button(
                label="Download PDF",
                data=pdf_bytes,
                file_name="market-sentinel-brief.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception as exc:
            st.warning(f"PDF generation failed: {exc}")
