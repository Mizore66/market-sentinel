"""Streamlit chat UI for Market Sentinel.

Renders a chat interface and uses ``StreamlitCallbackHandler`` to expose the
agent's intermediate reasoning (tool calls, arguments, observations) in a
collapsible "thought process" expander, making the agentic behaviour visible
during a live demo.
"""

from __future__ import annotations

import os
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

from agent import build_agent_executor

load_dotenv()

st.set_page_config(
    page_title="Market Sentinel",
    page_icon="*",
    layout="wide",
)


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
def _get_agent_executor() -> Any:
    return build_agent_executor()


with st.sidebar:
    st.markdown("## Market Sentinel")
    st.caption("Autonomous financial intelligence agent")
    st.markdown("---")
    st.markdown("**LLM**")
    st.code(_provider_label(), language="text")
    st.markdown("**Tools**")
    st.markdown(
        "- `get_stock_price` (yfinance)\n"
        "- `search_market_news` (DuckDuckGo)"
    )
    st.markdown("---")
    if st.button("Clear chat history", use_container_width=True):
        st.session_state.pop("messages", None)
        st.rerun()
    st.caption(
        "Tip: ask things like 'What's the price of NVDA?' or "
        "'Latest news on the Federal Reserve rate decision'."
    )


st.title("Market Sentinel")
st.caption(
    "A ReAct agent that fetches real-time market data and news before answering."
)

if not _api_key_present():
    st.error(
        "No LLM API key detected. Copy `.env.example` to `.env` and add your "
        "free Groq API key from https://console.groq.com/keys, then refresh."
    )
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask Market Sentinel about a stock, ETF, or market event...")

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

        try:
            executor = _get_agent_executor()
            result = executor.invoke(
                {"input": prompt},
                config={"callbacks": [callback]},
            )
            answer = result.get("output", "(no response)")
        except Exception as exc:
            answer = f"Sorry, the agent failed: `{exc}`"

        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
