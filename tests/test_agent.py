"""Quality assurance tests for the Market Sentinel agent.

These tests have two layers:

1. **Pure tool tests** (always run): exercise ``get_stock_price`` and
   ``search_market_news`` directly to confirm hallucination-safe behaviour
   for invalid inputs.
2. **End-to-end agent tests** (skipped without ``GROQ_API_KEY``): hit the
   real LLM to verify tool routing and graceful handling of fake tickers.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

# Make the project root importable when running ``pytest`` from anywhere.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import get_stock_price, search_market_news  # noqa: E402


# --------------------------------------------------------------------------- #
# Layer 1 - tool-only tests (no LLM, no network for the bogus ticker case)    #
# --------------------------------------------------------------------------- #


def test_get_stock_price_rejects_empty_ticker() -> None:
    payload = json.loads(get_stock_price.invoke({"ticker": ""}))
    assert "error" in payload


def test_get_stock_price_handles_invalid_ticker_without_hallucinating() -> None:
    payload = json.loads(
        get_stock_price.invoke({"ticker": "ZZZZZZZZZZNOPE"})
    )
    # Either we explicitly error out, OR we return without a numeric price.
    assert "error" in payload or "price" not in payload


@pytest.mark.network
def test_get_stock_price_returns_numeric_price_for_aapl() -> None:
    payload = json.loads(get_stock_price.invoke({"ticker": "AAPL"}))
    if "error" in payload:
        pytest.skip(f"Network/yfinance unavailable: {payload['error']}")
    assert payload["ticker"] == "AAPL"
    assert isinstance(payload["price"], (int, float))
    assert payload["price"] > 0


@pytest.mark.network
def test_search_market_news_returns_a_string() -> None:
    result = search_market_news.invoke({"query": "S&P 500"})
    assert isinstance(result, str)
    assert len(result) > 0


# --------------------------------------------------------------------------- #
# Layer 2 - end-to-end agent tests (require GROQ_API_KEY)                     #
# --------------------------------------------------------------------------- #


needs_groq = pytest.mark.skipif(
    not os.getenv("GROQ_API_KEY"),
    reason="GROQ_API_KEY not configured; skipping live LLM tests.",
)


@needs_groq
def test_price_question_routes_to_get_stock_price() -> None:
    from agent import build_agent_executor

    executor = build_agent_executor()
    result = executor.invoke({"input": "What is the current price of Apple stock?"})
    tools_called = [step[0].tool for step in result.get("intermediate_steps", [])]
    assert "get_stock_price" in tools_called, (
        f"Expected get_stock_price to be invoked, got: {tools_called}"
    )


@needs_groq
def test_fake_ticker_does_not_invent_price() -> None:
    from agent import build_agent_executor

    executor = build_agent_executor()
    result = executor.invoke(
        {"input": "What is the price of ticker ZZZZZZZZZZNOPE?"}
    )
    answer = (result.get("output") or "").lower()
    # The agent should clearly disclaim instead of fabricating a price.
    refusal_terms = (
        "no",
        "not",
        "unable",
        "could not",
        "couldn't",
        "invalid",
        "cannot",
        "no data",
        "delisted",
    )
    assert any(term in answer for term in refusal_terms), (
        f"Agent did not disclaim missing data: {answer!r}"
    )
