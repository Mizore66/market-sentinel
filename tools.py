"""Tool definitions for the Market Sentinel agent.

Each tool is a thin, deterministic wrapper around a free data source so the LLM
can fetch real-time facts instead of hallucinating them.
"""

from __future__ import annotations

import json
from typing import Any

import yfinance as yf
from langchain_core.tools import tool


def _safe_get(info: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        value = info.get(key)
        if value not in (None, "", 0):
            return value
    return default


@tool
def get_stock_price(ticker: str) -> str:
    """Fetch the latest market price and key fundamentals for a stock or ETF ticker.

    Args:
        ticker: The exchange ticker symbol (e.g. "AAPL", "MSFT", "SPY").

    Returns:
        A JSON string with current price, currency, day range, 52-week range,
        market cap, P/E ratio, and the long company name. If the ticker is
        invalid or no data is available, returns a JSON object with an
        ``error`` field so the agent can react gracefully instead of
        hallucinating a number.
    """
    if not ticker or not isinstance(ticker, str):
        return json.dumps({"error": "ticker must be a non-empty string"})

    symbol = ticker.strip().upper()

    try:
        asset = yf.Ticker(symbol)
        info: dict[str, Any] = {}
        try:
            info = asset.info or {}
        except Exception:
            info = {}

        price = _safe_get(
            info,
            "regularMarketPrice",
            "currentPrice",
            "previousClose",
        )

        if price is None:
            history = asset.history(period="5d")
            if not history.empty:
                price = float(history["Close"].iloc[-1])

        if price is None:
            return json.dumps(
                {
                    "ticker": symbol,
                    "error": (
                        f"No market data found for ticker '{symbol}'. "
                        "It may be delisted or invalid."
                    ),
                }
            )

        payload = {
            "ticker": symbol,
            "name": _safe_get(info, "longName", "shortName", default=symbol),
            "price": round(float(price), 4),
            "currency": _safe_get(info, "currency", default="USD"),
            "previous_close": _safe_get(info, "previousClose"),
            "day_low": _safe_get(info, "dayLow"),
            "day_high": _safe_get(info, "dayHigh"),
            "fifty_two_week_low": _safe_get(info, "fiftyTwoWeekLow"),
            "fifty_two_week_high": _safe_get(info, "fiftyTwoWeekHigh"),
            "market_cap": _safe_get(info, "marketCap"),
            "pe_ratio": _safe_get(info, "trailingPE", "forwardPE"),
            "exchange": _safe_get(info, "fullExchangeName", "exchange"),
        }
        return json.dumps(payload, default=str)

    except Exception as exc:
        return json.dumps(
            {
                "ticker": symbol,
                "error": f"Failed to retrieve data for '{symbol}': {exc!s}",
            }
        )


@tool
def search_market_news(query: str) -> str:
    """Search the web for the latest market or financial news on a topic.

    Use this when the user asks about news, sentiment, earnings,
    macroeconomic events, or any fact that is not a numeric quote
    (numeric quotes should use ``get_stock_price`` instead).

    Args:
        query: A free-text search query, e.g. "Tesla Q3 earnings" or
            "Federal Reserve rate decision today".

    Returns:
        A short, plain-text digest of the top news snippets, or a clear
        message if no results were returned.
    """
    if not query or not isinstance(query, str):
        return "error: query must be a non-empty string"

    try:
        from langchain_community.tools import DuckDuckGoSearchRun
        from langchain_community.utilities.duckduckgo_search import (
            DuckDuckGoSearchAPIWrapper,
        )

        # ``source="news"`` + weekly window fits market-news questions better than
        # plain web snippets. Requires the ``ddgs`` PyPI package (see requirements.txt).
        search = DuckDuckGoSearchRun(
            api_wrapper=DuckDuckGoSearchAPIWrapper(
                source="news",
                time="w",
                max_results=8,
            )
        )
        result = search.invoke(query)
        if not result or not result.strip():
            return f"No news results found for query: {query!r}"
        return result
    except Exception as exc:
        return f"News search failed for query {query!r}: {exc!s}"


ALL_TOOLS = [get_stock_price, search_market_news]
