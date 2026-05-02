"""Extension tools for the Market Sentinel agent.

Adds, on top of the base ``tools.py``:

- ``get_price_history`` - quick technical snapshot (returns + volatility)
- ``query_sec_10k``     - Retrieval-Augmented Generation over the latest 10-K
                          from SEC EDGAR (free public API)
- ``query_uploaded_document`` - RAG over a PDF the user uploaded in the UI
- ``get_market_sentiment`` - VADER sentiment score over recent news headlines
- ``get_reddit_buzz``    - top mentions of a ticker on r/wallstreetbets / r/stocks
                          with a sentiment score per post

All tools are free / zero-auth. SEC EDGAR requires only a custom User-Agent.
"""

from __future__ import annotations

import json
import logging
import os
import time
from functools import lru_cache
from statistics import StatisticsError, mean, stdev
from typing import Any

import requests
from langchain_core.tools import tool

from rag import (
    UPLOADED_DOC,
    DocumentIndex,
    extract_text_from_html,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# HTTP                                                                        #
# --------------------------------------------------------------------------- #


def _user_agent() -> str:
    """SEC requires a contact-style User-Agent for programmatic access."""
    return os.getenv(
        "SEC_USER_AGENT",
        "Market Sentinel Research market-sentinel@example.com",
    )


def _http_get(url: str, *, timeout: int = 20, accept: str = "application/json") -> requests.Response:
    headers = {
        "User-Agent": _user_agent(),
        "Accept": accept,
        "Accept-Encoding": "gzip, deflate",
    }
    return requests.get(url, headers=headers, timeout=timeout)


# --------------------------------------------------------------------------- #
# Technical: price history snapshot                                           #
# --------------------------------------------------------------------------- #


@tool
def get_price_history(ticker: str, period: str = "3mo") -> str:
    """Return a compact technical snapshot of recent price action.

    Args:
        ticker: Ticker symbol (e.g. ``"AAPL"``).
        period: yfinance period code. Common values: ``"1mo"``, ``"3mo"``,
            ``"6mo"``, ``"1y"``, ``"5y"``.

    Returns:
        JSON string with first/last close, total return %, annualized
        volatility, max drawdown, average daily volume, and the number of
        trading days observed. Returns an ``error`` field if the ticker
        has no data.
    """
    if not ticker or not isinstance(ticker, str):
        return json.dumps({"error": "ticker must be a non-empty string"})

    symbol = ticker.strip().upper()
    try:
        import yfinance as yf

        history = yf.Ticker(symbol).history(period=period, auto_adjust=True)
        if history.empty:
            return json.dumps(
                {"ticker": symbol, "error": f"No history found for '{symbol}'."}
            )

        closes = history["Close"].dropna().tolist()
        volumes = history["Volume"].dropna().tolist()
        if len(closes) < 2:
            return json.dumps(
                {"ticker": symbol, "error": "Not enough data points to compute stats."}
            )

        first, last = float(closes[0]), float(closes[-1])
        total_return = (last / first - 1.0) * 100.0
        daily_returns = [
            (closes[i] / closes[i - 1] - 1.0) for i in range(1, len(closes))
        ]
        try:
            vol = stdev(daily_returns) * (252 ** 0.5) * 100.0
        except StatisticsError:
            vol = 0.0

        running_max = closes[0]
        max_dd = 0.0
        for c in closes:
            running_max = max(running_max, c)
            dd = (c / running_max - 1.0) * 100.0
            max_dd = min(max_dd, dd)

        payload = {
            "ticker": symbol,
            "period": period,
            "trading_days": len(closes),
            "first_close": round(first, 4),
            "last_close": round(last, 4),
            "total_return_pct": round(total_return, 2),
            "annualized_volatility_pct": round(vol, 2),
            "max_drawdown_pct": round(max_dd, 2),
            "avg_daily_volume": int(mean(volumes)) if volumes else None,
        }
        return json.dumps(payload)
    except Exception as exc:
        return json.dumps({"ticker": symbol, "error": f"Failed: {exc!s}"})


# --------------------------------------------------------------------------- #
# SEC EDGAR: 10-K RAG                                                          #
# --------------------------------------------------------------------------- #


_SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"


@lru_cache(maxsize=1)
def _ticker_to_cik_table() -> dict[str, str]:
    """Download (and cache) SEC's ticker -> 10-digit CIK map."""
    resp = _http_get(_SEC_TICKERS_URL)
    resp.raise_for_status()
    raw = resp.json()
    out: dict[str, str] = {}
    for _, row in raw.items():
        ticker = str(row["ticker"]).upper()
        cik = str(row["cik_str"]).zfill(10)
        out[ticker] = cik
    return out


def _resolve_cik(ticker: str) -> str:
    table = _ticker_to_cik_table()
    cik = table.get(ticker.upper())
    if not cik:
        raise LookupError(f"No SEC CIK found for ticker '{ticker}'.")
    return cik


def _latest_10k_metadata(cik: str) -> dict[str, Any]:
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    resp = _http_get(url)
    resp.raise_for_status()
    submissions = resp.json()
    recent = submissions.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accession_numbers = recent.get("accessionNumber", [])
    primary_documents = recent.get("primaryDocument", [])
    filing_dates = recent.get("filingDate", [])
    for i, form in enumerate(forms):
        if form == "10-K":
            accession = accession_numbers[i].replace("-", "")
            return {
                "accession": accession_numbers[i],
                "filing_date": filing_dates[i],
                "primary_document": primary_documents[i],
                "url": (
                    f"https://www.sec.gov/Archives/edgar/data/"
                    f"{int(cik)}/{accession}/{primary_documents[i]}"
                ),
                "company": submissions.get("name", ""),
            }
    raise LookupError("No 10-K found in recent filings.")


@lru_cache(maxsize=8)
def _load_10k_index(ticker: str) -> tuple[DocumentIndex, dict[str, Any]]:
    """Download + chunk + index the latest 10-K for ``ticker`` (cached)."""
    cik = _resolve_cik(ticker)
    meta = _latest_10k_metadata(cik)
    # Polite pause: SEC asks for <=10 req/sec.
    time.sleep(0.25)
    resp = _http_get(meta["url"], accept="text/html")
    resp.raise_for_status()
    text = extract_text_from_html(resp.text)
    if not text:
        raise RuntimeError("Failed to extract text from the 10-K HTML.")
    index = DocumentIndex.from_text(text, label=f"{ticker.upper()} 10-K {meta['filing_date']}")
    return index, meta


@tool
def query_sec_10k(ticker: str, question: str) -> str:
    """Search the most recent 10-K filing for ``ticker`` and return relevant excerpts.

    Use this when the user asks about a company's risks, business model,
    revenue segments, management discussion, or any other topic typically
    disclosed in an annual report.

    Args:
        ticker: Stock ticker (e.g. ``"AAPL"``, ``"MSFT"``).
        question: Free-text question to retrieve excerpts for, e.g.
            "What are the main supply-chain risks?".

    Returns:
        A plain-text block listing the top excerpts with their relevance
        scores, plus the source URL. Returns an error string if the filing
        cannot be located or downloaded.
    """
    if not ticker or not question:
        return "error: both `ticker` and `question` are required."

    try:
        index, meta = _load_10k_index(ticker.strip().upper())
    except LookupError as exc:
        return f"No 10-K available for {ticker!r}: {exc}"
    except requests.HTTPError as exc:
        return f"SEC EDGAR request failed: {exc}"
    except Exception as exc:  # pragma: no cover - network surprises
        return f"Failed to load 10-K for {ticker!r}: {exc}"

    hits = index.search(question, top_k=4)
    if not hits:
        return (
            f"The latest 10-K for {meta['company']} ({meta['filing_date']}) "
            f"contains nothing closely matching: {question!r}.\n"
            f"Source: {meta['url']}"
        )

    body = "\n\n".join(h.render() for h in hits)
    return (
        f"Source: {meta['company']} 10-K filed {meta['filing_date']}\n"
        f"URL: {meta['url']}\n\n"
        f"Top excerpts for: {question!r}\n\n{body}"
    )


# --------------------------------------------------------------------------- #
# Uploaded-doc RAG                                                            #
# --------------------------------------------------------------------------- #


@tool
def query_uploaded_document(question: str) -> str:
    """Answer a question using the document the user uploaded in the UI.

    Use this when the user refers to "the document", "the report I uploaded",
    "the PDF", or asks a follow-up that clearly targets uploaded material.

    Args:
        question: Natural-language question to look up in the document.

    Returns:
        Top matching excerpts with relevance scores, or a clear message if
        the user has not uploaded anything.
    """
    if not UPLOADED_DOC.is_ready:
        return (
            "No document has been uploaded yet. Ask the user to upload a PDF "
            "in the sidebar before invoking this tool."
        )

    hits = UPLOADED_DOC.index.search(question, top_k=4)  # type: ignore[union-attr]
    if not hits:
        return (
            f"The uploaded document ({UPLOADED_DOC.source_label}) does not "
            f"contain anything relevant to {question!r}."
        )
    body = "\n\n".join(h.render() for h in hits)
    return (
        f"Source: uploaded document - {UPLOADED_DOC.source_label}\n\n"
        f"Top excerpts for: {question!r}\n\n{body}"
    )


# --------------------------------------------------------------------------- #
# Sentiment                                                                   #
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=1)
def _vader():
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    return SentimentIntensityAnalyzer()


def score_sentiment(text: str) -> dict[str, Any]:
    """Public helper used by the panel and the report writer."""
    if not text or not text.strip():
        return {"label": "Neutral", "compound": 0.0, "details": {}}
    scores = _vader().polarity_scores(text)
    compound = scores["compound"]
    if compound >= 0.05:
        label = "Positive"
    elif compound <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"
    return {"label": label, "compound": round(compound, 4), "details": scores}


@tool
def get_market_sentiment(ticker_or_topic: str) -> str:
    """Score market sentiment for a ticker or topic by analyzing fresh news headlines.

    Pulls the most recent news snippets via DuckDuckGo News and runs VADER
    sentiment on each one, returning an aggregate label
    (Positive / Neutral / Negative), a compound score in [-1, 1], and a
    breakdown by headline.

    Args:
        ticker_or_topic: e.g. ``"NVDA"``, ``"Federal Reserve"``,
            ``"semiconductor demand"``.
    """
    if not ticker_or_topic or not isinstance(ticker_or_topic, str):
        return "error: query must be a non-empty string."

    try:
        from langchain_community.utilities.duckduckgo_search import (
            DuckDuckGoSearchAPIWrapper,
        )

        wrapper = DuckDuckGoSearchAPIWrapper(source="news", time="w", max_results=10)
        results = wrapper.results(ticker_or_topic, max_results=10, source="news")
    except Exception as exc:
        return f"News fetch failed for {ticker_or_topic!r}: {exc}"

    if not results:
        return f"No recent news found for {ticker_or_topic!r}."

    per_item: list[dict[str, Any]] = []
    compounds: list[float] = []
    for r in results:
        title = r.get("title", "")
        snippet = r.get("snippet", "")
        link = r.get("link", "")
        text = f"{title}. {snippet}".strip()
        s = score_sentiment(text)
        compounds.append(s["compound"])
        per_item.append(
            {"title": title, "link": link, "label": s["label"], "compound": s["compound"]}
        )

    avg = round(sum(compounds) / len(compounds), 4) if compounds else 0.0
    if avg >= 0.05:
        agg_label = "Positive"
    elif avg <= -0.05:
        agg_label = "Negative"
    else:
        agg_label = "Neutral"

    return json.dumps(
        {
            "topic": ticker_or_topic,
            "headlines_analyzed": len(per_item),
            "aggregate_label": agg_label,
            "aggregate_compound": avg,
            "headlines": per_item,
        },
        default=str,
    )


# --------------------------------------------------------------------------- #
# Reddit buzz (no auth required)                                              #
# --------------------------------------------------------------------------- #


_REDDIT_SUBS = ("wallstreetbets", "stocks", "investing")


@tool
def get_reddit_buzz(ticker: str, max_posts: int = 10) -> str:
    """Pull recent Reddit posts mentioning ``ticker`` and score their sentiment.

    Searches r/wallstreetbets, r/stocks, and r/investing via Reddit's public
    JSON endpoints (no auth required). Returns the top posts with title,
    permalink, score, and a per-post VADER sentiment label.
    """
    if not ticker or not isinstance(ticker, str):
        return "error: ticker must be a non-empty string."

    headers = {"User-Agent": "MarketSentinel/1.0 (research)"}
    posts: list[dict[str, Any]] = []
    seen: set[str] = set()

    for sub in _REDDIT_SUBS:
        url = (
            f"https://www.reddit.com/r/{sub}/search.json"
            f"?q={requests.utils.quote(ticker)}&restrict_sr=on&sort=new&t=week&limit={max_posts}"
        )
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            if resp.status_code != 200:
                continue
            data = resp.json().get("data", {}).get("children", [])
        except Exception as exc:
            logger.debug("Reddit fetch failed for r/%s: %s", sub, exc)
            continue

        for child in data:
            d = child.get("data", {})
            permalink = d.get("permalink", "")
            if permalink in seen:
                continue
            seen.add(permalink)
            title = d.get("title", "")
            body = d.get("selftext", "") or ""
            sentiment = score_sentiment(f"{title}. {body[:500]}")
            posts.append(
                {
                    "subreddit": sub,
                    "title": title,
                    "score": d.get("score", 0),
                    "num_comments": d.get("num_comments", 0),
                    "permalink": f"https://reddit.com{permalink}",
                    "sentiment_label": sentiment["label"],
                    "sentiment_compound": sentiment["compound"],
                }
            )

    if not posts:
        return f"No recent Reddit posts found mentioning {ticker!r}."

    posts.sort(key=lambda p: p["score"], reverse=True)
    posts = posts[:max_posts]
    compounds = [p["sentiment_compound"] for p in posts]
    avg = round(sum(compounds) / len(compounds), 4)
    if avg >= 0.05:
        agg = "Positive"
    elif avg <= -0.05:
        agg = "Negative"
    else:
        agg = "Neutral"

    return json.dumps(
        {
            "ticker": ticker.upper(),
            "posts_analyzed": len(posts),
            "aggregate_label": agg,
            "aggregate_compound": avg,
            "posts": posts,
        },
        default=str,
    )


# --------------------------------------------------------------------------- #
# Tool catalog                                                                #
# --------------------------------------------------------------------------- #


TECHNICAL_TOOLS = [
    get_price_history,
]
FUNDAMENTAL_TOOLS = [
    query_sec_10k,
    query_uploaded_document,
    get_market_sentiment,
    get_reddit_buzz,
]
EXTRA_TOOLS = [
    get_price_history,
    query_sec_10k,
    query_uploaded_document,
    get_market_sentiment,
    get_reddit_buzz,
]
