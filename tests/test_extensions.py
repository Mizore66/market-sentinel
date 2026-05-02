"""Tests for the four extensions: multi-agent panel, 10-K RAG, sentiment, brief export."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag import (  # noqa: E402
    UPLOADED_DOC,
    DocumentIndex,
    chunk_text,
    extract_text_from_html,
)
from report import build_markdown_brief, build_pdf_brief  # noqa: E402
from tools_extra import (  # noqa: E402
    EXTRA_TOOLS,
    FUNDAMENTAL_TOOLS,
    TECHNICAL_TOOLS,
    get_market_sentiment,
    get_price_history,
    query_uploaded_document,
    score_sentiment,
)


# --------------------------------------------------------------------------- #
# RAG core                                                                    #
# --------------------------------------------------------------------------- #


def test_chunk_text_splits_on_paragraphs() -> None:
    text = "\n\n".join([f"Paragraph {i} contains useful information." for i in range(20)])
    chunks = chunk_text(text, target_chars=200, overlap_chars=20)
    assert len(chunks) > 1
    assert all(c.strip() for c in chunks)


def test_document_index_retrieves_relevant_chunk() -> None:
    chunks = [
        "Apple Inc. designs and manufactures smartphones, including the iPhone.",
        "Tesla Inc. produces electric vehicles such as the Model 3 and Model Y.",
        "NVIDIA Corporation builds graphics processing units used for AI training.",
        "Microsoft Corporation develops the Windows operating system and Azure cloud.",
    ]
    idx = DocumentIndex(chunks, label="companies")
    hits = idx.search("Which company makes GPUs for AI?", top_k=2)
    assert hits, "expected at least one hit"
    assert "NVIDIA" in hits[0].text


def test_extract_text_from_html_strips_tags() -> None:
    html = "<html><body><h1>Risks</h1><p>Foreign currency.</p><script>x</script></body></html>"
    text = extract_text_from_html(html)
    assert "Risks" in text
    assert "Foreign currency" in text
    assert "<" not in text


def test_uploaded_doc_store_roundtrip() -> None:
    text = (
        "The company faces significant supply chain risk in Asia.\n\n"
        "Revenue grew 18% year over year driven by services.\n\n"
        "Management discussed inflationary pressures on margins."
    )
    n_chunks = UPLOADED_DOC.set_from_text(text, label="test-doc.pdf")
    try:
        assert n_chunks >= 1
        assert UPLOADED_DOC.is_ready
        result = query_uploaded_document.invoke({"question": "supply chain"})
        assert "supply chain" in result.lower() or "asia" in result.lower()
    finally:
        UPLOADED_DOC.clear()
    assert not UPLOADED_DOC.is_ready


def test_query_uploaded_document_handles_missing_doc() -> None:
    UPLOADED_DOC.clear()
    result = query_uploaded_document.invoke({"question": "anything"})
    assert "no document" in result.lower()


# --------------------------------------------------------------------------- #
# Sentiment                                                                   #
# --------------------------------------------------------------------------- #


def test_score_sentiment_labels() -> None:
    pos = score_sentiment("Wonderful results, fantastic outlook, profits soaring.")
    neg = score_sentiment("Terrible loss, awful management, stock plummeted in disaster.")
    neu = score_sentiment("The company filed paperwork with the regulator today.")
    assert pos["label"] == "Positive"
    assert neg["label"] == "Negative"
    assert neu["label"] == "Neutral"
    assert -1.0 <= pos["compound"] <= 1.0


def test_score_sentiment_handles_empty() -> None:
    out = score_sentiment("")
    assert out["label"] == "Neutral"
    assert out["compound"] == 0.0


# --------------------------------------------------------------------------- #
# Tool catalogs                                                               #
# --------------------------------------------------------------------------- #


def test_tool_catalogs_are_populated() -> None:
    assert len(TECHNICAL_TOOLS) >= 1
    assert len(FUNDAMENTAL_TOOLS) >= 3
    assert len(EXTRA_TOOLS) >= 4


# --------------------------------------------------------------------------- #
# Technical history (network-dependent)                                       #
# --------------------------------------------------------------------------- #


@pytest.mark.network
def test_get_price_history_returns_stats_for_aapl() -> None:
    payload = json.loads(get_price_history.invoke({"ticker": "AAPL", "period": "1mo"}))
    if "error" in payload:
        pytest.skip(f"yfinance unavailable: {payload['error']}")
    assert payload["ticker"] == "AAPL"
    assert payload["trading_days"] >= 5
    assert isinstance(payload["total_return_pct"], (int, float))
    assert isinstance(payload["annualized_volatility_pct"], (int, float))


@pytest.mark.network
def test_get_market_sentiment_returns_aggregate() -> None:
    raw = get_market_sentiment.invoke({"ticker_or_topic": "S&P 500"})
    if raw.startswith("News fetch failed") or raw.startswith("No recent news"):
        pytest.skip(raw)
    payload = json.loads(raw)
    assert payload["aggregate_label"] in {"Positive", "Negative", "Neutral"}
    assert -1.0 <= payload["aggregate_compound"] <= 1.0
    assert payload["headlines_analyzed"] >= 1


@pytest.mark.network
def test_query_sec_10k_pulls_real_filing() -> None:
    from tools_extra import query_sec_10k

    out = query_sec_10k.invoke(
        {"ticker": "AAPL", "question": "What are the main risk factors?"}
    )
    if out.lower().startswith(("no 10-k", "sec edgar request failed", "failed to load")):
        pytest.skip(out)
    assert "10-K" in out or "Source" in out
    assert "https://www.sec.gov" in out


# --------------------------------------------------------------------------- #
# Executive brief export                                                      #
# --------------------------------------------------------------------------- #


def _sample_payload() -> dict:
    return {
        "question": "Should I worry about NVDA's supply chain?",
        "answer": "Recent disclosures point to concentration risk. See https://example.com/source",
        "mode": "Expert Panel",
        "sentiment": {"label": "Neutral", "compound": 0.02},
        "technical_brief": "- Price up 4% over 1mo\n- Volatility 35%",
        "fundamental_brief": "- News mostly neutral\n- 10-K flags supply concentration",
        "intermediate_steps": [
            (
                "Fundamental Researcher",
                (
                    type("A", (), {"tool": "query_sec_10k", "tool_input": {"ticker": "NVDA"}})(),
                    "Excerpt about supply chain concentration in Asia.",
                ),
            )
        ],
    }


def test_build_markdown_brief_contains_sections() -> None:
    md = build_markdown_brief(**_sample_payload())
    assert "# Market Sentinel" in md
    assert "## Question" in md
    assert "## Final Answer" in md
    assert "## Technical Analyst" in md
    assert "## Fundamental Researcher" in md
    assert "## Reasoning Trail" in md
    assert "## Sources" in md
    assert "https://example.com/source" in md


def test_build_pdf_brief_returns_pdf_bytes() -> None:
    pdf = build_pdf_brief(**_sample_payload())
    assert isinstance(pdf, (bytes, bytearray))
    assert pdf.startswith(b"%PDF"), "output must be a valid PDF"
    assert len(pdf) > 500


# --------------------------------------------------------------------------- #
# Panel graph (live LLM only)                                                  #
# --------------------------------------------------------------------------- #


needs_groq = pytest.mark.skipif(
    not os.getenv("GROQ_API_KEY"),
    reason="GROQ_API_KEY not configured; skipping live panel tests.",
)


def test_panel_graph_compiles_without_invoking() -> None:
    from panel import build_panel_graph

    graph = build_panel_graph()
    assert graph is not None


@needs_groq
@pytest.mark.network
def test_panel_runs_end_to_end_with_two_briefs() -> None:
    from panel import run_panel

    state = run_panel("Give me a quick read on AAPL right now.")
    assert state.get("final_answer"), "panel must return a synthesized final answer"
    # Both workers should produce non-empty briefs.
    assert state.get("technical_brief"), "technical brief missing"
    assert state.get("fundamental_brief"), "fundamental brief missing"
    sentiment = state.get("sentiment") or {}
    assert sentiment.get("label") in {"Positive", "Negative", "Neutral"}
