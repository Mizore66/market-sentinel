# Market Sentinel

Autonomous **Financial Intelligence Agent** with two modes:

- **Single Agent** (default) - one ReAct loop over all tools.
- **Expert Panel** - a **LangGraph multi-agent** team (Technical Analyst +
  Fundamental Researcher) running in **parallel**, then a Synthesizer that
  combines their notes into one final answer.

It answers questions by deciding, on the fly, whether to fetch a quote,
search the news, score sentiment, scrape Reddit, or RAG over an SEC 10-K
filing or a PDF you uploaded - then exports the whole investigation as a
downloadable **Markdown or PDF Executive Brief**.

The frontend is a Streamlit chat that visually exposes every intermediate
tool call so you can watch the agent think.

> Built from `plans/new_plan.md`. **Cheapest options throughout**:
> Groq's free tier for the LLM, public SEC EDGAR + Reddit JSON for data,
> TF-IDF (no embedding model downloads) for RAG, VADER for sentiment,
> `fpdf2` for PDF export.

---

## Architecture

```text
                                   +------------------------+
                                   |  Streamlit Chat UI     |
                                   |  - mode toggle         |
                                   |  - PDF uploader (RAG)  |
                                   |  - Brief download      |
                                   +-----------+------------+
                                               |
                       +-----------------------+----------------------+
                       |                                              |
              [Single Agent mode]                          [Expert Panel mode]
                       |                                              |
            LangChain ReAct executor                       LangGraph state machine
            (one LLM, all tools)                                      |
                       |                              +---------------+---------------+
                       |                              |                               |
                       |                       Technical Analyst         Fundamental Researcher
                       |                       (price + history)         (news + sentiment +
                       |                              |                   reddit + 10-K + PDF)
                       |                              +-------+   +-------+
                       |                                      |   |
                       v                                      v   v
                                                       Synthesizer (LLM)
                                                              |
                                                      Final answer + sentiment chip

Tools (zero-auth, free):
- get_stock_price       (yfinance)
- get_price_history     (yfinance, returns/vol/drawdown)
- search_market_news    (DuckDuckGo via ddgs)
- get_market_sentiment  (DuckDuckGo News + VADER)
- get_reddit_buzz       (Reddit public JSON + VADER)
- query_sec_10k         (SEC EDGAR + TF-IDF RAG)
- query_uploaded_document (TF-IDF RAG over an uploaded PDF)

LLM: Groq llama-3.3-70b-versatile (default, free tier)
     OpenAI gpt-4o-mini (optional fallback)
```

| Layer            | Choice                                      | Why                                              |
| ---------------- | ------------------------------------------- | ------------------------------------------------ |
| LLM              | **Groq `llama-3.3-70b-versatile`**          | Free tier, fast inference                        |
| Single agent     | LangChain `create_tool_calling_agent`       | Native tool-calling for ReAct                    |
| Multi-agent      | **LangGraph** (parallel branches + sync)    | Demonstrates state hand-offs between specialists |
| Quote / history  | `yfinance`                                  | Zero-auth, free                                  |
| News             | `ddgs` (DuckDuckGo)                         | Zero-auth, free                                  |
| Filings RAG      | SEC EDGAR + `scikit-learn` **TF-IDF**       | Zero model downloads; runs everywhere            |
| Sentiment        | `vaderSentiment`                            | Lightweight rule-based; no GPU needed            |
| Social           | Reddit `.json` (no auth)                    | Free, no API keys                                |
| Brief export     | `fpdf2`                                     | Pure Python, no system fonts                     |
| UI               | Streamlit + `StreamlitCallbackHandler`      | Live "thought process" panel                     |
| Tests            | `pytest`                                    | 20 tests, incl. live LLM + network               |

---

## Four Production-Grade Extensions

### 1. Multi-Agent "Expert Panel" (`panel.py`)

A LangGraph state machine fans out to two specialists in parallel and a
Synthesizer merges their notes:

- **Technical Analyst** - tools: `get_stock_price`, `get_price_history`.
  Returns price, return %, annualized vol, max drawdown, 52-week levels.
- **Fundamental Researcher** - tools: `search_market_news`,
  `get_market_sentiment`, `get_reddit_buzz`, `query_sec_10k`,
  `query_uploaded_document`. Returns the *why* behind the price.
- **Sentiment scorer** - lightweight VADER pass over the fundamental brief,
  surfaced as a Positive/Neutral/Negative chip in the UI.
- **Synthesizer** - the Lead Editor LLM combines both briefs + sentiment
  into one structured answer.

Toggle "Expert Panel" in the sidebar to use it.

### 2. RAG on 10-K Filings & Uploaded PDFs (`rag.py`, `tools_extra.py`)

- **`query_sec_10k(ticker, question)`** - looks up the company's most recent
  **10-K from SEC EDGAR** (free public API), strips HTML, chunks the text,
  builds an in-memory **TF-IDF index**, and returns the top excerpts most
  relevant to the question. Cached per ticker so repeated queries are
  instant.
- **`query_uploaded_document(question)`** - drop any **PDF** into the
  Streamlit sidebar and the agent can RAG over it (e.g. an annual report,
  earnings transcript, or research note).

Why TF-IDF instead of dense embeddings? Zero downloads, zero GPU, zero API
cost. More than enough recall inside a single 10-K-sized document. Easy to
swap for `fastembed` / `sentence-transformers` later if you want.

### 3. Sentiment Layer (`tools_extra.py`)

- **`get_market_sentiment(topic)`** - pulls 10 recent DuckDuckGo News
  headlines for the topic, runs **VADER** sentiment on each, and returns
  an aggregate label + per-headline breakdown.
- **`get_reddit_buzz(ticker)`** - hits Reddit's public JSON endpoints for
  r/wallstreetbets, r/stocks, r/investing (no auth), and scores every post
  with VADER.
- The **Expert Panel** also surfaces an aggregate sentiment chip on top of
  every answer.

Caveat: VADER is a general-purpose lexicon; it sometimes mis-scores
domain-specific terms (e.g. it reads "earnings crushed expectations" as
negative because of the word "crushed"). Cheapest option that works
out-of-the-box; for production, swap in a finance-tuned model like
**FinBERT** by replacing `score_sentiment` in `tools_extra.py`.

### 4. Executive Brief Export (`report.py`)

After every answer the UI shows a **"Executive Brief Export"** panel with
two download buttons:

- **Markdown** - readable in any editor, ideal for follow-up edits.
- **PDF** (via `fpdf2`) - printable A4 brief with sections for the
  question, sentiment chip, final answer, both specialist briefs, the
  full reasoning trail, and an auto-extracted source-URL list.
  Prose sections are converted from **Markdown to HTML** (`markdown` library
  + `FPDF.write_html`) so headings, bold, lists, links, tables, and fenced
  code blocks render as real typography instead of raw `##` / `**` text.
  Unicode punctuation that Helvetica cannot draw (em dash `—`, smart quotes,
  etc.) is normalized to ASCII before rendering so PDF export does not fail.

The PDF includes a research disclaimer in the footer.

---

## Quick Start

### 1. Install dependencies

```bash
git clone https://github.com/Mizore66/market-sentinel.git
cd market-sentinel

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Add a free Groq API key

```bash
cp .env.example .env       # macOS / Linux
copy .env.example .env     # Windows
```

Get a free key at [console.groq.com/keys](https://console.groq.com/keys) and
paste it into `.env` as `GROQ_API_KEY=...`.

### 3. Run the chat app

```bash
streamlit run app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`).

Try these:

- *"What is NVDA trading at and what's the recent buzz on Reddit?"*
  -> Expert Panel: TA quotes price, FR pulls Reddit + news.
- *"What are Apple's main supply-chain risks?"*
  -> `query_sec_10k` retrieves excerpts from AAPL's latest 10-K.
- Upload an annual-report PDF in the sidebar, then ask
  *"Summarize the risk factors in the document I uploaded."*

### 4. (Optional) CLI entry points

```bash
python agent.py "Should I buy NVDA right now?"
python panel.py "Quick read on AAPL: technicals, news, and 10-K risks."
```

---

## Project Layout

```text
market-sentinel/
├── agent.py              # Single ReAct agent (all tools)
├── panel.py              # LangGraph expert panel (multi-agent)
├── tools.py              # get_stock_price, search_market_news
├── tools_extra.py        # SEC 10-K RAG, sentiment, Reddit, history, uploaded-doc RAG
├── rag.py                # PDF/HTML extraction, chunking, TF-IDF index
├── report.py             # Markdown + PDF executive-brief export
├── app.py                # Streamlit chat UI
├── requirements.txt
├── .env.example
├── pytest.ini
├── tests/
│   ├── conftest.py       # Loads .env so live-LLM markers see GROQ_API_KEY
│   ├── test_agent.py     # Tool-only + live-LLM tests for the single agent
│   └── test_extensions.py# RAG, sentiment, panel, brief export
└── README.md
```

---

## Testing

```bash
# All tests, online + offline:
pytest tests/

# Skip anything that needs a network:
pytest tests/ -m "not network"

# Skip live LLM (set GROQ_API_KEY=) for fast offline runs:
GROQ_API_KEY= pytest tests/
```

Coverage:

1. **Tool routing** - price questions hit `get_stock_price`, not search.
2. **Hallucination check** - bogus tickers produce a clear "no data" disclaimer.
3. **Sentiment** - VADER labels for positive / negative / neutral text and
   empty-input safety.
4. **RAG** - TF-IDF retrieves the right chunk; HTML stripping; PDF parsing
   round-trip via the uploaded-doc store.
5. **SEC EDGAR** (network) - real fetch + RAG over Apple's latest 10-K.
6. **Multi-agent panel** (network + live LLM) - end-to-end run that produces
   a final answer plus both specialist briefs and a sentiment chip.
7. **Executive brief** - Markdown contains every section + sources;
   PDF starts with `%PDF` magic bytes and is non-trivial in size.

---

## Switching to OpenAI (paid) instead of Groq (free)

In `.env`:

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
```

Plus:

```bash
pip install langchain-openai
```

---

## License

MIT
