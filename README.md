# Market Sentinel

Autonomous **Financial Intelligence Agent** built on a **ReAct** (Reason + Act) loop.
It answers user questions by deciding, on the fly, whether to fetch real-time
market data (via `yfinance`) or scrape the latest news (via DuckDuckGo), then
synthesizing a concise, source-grounded answer.

The frontend is a Streamlit chat that visually exposes the agent's intermediate
tool calls so you can watch it think.

> Built from `plans/new_plan.md`. Defaults to the **cheapest** LLM option:
> Groq's free tier (`llama-3.3-70b-versatile`) instead of the paid OpenAI path.

---

## Architecture

```text
+------------+     +-----------------+     +--------------------+
|  User UI   | --> | LangChain Agent | --> | Tools              |
| Streamlit  |     | (ReAct / tool-  |     |  - yfinance        |
|            | <-- |  calling)       | <-- |  - DuckDuckGo News |
+------------+     +-----------------+     +--------------------+
                          |
                          v
                  Groq LLM (default)
                  llama-3.3-70b-versatile
```

| Layer        | Choice                                  | Why                            |
| ------------ | --------------------------------------- | ------------------------------ |
| LLM          | **Groq `llama-3.3-70b-versatile`**      | Free tier, fast inference      |
| Framework    | LangChain `create_tool_calling_agent`   | Native tool-calling for ReAct  |
| Tools        | `yfinance`, `duckduckgo-search`         | Zero-auth, free data sources   |
| UI           | Streamlit + `StreamlitCallbackHandler`  | Live "thought process" visible |
| Tests        | `pytest`                                | Tool routing + safety checks   |

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

### 4. (Optional) Run the agent from the CLI

```bash
python agent.py "What is the current price of NVDA?"
```

---

## Project Layout

```text
market-sentinel/
├── agent.py              # LangChain agent executor (Groq by default)
├── app.py                # Streamlit chat UI
├── tools.py              # get_stock_price, search_market_news
├── requirements.txt
├── .env.example
├── pytest.ini
├── tests/
│   └── test_agent.py     # Tool-only + live-LLM tests
└── README.md
```

---

## Testing

```bash
# Tool-level safety tests (no LLM key needed):
pytest tests/

# Full end-to-end including tool routing & hallucination check:
# (requires GROQ_API_KEY in .env)
pytest tests/ -v
```

What the tests cover (per the plan's QA spec):

1. **Tool routing**: a price question triggers `get_stock_price`, not the
   web search tool.
2. **Hallucination check**: a fake ticker like `ZZZZZZZZZZNOPE` produces a
   clear "no data" disclaimer instead of an invented price.
3. **Tool safety**: `get_stock_price` rejects empty/invalid inputs and
   returns a structured `error` field for unknown tickers.

---

## Switching to OpenAI (paid) instead of Groq (free)

In `.env`, set:

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
```

You'll also need to install the optional package:

```bash
pip install langchain-openai
```

---

## License

MIT
