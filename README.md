# Underwater Systems Design Agent

A web-based AI agent that helps design AUVs, ROVs, and underwater drones by combining:
- A **local growing knowledge corpus** (Qdrant vector DB + BM25 hybrid search)
- **Live web search** across open academic sources (OpenAlex, arXiv, Semantic Scholar, Crossref, Unpaywall) and the open web (Tavily)
- **Auto-ingestion** of every open-access paper it finds into the corpus, so it gets smarter over time
- **Gemini 2.0 Flash** for synthesis and reasoning

## Architecture

```
User question
     │
     ▼
[Agent Router] ───► Has answer in corpus? ───► Yes ──► Local RAG ─┐
     │                                                            │
     └──────────────► No ──► Web search (OpenAlex, arXiv,         │
                              Semantic Scholar, Tavily)           │
                              │                                   │
                              ▼                                   │
                       Fetch open-access PDFs (Unpaywall)         │
                              │                                   │
                              ▼                                   │
                       Parse (Docling) + Ingest into corpus       │
                              │                                   │
                              └─────────────────────────────► Synthesize
                                                                  │
                                                                  ▼
                                                          Cited answer to user
```

## Quick start

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up API keys in .env
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY (and optional TAVILY_API_KEY)

# 4. Initialize empty corpus
python -m src.ingest --init

# 5. Run the app
streamlit run src/app.py
```

Open http://localhost:8501 in your browser.

## API keys needed

- **GOOGLE_API_KEY** (required): Free from https://aistudio.google.com/apikey
- **TAVILY_API_KEY** (optional, recommended): Free tier 1000 searches/month at https://tavily.com
- **SEMANTIC_SCHOLAR_API_KEY** (optional): Free, increases rate limits at https://www.semanticscholar.org/product/api

OpenAlex, arXiv, Crossref, and Unpaywall do not require keys.

## File structure

```
underwater-agent/
├── .env                       # your API keys (gitignored)
├── .env.example
├── requirements.txt
├── README.md
├── corpus/                    # downloaded PDFs (auto-managed, gitignored)
├── storage/                   # Qdrant vector DB (auto-managed, gitignored)
├── logs/                      # ingestion + query logs
├── data/                      # for manually adding PDFs (optional)
└── src/
    ├── __init__.py
    ├── config.py              # central config
    ├── app.py                 # Streamlit UI
    ├── agent.py               # main agent orchestrator
    ├── llm.py                 # Gemini wrapper
    ├── corpus.py              # Qdrant + BM25 hybrid retrieval
    ├── ingest.py              # PDF parsing + indexing pipeline
    ├── search/
    │   ├── __init__.py
    │   ├── openalex.py        # OpenAlex API
    │   ├── arxiv_search.py    # arXiv API
    │   ├── semantic_scholar.py
    │   ├── crossref.py
    │   ├── unpaywall.py       # open-access PDF resolver
    │   └── tavily_search.py   # general web search
    └── utils/
        ├── __init__.py
        └── pdf_fetch.py       # download + parse PDFs
```

## How it grows

Every time the agent does a web search and finds an open-access PDF:
1. Downloads the PDF to `corpus/`
2. Parses it with Docling (handles equations, tables, multi-column)
3. Chunks and embeds into Qdrant
4. Logs the source in `logs/ingestion.log`

After ~50 queries on related topics, the local corpus has hundreds of papers and most queries answer from local data — fast, free, and offline-capable.

## Domain focus

The agent is prompted for underwater systems design: AUVs, ROVs, gliders, underwater drones. It refuses off-topic questions and refuses to answer when retrieval confidence is low (rather than hallucinating).

## License

MIT. Personal/research use. Respect the licenses of papers ingested.
