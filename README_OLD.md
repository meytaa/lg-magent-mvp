# LG Magent MVP — Medical PDF Audit (LangGraph)

Purpose: audit medical PDFs and produce a structured JSON report with an executive narrative, grounded in precise citations (pages, tables, figures) and qualified with severity and confidence.

## Quick Start

1) Create `.env` from `.env.example` and fill keys:
- `OPENAI_API_KEY`
- `LANGSMITH_API_KEY`, optional tracing

2) Install
```
pip install -e .
```

3) Run
```
python -m lg_magent_mvp.app
# or CLI
lg-audit run --doc "data/MC15 Deines Chiropractic.pdf" --out report.json
```

Optional: enable approvals and persistence
```
echo "APPROVALS=pause-before-finalize" >> .env
echo "USE_MEMORY=true" >> .env

lg-audit run --doc data/MC15\ Deines\ Chiropractic.pdf --thread demo
# It pauses awaiting approval; then resume:
lg-audit approve --thread demo
```

## Features

- Preflight summary: pages, sections, table/figure overview (via PyMuPDF)
- Adaptive planning: plan considers question and summary
- Keyword + semantic search (FAISS cache in `.cache/faiss`) using text from PyMuPDF
- Structured tables (via PyMuPDF `find_tables`) and figure summaries (IDs `T{page}-{n}`, `F{page}-{n}`)
- Executive narrative (4–6 sentences + prioritized actions)
- Routing modes: rule, llm, hybrid; loop guard via `MAX_HOPS`
- Persistence: SQLite checkpointer and thread IDs
- Approvals: pause-before-finalize option
- Tracing: LangSmith via `.env` (optional)

## CLI

```
lg-audit run --doc PATH --question "..." --out report.json --thread NAME [--approve]
lg-audit approve --thread NAME [--doc PATH]
```

## Config (env)

- Provider: `OPENAI_API_KEY` (required)
- Tracing: `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT` (optional)
- Router: `ROUTER_MODE=rule|llm|hybrid`, `MAX_HOPS=12`
- Persistence: `USE_MEMORY=true`, `CHECKPOINT_DB=.cache/graph_state.sqlite`
- FAISS: `FAISS_DIR=.cache/faiss`, `FAISS_CHUNK_SIZE=800`, `FAISS_CHUNK_OVERLAP=150`
- Vision: `VISION_USE_IMAGES=false`

## Tests
```
pytest -q
```


## Develop Note

- Summarize should change to some general node, the specific functionality should be added.
- Have two level of data context one for planning and revieweing and one for interpretation.
    - Currently, the state is being changed to have context and trace. 
- Check if it is possible to do have a chat in the orchestrator to have the data history.
- Make chunking more robust

- Pass all the pages (if not plain texts) to LLM
    - What we want is to have a structured pdf, so for each page we know what are the text, what are the images, and what are the tables
    - For images and tabled, they should be extracted and passed to the LLM as well.
    - We should have a catalog for each table and figure (an explanation that can to be indexed)
    - So the orchestrator can have a good overview of all files.
