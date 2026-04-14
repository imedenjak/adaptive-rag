# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`adaptive-rag` is a stateful RAG (Retrieval-Augmented Generation) agent built with LangGraph. It combines hybrid search (dense + sparse) with an agentic grading-and-retry loop to produce grounded answers from source documents.

## Commands

### Setup

```bash
uv sync                          # Install dependencies
cp .env.example .env             # Configure API keys
docker compose up qdrant -d      # Start Qdrant vector DB
python -m app.ingest <URL> ...   # Ingest documents (one-time)
```

### Run

```bash
streamlit run app/streamlit_app.py   # Local dev (http://localhost:8501)
docker compose up --build -d         # Full containerized run
```

### Evaluate

```bash
docker compose exec app python -m eval.evaluate   # RAGAS quality metrics → eval/results.json
```

### Debug

```bash
python app/check.py                                                        # Verify Qdrant collection config
docker compose exec postgres psql -U rag -d chat -c "SELECT * FROM messages;"
docker compose logs -f app
langgraph dev                                                               # LangGraph Studio (requires uv add langgraph-cli)
```

## Architecture

### Agent Workflow (LangGraph)

The graph in [app/agent.py](app/agent.py) defines five nodes with conditional routing:

1. **retrieve** — runs multi-query expansion (5 variants via LLM) then hybrid search + RRF, returns top 5 docs
2. **generate** — calls the LLM with retrieved docs + chat history to produce an answer
3. **grade_answer** — a grader LLM checks whether the answer is grounded in the retrieved docs
4. **rewrite_question** — rewrites the query if grading fails, feeding back into `retrieve`
5. **fallback** — returns a safe response after `max_retries` exhausted

Flow: `retrieve → generate → grade_answer → (grounded? end : retry_count < max? rewrite → retrieve : fallback)`

### Retrieval Pipeline ([app/rag.py](app/rag.py))

For each query variant:
- **Dense search**: OpenAI embeddings (`text-embedding-3-small`, 1536d) via Qdrant
- **Sparse search**: BM25 keyword matching via FastEmbed (`Qdrant/bm25`)
- **Fusion**: Reciprocal Rank Fusion merges ranked lists — `score = Σ 1/(rank + k)`

Multi-query generation is done by the LLM rewriting the user's question into 5 variations to improve recall.

### Key Files

| File | Purpose |
|------|---------|
| [app/agent.py](app/agent.py) | LangGraph graph definition — nodes, edges, conditional routing |
| [app/rag.py](app/rag.py) | Retrieval: multi-query generation, hybrid search, RRF fusion |
| [app/config.py](app/config.py) | All environment variable loading (single source of truth) |
| [app/ingest.py](app/ingest.py) | Web document loading, chunking, and indexing into Qdrant |
| [app/streamlit_app.py](app/streamlit_app.py) | Chat UI with session state, source display, PostgreSQL history |
| [app/chat_history.py](app/chat_history.py) | PostgreSQL-backed persistent chat history |
| [eval/evaluate.py](eval/evaluate.py) | RAGAS evaluation over [eval/testset.json](eval/testset.json) |

### Configuration

All runtime config comes from environment variables (`.env`). Key variables:

| Variable | Default | Purpose |
|----------|---------|---------|
| `OPENAI_API_KEY` | required | LLM + embeddings |
| `OPENAI_CHAT_MODEL` | `gpt-4o-mini` | Generation, grading, rewriting |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Dense embeddings |
| `FAST_EMBED_SPARSE` | `Qdrant/bm25` | Sparse keyword embeddings |
| `QDRANT_URL` | `http://localhost:6333` | Vector DB |
| `QDRANT_COLLECTION_NAME` | `rag_docs` | Collection |
| `DATABASE_URL` | `postgresql://rag:rag@localhost:5432/chat` | PostgreSQL connection string |
| `LOG_FORMAT` | `dev` | `dev` (colored) or `json` (production) |

### Infrastructure

- **Qdrant**: vector DB for hybrid search — runs via Docker Compose
- **PostgreSQL**: chat history — runs via Docker Compose, persisted in `postgres_data` named volume
- **Docker Compose**: defines `qdrant`, `postgres`, and `app` services
- **uv**: package manager — use `uv sync` and `uv add` instead of pip

## Development Notes

- **Keep `README.md` in sync**: update `README.md` whenever you make changes that affect setup, configuration, architecture, or usage — this includes new env variables, changed commands, new features, or modified workflows.

