# RAG Agent

A Retrieval-Augmented Generation (RAG) agent built with LangChain, LangGraph, Qdrant, and Streamlit.

The agent generates multiple query variants, retrieves documents using hybrid search (dense + sparse) with reciprocal rank fusion, generates an answer, grades it for hallucinations, and retries with a rewritten question if the answer is not grounded.

## Tech Stack

- **LangGraph** — stateful agent with grading and retry logic
- **LangChain** — RAG pipeline orchestration
- **Qdrant** — persistent vector store with hybrid search (dense + sparse)
- **FastEmbed** — local sparse (BM25) embeddings for keyword matching
- **OpenAI** — configurable embeddings and chat models
- **Streamlit** — chat UI
- **LangSmith** — tracing and observability
- **Docker** — containerization

## Project Structure

```
adaptive_rag/
├── app/
│   ├── __init__.py         # makes app a Python package
│   ├── agent.py            # LangGraph agent — retrieve, generate, grade, retry
│   ├── chat_history.py     # PostgreSQL-backed persistent chat history
│   ├── config.py           # env-based configuration
│   ├── check.py            # verify Qdrant collection vector config
│   ├── ingest.py           # build Qdrant index (run once)
│   ├── logging_config.py   # structured logging (dev/json)
│   ├── rag.py              # multi-query + hybrid search + reciprocal rank fusion
│   └── streamlit_app.py    # Streamlit chat UI
├── eval/
│   ├── testset.json        # small evaluation dataset
│   └── evaluate.py         # RAGAS evaluation script
├── .env                    # API keys (never commit)
├── .env.example            # template
├── pyproject.toml          # dependencies and package metadata
├── langgraph.json          # LangGraph Studio config
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INGESTION (run once)                        │
│                                                                     │
│  Source URLs                                                        │
│      │                                                              │
│      ▼                                                              │
│  WebBaseLoader + BeautifulSoup  (fetch & parse HTML)                │
│      │                                                              │
│      ▼                                                              │
│  RecursiveCharacterTextSplitter (400 tokens, 50 overlap)            │
│      │                                                              │
│      ├──► OpenAI text-embedding-3-small  → dense vectors (1536d)    │
│      └──► FastEmbed BM25                 → sparse vectors           │
│                          │                                          │
│                          ▼                                          │
│                   Qdrant Collection                                 │
│              (hybrid index: dense + sparse)                         │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                      RETRIEVAL  (per query)                         │
│                                                                     │
│  User Question                                                      │
│      │                                                              │
│      ▼                                                              │
│  Multi-Query Generation  (LLM generates 5 query variants)           │
│      │                                                              │
│      ▼  (for each variant)                                          │
│  Hybrid Search in Qdrant  (dense cosine + BM25 sparse)              │
│      │                                                              │
│      ▼                                                              │
│  Reciprocal Rank Fusion   (merge ranked lists → top 5 docs)         │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    AGENT GRAPH  (LangGraph)                         │
│                                                                     │
│                       ┌──────────┐                                  │
│                       │ retrieve │◄─────────────────┐               │
│                       └────┬─────┘                  │               │
│                            │                        │               │
│                            ▼                        │               │
│                       ┌──────────┐           ┌──────┴───────┐       │
│                       │ generate │           │   rewrite_   │       │
│                       └────┬─────┘           │   question   │       │
│                            │                 └──────────────┘       │
│                            ▼                        ▲               │
│                     ┌────────────┐                  │               │
│                     │   grade_   │  not grounded,   │               │
│                     │   answer   │  retries left ───┘               │
│                     └─────┬──────┘                                  │
│                           │                                         │
│              ┌────────────┼─────────────┐                           │
│              │            │             │                           │
│           grounded   not grounded    max retries                    │
│              │        retries left    exceeded                      │
│              ▼                           ▼                          │
│             END                      ┌──────────┐                   │
│                                      │ fallback │──► END            │
│                                      └──────────┘                   │
└─────────────────────────────────────────────────────────────────────┘
```

**Why each component exists:**

| Component | Why |
|-----------|-----|
| Multi-query generation | A single query may miss relevant docs due to vocabulary mismatch — 5 variants improve recall |
| Hybrid search (dense + sparse) | Dense embeddings excel at semantic similarity; BM25 catches exact keywords, names, acronyms |
| Reciprocal Rank Fusion | Merges ranked results from multiple query variants without requiring score normalization |
| LangGraph agent | Enables stateful retry loop — grade → rewrite → retrieve again if answer is not grounded |
| Answer grading | Guards against hallucinations before returning a response to the user |

## Prerequisites

- Python 3.12
- Docker Desktop
- OpenAI API key
- LangSmith API key (optional, for tracing)

## Getting Started

### 1. Clone and configure

```bash
git clone <your-repo-url>
cd adaptive_rag
cp .env.example .env
# Edit .env and fill in your API keys
```

### 2. Install dependencies

```bash
uv sync
```

### 3. Start Qdrant and ingest documents

```bash
# Start Qdrant
docker compose up qdrant -d

# Build the vector index (run once, or after changing sources/models)
python -m app.ingest
```

### 4. Run locally

```bash
streamlit run app/streamlit_app.py
```

Open `http://localhost:8501`.

## Run with Docker Compose

```bash
# First run — build and start
docker compose up --build -d

# Ingest documents into the running container
docker compose exec app python -m app.ingest

# View logs
docker compose logs -f app
```

Open `http://localhost:8501`.

## LangGraph Studio (browser)

```bash
uv add langgraph-cli
langgraph dev
```

Open: `https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024`

## Environment Variables

| Variable                     | Required | Default                  | Description                                          |
|------------------------------|----------|--------------------------|------------------------------------------------------|
| `OPENAI_API_KEY`             | Yes      | —                        | OpenAI API key                                       |
| `OPENAI_CHAT_MODEL`          | No       | `gpt-4o-mini`            | Model for generation, grading, and rewriting         |
| `OPENAI_QUERY_MODEL`         | No       | same as chat model       | Model for generating query variants                  |
| `OPENAI_EMBEDDING_MODEL`     | No       | `text-embedding-3-small` | Embedding model                                      |
| `OPENAI_EMBEDDING_DIMENSIONS`| No       | auto-detected            | Override vector size for custom embedding models     |
| `FAST_EMBED_SPARSE`          | No       | `Qdrant/bm25`            | FastEmbed sparse model for keyword matching          |
| `QDRANT_URL`                 | No       | `http://localhost:6333`  | Qdrant server URL                                    |
| `QDRANT_COLLECTION_NAME`     | No       | `rag_docs`               | Qdrant collection name                               |
| `LANGCHAIN_TRACING_V2`       | No       | `false`                  | Enable LangSmith tracing                             |
| `LANGCHAIN_API_KEY`          | No       | —                        | LangSmith API key                                    |
| `LANGCHAIN_PROJECT`          | No       | —                        | LangSmith project name                               |
| `DATABASE_URL`               | No       | `postgresql://rag:rag@localhost:5432/chat` | PostgreSQL connection string for chat history |
| `APP_PASSWORD`               | No       | —                        | Password gate for the Streamlit UI (unset = no auth) |

## RAGAS Evaluation

The `eval/` folder contains a small test set and an evaluation script that measures pipeline quality using [RAGAS](https://docs.ragas.io) — no human-labelled answers required.

**Metrics:**

| Metric                    | What it measures                                              |
|---------------------------|---------------------------------------------------------------|
| **Answer Relevancy**      | Is the answer about the question asked?                       |
| **Faithfulness**          | Is every claim in the answer supported by retrieved chunks?   |
| **Context Precision**     | Are the retrieved chunks relevant to the question?            |
| **Context Recall**        | Does the retrieved context cover the reference answer?        |

**Run evaluation:**

```bash
docker compose exec app python -m eval.evaluate
```

Results are printed to stdout and saved to `eval/results.json`.

**Latest results** (13-question test set, `gpt-4o-mini`, `text-embedding-3-small`):

| Metric            | Score |
|-------------------|-------|
| Answer Relevancy  | 0.92  |
| Faithfulness      | 0.95  |
| Context Precision | 0.95  |
| Context Recall    | 0.97  |

**Interpreting scores** (all metrics are 0–1):
- `≥ 0.8` — good
- `0.5–0.8` — room for improvement
- `< 0.5` — investigate retrieval or generation quality

## Authentication

The Streamlit UI supports optional password protection. Set `APP_PASSWORD` in your `.env` to enable it:

```env
APP_PASSWORD=changeme
```

When set, users must enter the password before accessing the chat interface. Leave the variable unset (or commented out) to disable the password gate entirely.

## Persistent Chat History

Chat history is stored in a PostgreSQL database running as a Docker service (`postgres`), persisted in the `postgres_data` named volume so it survives container restarts. To inspect it:

```bash
docker compose exec postgres psql -U rag -d chat -c "SELECT * FROM messages;"
```

## Notes

- Re-run `docker compose exec app python -m app.ingest <url>` after changing source URLs, chunk settings, or embedding model.
- Switching embedding models requires deleting the Qdrant collection first — vector dimensions must match.
- Run `python app/check.py` to verify the Qdrant collection has both dense and sparse vectors configured correctly.
- `@st.cache_resource` ensures the agent graph is built once per Streamlit session.
- Use `docker compose down -v` to also remove the `postgres_data` volume (deletes all chat history).
