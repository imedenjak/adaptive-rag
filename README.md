# RAG API 🤖

A Retrieval-Augmented Generation (RAG) API built with LangChain, PostgreSQL + pgvector, and FastAPI, containerized with Docker Compose.

## Overview

This project loads content from a web page, splits it into chunks, stores embeddings in **PostgreSQL with pgvector**, and exposes a FastAPI endpoint to answer questions based on the retrieved context using OpenAI's GPT-3.5-turbo.

Indexing and serving are **separated** — run ingestion once locally to build the vector store, then Docker Compose spins up both the database and API together.

## Tech Stack

- **LangChain** — RAG pipeline orchestration
- **PostgreSQL + pgvector** — Persistent vector store
- **OpenAI** — Embeddings + LLM (GPT-3.5-turbo)
- **FastAPI** — REST API
- **Docker Compose** — Multi-container orchestration

## Project Structure

```
my_rag/
├── app/
│   ├── main.py          # FastAPI app
│   ├── rag.py           # RAG chain logic
│   └── ingest.py        # Run once to build pgvector index
├── .env                 # API keys (never commit)
├── .env.example         # Template for API keys
├── requirements.txt     # Python dependencies
├── Dockerfile           # Docker build instructions
├── docker-compose.yml   # Runs API + PostgreSQL together
└── README.md
```

## Prerequisites

- Python 3.12
- Docker Desktop
- OpenAI API key

## Getting Started

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd my_rag
```

### 2. Set up environment variables

```bash
# Copy the example env file
cp .env.example .env

# Edit .env and add your OpenAI API key
OPENAI_API_KEY=sk-xxx
```

### 3. Start PostgreSQL locally for ingestion

```bash
docker compose up db -d
```

### 4. Install dependencies locally

```bash
pip install -r requirements.txt
```

### 5. Run ingestion — build pgvector index (once only)

```bash
cd app
python ingest.py
```

Expected output:
```
Loading documents...
Loaded 1 documents
Splitting documents...
Created 66 chunks
Embedding and saving to pgvector...
Saved 66 vectors to PostgreSQL collection 'rag_docs'
Ingestion complete!
```

> This step fetches the web page, splits it into chunks, generates embeddings, and saves them to PostgreSQL. Only needs to be run once — or when you want to re-index new content.

### 6. Start everything with Docker Compose

```bash
docker compose up --build -d
```

This starts:
- `db` — PostgreSQL with pgvector on port 5432
- `api` — FastAPI app on port 8000

### 7. Check logs

```bash
# All services
docker compose logs -f

# API only
docker compose logs -f api

# DB only
docker compose logs -f db
```

Wait until you see:
```
RAG chain ready!
```

## API Usage

### Swagger UI (Recommended)

```
http://localhost:8000/docs
```

### Health Check

```bash
# Windows
curl.exe http://localhost:8000/health

# Linux / Mac
curl http://localhost:8000/health
```

Response:
```json
{"status": "ok", "chain_ready": true}
```

### Ask a Question

**Windows PowerShell:**
```bash
curl.exe -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d "{\"question\": \"What is Task Decomposition?\"}"
```

**Linux / Mac:**
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Task Decomposition?"}'
```

Response:
```json
{
  "question": "What is Task Decomposition?",
  "answer": "Task decomposition is the process of breaking down a complex task into smaller, more manageable steps..."
}
```

### Python Client

```python
import requests

response = requests.post(
    "http://localhost:8000/ask",
    json={"question": "What is Task Decomposition?"}
)
print(response.json()["answer"])
```

## Docker Compose Commands

```bash
# Start all services
docker compose up -d

# Start and rebuild images
docker compose up --build -d

# Stop all services
docker compose down

# Stop and delete all data (including postgres volume)
docker compose down -v

# View logs
docker compose logs -f

# Restart API only (after code change)
docker compose restart api
```

## When to Re-run ingest.py

```bash
python ingest.py
```

Re-run ingestion when you want to:
- Index a different or updated web page
- Change chunk size or overlap settings
- Rebuild the vector store from scratch

> `pre_delete_collection=True` in ingest.py clears old vectors before re-indexing — no duplicates.

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | ✅ Yes | Your OpenAI API key |
| `POSTGRES_CONNECTION_STRING` | ❌ No | Set automatically by docker-compose.yml |
| `LANGSMITH_TRACING` | ❌ No | Enable LangSmith tracing (default: false) |
| `LANGSMITH_API_KEY` | ❌ No | LangSmith API key (only if tracing enabled) |
| `LANGSMITH_ENDPOINT` | ❌ No | LangSmith endpoint URL |
| `LANGSMITH_PROJECT` | ❌ No | LangSmith project name |

## Architecture

```
┌─────────────────────────────────────────┐
│  ingest.py (run once locally)           │
│                                         │
│  Web Page → Split → Embed → pgvector    │
└─────────────────┬───────────────────────┘
                  │ PostgreSQL volume
┌─────────────────▼───────────────────────┐
│  docker-compose                         │
│                                         │
│  ┌─────────────┐    ┌────────────────┐  │
│  │  PostgreSQL │◄───│  FastAPI (api) │  │
│  │  + pgvector │    │                │  │
│  └─────────────┘    └────────────────┘  │
│  port 5432          port 8000           │
└─────────────────────────────────────────┘
```

## Why pgvector over ChromaDB

| | ChromaDB | PostgreSQL + pgvector |
|--|---------|----------------------|
| **Data persistence** | Requires volume mount | ✅ Native |
| **Survives restarts** | ⚠️ Volume needed | ✅ Always |
| **Inspect data** | Custom tooling | ✅ Standard SQL |
| **Scales to production** | Limited | ✅ Yes |
| **One less service** | Separate DB file | ✅ Standard Postgres |
| **Docker native** | ⚠️ Volume complexity | ✅ Official image |

## Notes

- The API container **waits for PostgreSQL** to be healthy before starting — no race conditions.
- PostgreSQL data is stored in a **named Docker volume** (`pgdata`) — survives container restarts and rebuilds.
- To fully reset all data: `docker compose down -v` then re-run `ingest.py`.
- Question must be between 3 and 500 characters.
