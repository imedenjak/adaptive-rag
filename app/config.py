import os

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
# "dev" → colored human-readable console output (local development)
# "json" → JSON lines to stdout (production / log aggregators)
LOG_FORMAT = os.getenv("LOG_FORMAT", "dev")

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "rag_docs")

OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
OPENAI_QUERY_MODEL = os.getenv("OPENAI_QUERY_MODEL", OPENAI_CHAT_MODEL)
OPENAI_EMBEDDING_MODEL = os.getenv(
    "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
)

_KNOWN_EMBEDDING_DIMENSIONS = {
    "text-embedding-ada-002": 1536,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
}


def get_embedding_dimensions() -> int:
    override = os.getenv("OPENAI_EMBEDDING_DIMENSIONS")
    if override:
        return int(override)

    return _KNOWN_EMBEDDING_DIMENSIONS.get(OPENAI_EMBEDDING_MODEL, 1536)
