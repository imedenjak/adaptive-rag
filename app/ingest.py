"""
Document ingestion pipeline.

Drops and recreates the collection on every run.

Usage:
    python -m app.ingest https://example.com/page1 https://example.com/page2
"""

import argparse

import bs4
import structlog
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

try:
    from .config import (
        COLLECTION_NAME,
        OPENAI_EMBEDDING_MODEL,
        QDRANT_URL,
        get_embedding_dimensions,
    )
    from .logging_config import configure_logging
except ImportError:
    from config import (  # noqa: E402
        COLLECTION_NAME,
        OPENAI_EMBEDDING_MODEL,
        QDRANT_URL,
        get_embedding_dimensions,
    )
    from logging_config import configure_logging  # noqa: E402

load_dotenv()

logger = structlog.get_logger(__name__)

# BeautifulSoup strainer used only for known blog-style pages.
# None means load all visible text (safe default for arbitrary URLs).
_BLOG_STRAINER = bs4.SoupStrainer(class_=("post-content", "post-title", "post-header"))


def _load_url(url: str) -> list:
    """Load a single URL. Uses a blog-specific strainer for known domains, plain text otherwise."""
    is_blog = "lilianweng.github.io" in url
    loader = WebBaseLoader(
        web_paths=(url,),
        bs_kwargs=dict(parse_only=_BLOG_STRAINER) if is_blog else {},
    )
    return loader.load()


def ingest(urls: list[str]) -> None:
    client = QdrantClient(url=QDRANT_URL)

    if client.collection_exists(COLLECTION_NAME):
        logger.info("ingest.collection_drop", collection=COLLECTION_NAME)
        client.delete_collection(COLLECTION_NAME)

    dims = get_embedding_dimensions()
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=dims, distance=Distance.COSINE),
    )
    logger.info(
        "ingest.collection_created", collection=COLLECTION_NAME, dimensions=dims
    )

    embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=400,  # 400 tokens, not characters
        chunk_overlap=50,
    )

    total_chunks = 0
    for url in urls:
        logger.info("ingest.start", url=url)
        docs = _load_url(url)
        logger.info("ingest.loaded", url=url, doc_count=len(docs))

        chunks = splitter.split_documents(docs)
        logger.info("ingest.split", url=url, chunk_count=len(chunks))

        vectorstore.add_documents(chunks)
        logger.info("ingest.indexed", url=url, chunk_count=len(chunks))
        total_chunks += len(chunks)

    logger.info("ingest.done", urls_processed=len(urls), total_chunks=total_chunks)


if __name__ == "__main__":
    configure_logging()

    parser = argparse.ArgumentParser(
        description="Ingest web pages into the RAG vector store."
    )
    parser.add_argument("urls", nargs="+", help="One or more URLs to ingest")
    args = parser.parse_args()

    ingest(urls=args.urls)
