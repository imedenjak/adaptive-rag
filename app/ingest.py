import bs4
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_qdrant import QdrantVectorStore

from langchain_openai import OpenAIEmbeddings

# from langchain_ollama import OllamaEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
try:
    from .config import (
        COLLECTION_NAME,
        OPENAI_EMBEDDING_MODEL,
        QDRANT_URL,
        get_embedding_dimensions,
    )
except ImportError:
    from config import (  # noqa: E402 — fallback when run as a script
        COLLECTION_NAME,
        OPENAI_EMBEDDING_MODEL,
        QDRANT_URL,
        get_embedding_dimensions,
    )

load_dotenv()


def ingest():
    print("Loading documents...")
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    docs = loader.load()
    print(f"Loaded {len(docs)} documents")

    print("Splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50)

    splits = text_splitter.split_documents(docs)
    print(f"Created {len(splits)} chunks")

    print("Embedding and saving to Qdrant...")
    embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)
    # embeddings = OllamaEmbeddings(model="llama3.2")

    # Create local persistent client
    client = QdrantClient(url=QDRANT_URL)

    # Check if collection already exists
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    # Create collection
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=get_embedding_dimensions(),
            distance=Distance.COSINE,
        ),
    )

    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )
    vectorstore.add_documents(splits)
    print(f"Saved {len(splits)} vectors to Qdrant")
    print("Ingestion complete!")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    ingest()
