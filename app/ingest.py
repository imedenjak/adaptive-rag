import bs4
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_qdrant import QdrantVectorStore

from langchain_openai import OpenAIEmbeddings

# from langchain_ollama import OllamaEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

load_dotenv()

COLLECTION_NAME = "rag_docs"


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
    embeddings = OpenAIEmbeddings()
    # embeddings = OllamaEmbeddings(model="llama3.2")

    # Create local persistent client
    client = QdrantClient(url="http://localhost:6333")

    # Check if collection already exists
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    # Create collection
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=1536,  # OpenAI text-embedding-ada-002 dimension
            # size=3072,  # llama3.2 dimension
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
    ingest()
