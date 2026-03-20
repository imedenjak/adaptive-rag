import bs4
import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import PGVector
from langchain_openai import OpenAIEmbeddings

load_dotenv()

CONNECTION_STRING = os.getenv(
    "POSTGRES_CONNECTION_STRING",
    "postgresql+psycopg://rag_user:rag_password@localhost:5433/rag_db",  # ← localhost
)
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
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    print(f"Created {len(splits)} chunks")

    print("Embedding and saving to pgvector...")
    vectorstore = PGVector.from_documents(
        documents=splits,
        embedding=OpenAIEmbeddings(),
        collection_name=COLLECTION_NAME,
        connection_string=CONNECTION_STRING,
        pre_delete_collection=True,  # clears old data before re-indexing
    )
    print(f"Saved {len(splits)} vectors to PostgreSQL collection '{COLLECTION_NAME}'")
    print("Ingestion complete!")


if __name__ == "__main__":
    ingest()
