import os
from langchain_qdrant import QdrantVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from qdrant_client import QdrantClient

from langchain_core.load import dumps, loads

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "rag_docs"


def get_unique_union(documents: list[list]):
    """Unique union of retrieved docs"""
    # Flatten list of lists, and convert each Document to string
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    # Get unique documents
    unique_docs = list(set(flattened_docs))
    # Return
    return [loads(doc) for doc in unique_docs]


def build_rag_chain():
    print("Loading Qdrant vectorstore...")

    embeddings = OpenAIEmbeddings()
    # embeddings = OllamaEmbeddings(model="llama3.2")
    client = QdrantClient(url=QDRANT_URL)

    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )
    print("Qdrant loaded successfully!")

    retriever = vectorstore.as_retriever()

    # Multi Query: Different Perspectives
    template = """You are an AI language model assistant. Your task is to generate five 
                  different versions of the given user question to retrieve relevant documents from a vector 
                  database. By generating multiple perspectives on the user question, your goal is to help
                  the user overcome some of the limitations of the distance-based similarity search. 
                  Provide these alternative questions separated by newlines. Original question: {question}"""

    prompt_perspectives = ChatPromptTemplate.from_template(template)

    generate_queries = (
        prompt_perspectives
        | ChatOpenAI(temperature=0)
        | StrOutputParser()
        | (lambda x: x.split("\n"))
    )

    retrieval_chain = generate_queries | retriever.map() | get_unique_union

    # RAG
    template = template = """Answer the following question based on this context:

{context}

Question: {question}
"""
    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    # llm = ChatOllama(model="llama3.2", temperature=0)

    # def format_docs(docs):
    #     return "\n\n".join(doc.page_content for doc in docs)

    final_rag_chain = (
        {"context": retrieval_chain, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return final_rag_chain
