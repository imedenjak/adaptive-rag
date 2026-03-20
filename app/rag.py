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


def reciprocal_rank_fusion(results: list[list], k=60):
    """Reciprocal_rank_fusion that takes multiple lists of ranked documents
    and an optional parameter k used in the RRF formula"""

    # Initialize a dictionary to hold fused scores for each unique document
    fused_scores = {}

    # Iterate through each list of ranked documents
    for docs in results:
        # Iterate through each document in the list, with its rank (position in the list)
        for rank, doc in enumerate(docs):
            # Convert the document to a string format to use as a key (assumes documents can be serialized to JSON)
            doc_str = dumps(doc)
            # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            # Retrieve the current score of the document, if any
            fused_scores[doc_str]
            # Update the score of the document using the RRF formula: 1 / (rank + k)
            fused_scores[doc_str] += 1 / (rank + k)

    # Sort the documents based on their fused scores in descending order to get the final reranked results
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    # Return the reranked results as a list of tuples, each containing the document and its fused score
    return reranked_results


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

    # RAG-Fusion: Related
    template = """You are a helpful assistant that generates multiple search queries based on a single input query. \n
Generate multiple search queries related to: {question} \n
Output (4 queries):"""

    prompt_rag_fusion = ChatPromptTemplate.from_template(template)

    generate_queries = (
        prompt_rag_fusion
        | ChatOpenAI(temperature=0)
        | StrOutputParser()
        | (lambda x: x.split("\n"))
    )

    retrieval_chain_rag_fusion = generate_queries | retriever.map() | reciprocal_rank_fusion

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
        {"context": retrieval_chain_rag_fusion, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return final_rag_chain
