import structlog
from langchain_qdrant import QdrantVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from qdrant_client import QdrantClient

from langchain_core.load import dumps, loads
from .config import (
    COLLECTION_NAME,
    OPENAI_EMBEDDING_MODEL,
    OPENAI_QUERY_MODEL,
    QDRANT_URL,
)

logger = structlog.get_logger(__name__)


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


def build_retrieval_chain():
    """Returns just the retrieval chain for agent use"""
    logger.info("retrieval_chain.build", qdrant_url=QDRANT_URL, collection=COLLECTION_NAME)
    client = QdrantClient(url=QDRANT_URL)
    embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)

    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )
    retriever = vectorstore.as_retriever()

    # Define prompt_perspectives here
    template = """You are an AI language model assistant. Your task is to generate five 
different versions of the given user question to retrieve relevant documents from a vector 
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search. 
Provide these alternative questions separated by newlines. Original question: {question}"""

    prompt_perspective = ChatPromptTemplate.from_template(template)

    generate_queries = (
        prompt_perspective
        | ChatOpenAI(model=OPENAI_QUERY_MODEL, temperature=0)
        | StrOutputParser()
        | (lambda x: x.split("\n"))
    )

    retrieval_chain_rag_fusion = (
        generate_queries | retriever.map() | reciprocal_rank_fusion
    )

    return retrieval_chain_rag_fusion
