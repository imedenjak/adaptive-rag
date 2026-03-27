from typing import List, TypedDict
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from .config import OPENAI_CHAT_MODEL
from .rag import build_retrieval_chain


# ------------------------- State ---------------------------------------------------------------------------
class AgentState(TypedDict):
    question: str
    rewritten_question: str
    documents: List[Document]
    answer: str
    is_grounded: bool
    retry_count: int
    max_retries: int


# ------------------------- LLM ---------------------------------------------------------------------------
llm = ChatOpenAI(model=OPENAI_CHAT_MODEL, temperature=0)

# Build once
retrieval_chain = build_retrieval_chain()


# ------------------------- Nodes ---------------------------------------------------------------------------
def retriever_node(state: AgentState) -> AgentState:
    """Retrieve relevant documents using your existing RAG + rank fusion"""
    print("--- RETRIEVING ---")
    question = state.get("rewritten_question") or state["question"]
    # reciprocal_rank_fusion returns (doc, score) tuples — extract docs only
    results = retrieval_chain.invoke({"question": question})
    documents = [doc for doc, score in results[:5]]
    return {"documents": documents}


def generate_node(state: AgentState) -> AgentState:
    """Generate answer from retrieved documents"""
    print("--- GENERATING ---")
    question = state["question"]
    documents = state["documents"]

    context = "\n\n".join(doc.page_content for doc in documents)

    prompt = ChatPromptTemplate.from_template("""
Answer the following question based on this context.
If the context is insufficient, say so clearly instead of making up facts.

{context}

Question: {question}
""")

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    return {"answer": answer}


def rewrite_question_node(state: AgentState) -> AgentState:
    """Rewrite the question to improve retrieval on retry."""
    print("--- REWRITING QUESTION ---")

    question = state["question"]
    answer = state["answer"]
    documents = state["documents"]
    retry_count = state.get("retry_count", 0) + 1
    context = "\n\n".join(doc.page_content for doc in documents)

    prompt = ChatPromptTemplate.from_template("""
Rewrite the user's question for a vector retriever.
Keep the meaning the same, but make it more specific and easier to match in source documents.
Return only the rewritten question.

Original question: {question}
Previous answer: {answer}
Retrieved context: {context}
""")

    chain = prompt | llm | StrOutputParser()
    rewritten_question = chain.invoke(
        {"question": question, "answer": answer, "context": context}
    ).strip()
    return {"rewritten_question": rewritten_question, "retry_count": retry_count}


def grade_answer_node(state: AgentState) -> AgentState:
    """Check if answer is grounded in documents — no hallucination"""
    print("--- GRADING ANSWER ---")

    documents = state["documents"]
    answer = state["answer"]

    context = "\n\n".join(doc.page_content for doc in documents)

    prompt = ChatPromptTemplate.from_template("""
You are a grader checking if an answer is grounded in the provided context.
Answer ONLY 'yes' or 'no'.

Context: {context}
Answer: {answer}

Is the answer grounded in the context? (yes/no):
""")

    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"context": context, "answer": answer})
    is_grounded = result.strip().lower() == "yes"
    return {"is_grounded": is_grounded}


def fallback_node(state: AgentState) -> AgentState:
    """Return a safe answer when grounded retrieval fails repeatedly."""
    print("--- FALLBACK ANSWER ---")
    return {
        "answer": (
            "I couldn't produce a grounded answer from the retrieved context. "
            "Please try rephrasing the question or adding more source material."
        )
    }


# ------------------------- Conditional Edge ------------------------------------------------------------
def should_retry(state: AgentState) -> str:
    """Decide whether to retry generation or finish"""
    if state["is_grounded"]:
        print("--- ANSWER GROUNDED, FINISHING ---")
        return "end"

    if state.get("retry_count", 0) < state.get("max_retries", 1):
        print("--- ANSWER NOT GROUNDED, REWRITING AND RETRIEVING AGAIN ---")
        return "rewrite_question"

    print("--- ANSWER NOT GROUNDED, RETURNING FALLBACK ---")
    return "fallback"


# ------------------------- BUild Graph ------------------------------------------------------------
def build_graph():
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("retrieve", retriever_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("grade_answer", grade_answer_node)
    workflow.add_node("rewrite_question", rewrite_question_node)
    workflow.add_node("fallback", fallback_node)

    # Add edges
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "grade_answer")
    workflow.add_edge("rewrite_question", "retrieve")
    workflow.add_edge("fallback", END)

    # Conditional edge - retry if answer not grounded
    workflow.add_conditional_edges(
        "grade_answer",
        should_retry,
        {
            "end": END,
            "rewrite_question": "rewrite_question",
            "fallback": "fallback",
        },
    )

    return workflow.compile()
