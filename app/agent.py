from typing import List, TypedDict
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import structlog

from .config import OPENAI_CHAT_MODEL
from .rag import build_retrieval_chain

logger = structlog.get_logger(__name__)


# ------------------------- State ---------------------------------------------------------------------------
class AgentState(TypedDict):
    question: str
    rewritten_question: str
    documents: List[Document]
    answer: str
    is_grounded: bool
    retry_count: int
    max_retries: int
    chat_history: List[str]


# ------------------------- LLM ---------------------------------------------------------------------------
llm = ChatOpenAI(model=OPENAI_CHAT_MODEL, temperature=0)

# Build once
retrieval_chain = build_retrieval_chain()


# ------------------------- Nodes ---------------------------------------------------------------------------
def retriever_node(state: AgentState) -> AgentState:
    """Retrieve relevant documents using your existing RAG + rank fusion"""
    question = state.get("rewritten_question") or state["question"]
    logger.info("retrieve.start", question=question)
    # reciprocal_rank_fusion returns (doc, score) tuples — extract docs only
    results = retrieval_chain.invoke({"question": question})
    documents = [doc for doc, score in results[:5]]
    logger.info("retrieve.done", doc_count=len(documents))
    return {"documents": documents}


def generate_node(state: AgentState) -> AgentState:
    """Generate answer from retrieved documents"""
    logger.info("generate.start")
    question = state["question"]
    documents = state["documents"]
    chat_history = state.get("chat_history") or []

    context = "\n\n".join(doc.page_content for doc in documents)
    history_text = "\n".join(chat_history[-6:])     # last 3 turns (6 lines)

    prompt = ChatPromptTemplate.from_template("""
Answer the following question based on this context.
If the context is insufficient, say so clearly instead of making up facts.

{history_section}Context:
{context}

Question: {question}
""")

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({
        "context": context, 
        "question": question,
        "history_section": f"Conversation so far:\n{history_text}\n\n" if history_text else "",
        })
    logger.info("generate.done", answer_chars=len(answer))
    return {"answer": answer}


def rewrite_question_node(state: AgentState) -> AgentState:
    """Rewrite the question to improve retrieval on retry."""
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
    logger.info(
        "rewrite.done",
        retry_count=retry_count,
        original_question=question,
        rewritten_question=rewritten_question,
    )
    return {"rewritten_question": rewritten_question, "retry_count": retry_count}


def grade_answer_node(state: AgentState) -> AgentState:
    """Check if answer is grounded in documents — no hallucination"""
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
    logger.info("grade.done", is_grounded=is_grounded)
    return {"is_grounded": is_grounded}


def fallback_node(state: AgentState) -> AgentState:
    """Return a safe answer when grounded retrieval fails repeatedly."""
    logger.warning("fallback.triggered", retry_count=state.get("retry_count", 0))
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
        logger.info("route.end", reason="grounded")
        return "end"

    if state.get("retry_count", 0) < state.get("max_retries", 1):
        logger.info(
            "route.rewrite",
            reason="not_grounded",
            retry_count=state.get("retry_count", 0),
        )
        return "rewrite_question"

    logger.warning(
        "route.fallback",
        reason="max_retries_reached",
        retry_count=state.get("retry_count", 0),
    )
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

    return workflow.compile(checkpointer=MemorySaver())
