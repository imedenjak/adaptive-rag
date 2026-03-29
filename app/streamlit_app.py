import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import structlog
from dotenv import load_dotenv

from app.logging_config import configure_logging
from app.agent import build_graph

load_dotenv()
configure_logging()

logger = structlog.get_logger(__name__)

st.title("RAG Agent 🤖")


# Build agent once — cached so it doesn't rebuild on every interaction
@st.cache_resource
def get_agent():
    logger.info("agent.build")
    return build_graph()

agent = get_agent()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Input
if question := st.chat_input("Ask a question..."):
    query_id = str(uuid.uuid4())[:8]

    # Show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    # Run agent
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            structlog.contextvars.bind_contextvars(query_id=query_id)
            logger.info("query.received", question=question)
            t0 = time.perf_counter()

            result = agent.invoke(
                {"question": question, "rewritten_question": "", "retry_count": 0, "max_retries": 1}
            )

            elapsed_ms = round((time.perf_counter() - t0) * 1000)
            answer = result["answer"]
            documents = result.get("documents", [])

            logger.info(
                "query.complete",
                elapsed_ms=elapsed_ms,
                is_grounded=result.get("is_grounded"),
                retry_count=result.get("retry_count", 0),
                doc_count=len(documents),
                answer_chars=len(answer),
            )
            structlog.contextvars.clear_contextvars()

            st.write(answer)

            if documents:
                with st.expander("Sources"):
                    for index, doc in enumerate(documents, start=1):
                        source = doc.metadata.get("source", "Unknown source")
                        chunk = doc.page_content.strip().replace("\n", " ")
                        st.markdown(f"**{index}. {source}**")
                        st.write(chunk[:400] + ("..." if len(chunk) > 400 else ""))

    st.session_state.messages.append({"role": "assistant", "content": answer})
