import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import structlog
from dotenv import load_dotenv

from app.chat_history import load_history, save_message
from app.logging_config import configure_logging
from app.agent import build_graph
from app.config import APP_PASSWORD

load_dotenv()
configure_logging()

logger = structlog.get_logger(__name__)

st.title("RAG Agent 🤖")

# --- Authentication ---
if APP_PASSWORD:
    if not st.session_state.get("authenticated"):
        with st.form("login_form"):
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")
        if submitted:
            if password == APP_PASSWORD:
                st.session_state.authenticated = True
                logger.info("auth.success")
                st.rerun()
            else:
                logger.warning("auth.failure")
                st.error("Incorrect password.")
        st.stop()


# Build agent once — cached so it doesn't rebuild on every interaction
@st.cache_resource
def get_agent():
    logger.info("agent.build")
    return build_graph()


agent = get_agent()

# Chat history

if "messages" not in st.session_state:
    st.session_state.messages = load_history()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Input
if question := st.chat_input("Ask a question..."):
    query_id = str(uuid.uuid4())[:8]

    # Build history before appending current question to avoid duplication
    chat_history = [
        f"{'Human' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in st.session_state.messages
    ]

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
                {
                    "question": question,
                    "rewritten_question": "",
                    "retry_count": 0,
                    "max_retries": 1,
                    "chat_history": chat_history,
                },
                config={"configurable": {"thread_id": st.session_state.thread_id}},
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
    save_message("user", question)
    save_message("assistant", answer)
