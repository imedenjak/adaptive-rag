import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from dotenv import load_dotenv
from app.agent import build_graph

load_dotenv()

st.title("RAG Agent 🤖")


# Build agent once — cached so it doesn't rebuild on every interaction
@st.cache_resource
def get_agent():
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
    # Show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    # Run agent
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = agent.invoke(
                {"question": question, "rewritten_question": "", "retry_count": 0, "max_retries": 1}
            )
            answer = result["answer"]
            st.write(answer)

            documents = result.get("documents", [])
            if documents:
                with st.expander("Sources"):
                    for index, doc in enumerate(documents, start=1):
                        source = doc.metadata.get("source", "Unknown source")
                        chunk = doc.page_content.strip().replace("\n", " ")
                        st.markdown(f"**{index}. {source}**")
                        st.write(chunk[:400] + ("..." if len(chunk) > 400 else ""))

    st.session_state.messages.append({"role": "assistant", "content": answer})
