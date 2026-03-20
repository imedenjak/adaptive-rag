import os
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate

CHROMA_PATH = "./chroma_db"


def build_rag_chain():
    # Verify ChromaDB exists
    if not os.path.exists(CHROMA_PATH):
        raise RuntimeError("ChromaDB not found! Run ingest.py first:\npython ingest.py")

    print("Loading existing ChromaDB...")
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=OpenAIEmbeddings(),
    )
    print(f"Loaded {vectorstore._collection.count()} vectors from ChromaDB")

    retriever = vectorstore.as_retriever()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "human",
                """You are an assistant for question-answering tasks. 
                     Use the following pieces of retrieved context to answer the question. 
                     If you don't know the answer, just say that you don't know. 
                     Use three sentences maximum and keep the answer concise.

          Question: {question} 
          Context: {context} 
          Answer:""",
            )
        ]
    )

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain
