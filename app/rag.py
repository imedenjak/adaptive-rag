import os
from langchain_postgres.vectorstores import PGVector
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate

CONNECTION_STRING = os.getenv(
    "POSTGRES_CONNECTION_STRING",
    "postgresql+psycopg://rag_user:rag_password@db:5432/rag_db",  # 'db' = docker-compose service name
)
COLLECTION_NAME = "rag_docs"


def build_rag_chain():
    print("Connecting to pgvector...")
    vectorstore = PGVector(
        embeddings=OpenAIEmbeddings(),
        collection_name=COLLECTION_NAME,
        connection=CONNECTION_STRING,
        async_mode=False,
    )
    print("Connected to pgvector successfully!")

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
