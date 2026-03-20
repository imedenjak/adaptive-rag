from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from rag import build_rag_chain
import uvicorn

load_dotenv()

app = FastAPI(title="RAG API")
# Build chain once at startup — not on every request

rag_chain = None


@app.on_event("startup")
async def startup_event():
    global rag_chain
    print("Loading RAG chain...")
    rag_chain = build_rag_chain()
    print("RAG chain ready!")


class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)


class AnswerResponse(BaseModel):
    question: str
    answer: str

# Routes
@app.get("/health")
def health():
    return {"status": "ok", "chain_ready": rag_chain is not None}


@app.post("/ask", response_model=AnswerResponse)
async def ask(request: QuestionRequest):
    if rag_chain is None:
        raise HTTPException(status_code=503, detail="RAG chain not ready")
    try:
        answer = await run_in_threadpool(rag_chain.invoke, request.question)
        return AnswerResponse(question=request.question, answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
