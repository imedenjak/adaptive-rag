"""
RAGAS evaluation script.

Usage (inside the app container):
    python -m eval.evaluate
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, faithfulness, context_precision, context_recall
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.agent import build_graph

TESTSET_PATH = Path(__file__).parent / "testset.json"


def run_pipeline(agent, question: str) -> dict:
    """Run a single question through the agent and return answer + contexts."""
    result = agent.invoke(
        {
            "question": question,
            "rewritten_question": "",
            "retry_count": 0,
            "max_retries": 1,
        },
        config={"configurable": {"thread_id": question[:40]}},
    )

    answer = result["answer"]
    # Extract plain text from retrieved Document objects
    contexts = [doc.page_content for doc in result.get("documents", [])]
    return {"answer": answer, "contexts": contexts}


def main():
    print("Building agent...")
    agent = build_graph()

    testset = json.loads(TESTSET_PATH.read_text())

    questions, answers, contexts, ground_truths = [], [], [], []

    for i, sample in enumerate(testset, 1):
        question = sample["question"]
        print(f"[{i}/{len(testset)}] Running: {question}")
        output = run_pipeline(agent, question)

        questions.append(question)
        answers.append(output["answer"])
        contexts.append(output["contexts"])
        ground_truths.append(sample.get("ground_truth", ""))

    # RAGAS expects a HuggingFace Dataset with these exact column names
    dataset = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )

    print("\nRunning RAGAS evaluation...")
    result = evaluate(
        dataset=dataset,
        metrics=[answer_relevancy, faithfulness, context_precision, context_recall],
        llm=ChatOpenAI(model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")),
        embeddings=OpenAIEmbeddings(
            model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        ),
    )

    print("\n=== RAGAS Results ===")
    print(result)

    # Also save to a file for later inspection
    output_path = Path(__file__).parent / "results.json"
    result.to_pandas().to_json(output_path, orient="records", indent=2)
    print(f"\nDetailed results saved to {output_path}")


if __name__ == "__main__":
    main()
