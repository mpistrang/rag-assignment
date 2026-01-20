"""
Precision Eval - Measures retrieval quality using LLM-as-judge.
Precision = relevant_docs / retrieved_docs
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from retrieval import hybrid_search, vector_search, bm25_search
from questions import EVAL_QUESTIONS

load_dotenv()

OLLAMA_BASE_URL = os.environ["OLLAMA_BASE_URL"]
OLLAMA_LLM_MODEL = os.environ["OLLAMA_LLM_MODEL"]

RELEVANCE_PROMPT = """Does this document help answer the question?

Question: {question}

Document Title: {title}
Content (excerpt): {content}

Reply with "yes" if the document contains information that helps answer the question.
Reply with "no" if the document is unrelated or only tangentially related.

Answer (yes/no):"""


def judge_relevance(question: str, document) -> bool:
    """Use LLM to judge if a document is relevant to the question."""
    llm = ChatOllama(model=OLLAMA_LLM_MODEL, temperature=0, base_url=OLLAMA_BASE_URL)
    prompt = ChatPromptTemplate.from_template(RELEVANCE_PROMPT)

    result = (prompt | llm).invoke({
        "question": question,
        "title": document.metadata.get("title", "Unknown"),
        "content": document.page_content[:1000]
    })

    response = result.content.strip().lower()
    # Check first word to handle "yes, because..." or "no, it doesn't..."
    first_word = response.split()[0] if response else ""
    is_relevant = first_word.startswith("yes")
    #print(f"      [{document.metadata.get('title', '?')[:35]:35}] -> {response[:50]}")
    return is_relevant


def calculate_precision(question: str, documents: list) -> float:
    """Calculate precision = relevant_docs / total_docs."""
    if not documents:
        return 0.0

    relevant = sum(1 for doc in documents if judge_relevance(question, doc))
    return relevant / len(documents)


def run_evaluation(test_questions: list[str], k: int = 5):
    """Compare precision across BM25, Vector, and Hybrid search methods."""
    print("=" * 50)
    print("PRECISION EVALUATION")
    print("=" * 50)

    totals = {"hybrid": [], "vector": [], "bm25": []}

    for question in test_questions:
        print(f"\nQuery: {question}")

        hybrid_p = calculate_precision(question, hybrid_search(question, k))
        vector_p = calculate_precision(question, vector_search(question, k))
        bm25_p = calculate_precision(question, bm25_search(question, k))

        totals["hybrid"].append(hybrid_p)
        totals["vector"].append(vector_p)
        totals["bm25"].append(bm25_p)

        print(f"  Hybrid: {hybrid_p:.0%} | Vector: {vector_p:.0%} | BM25: {bm25_p:.0%}")

    # Averages
    print("\n" + "=" * 50)
    print("AVERAGE PRECISION")
    print("=" * 50)
    avgs = {}
    for method, scores in totals.items():
        avgs[method] = sum(scores) / len(scores) if scores else 0
        print(f"  {method.capitalize():8} {avgs[method]:.0%}")

    # Summary comparison
    print("\n" + "-" * 50)
    hybrid_vs_vector = avgs["hybrid"] - avgs["vector"]
    hybrid_vs_bm25 = avgs["hybrid"] - avgs["bm25"]
    print(f"Hybrid vs Vector: {hybrid_vs_vector:+.0%}")
    print(f"Hybrid vs BM25:   {hybrid_vs_bm25:+.0%}")


def main():
    run_evaluation(EVAL_QUESTIONS)


if __name__ == "__main__":
    main()
