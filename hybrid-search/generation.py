"""
Generation - Use retrieved context to generate answers with local Ollama LLM.
LangFuse provides observability (traces all LangChain operations).
"""

import os
from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from retrieval import hybrid_search, format_retrieved_context

load_dotenv()

# Configuration
OLLAMA_BASE_URL = os.environ["OLLAMA_BASE_URL"]
OLLAMA_LLM_MODEL = os.environ["OLLAMA_LLM_MODEL"]
LANGFUSE_HOST = os.environ["LANGFUSE_HOST"]

# LangFuse for observability (auto-traces LangChain operations)
from langfuse import Langfuse, get_client
from langfuse.langchain import CallbackHandler
Langfuse(host=LANGFUSE_HOST)

RAG_PROMPT = """You are a technical support analyst. Answer using ONLY the context below.
If the context doesn't have the answer, say so.

Context:
{context}

Question: {question}

Answer:"""


def generate_answer(question: str, k: int = 5) -> dict:
    """
    RAG pipeline: Retrieve -> Format Context -> Generate Answer.
    Returns answer + sources for transparency.
    """
    langfuse_handler = CallbackHandler()

    # 1. Retrieve
    documents = hybrid_search(question, k=k)
    if not documents:
        return {"answer": "No relevant documents found.", "sources": []}

    # 2. Format context
    context = format_retrieved_context(documents)

    # 3. Generate
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT)
    llm = ChatOllama(model=OLLAMA_LLM_MODEL, temperature=0, base_url=OLLAMA_BASE_URL)
    chain = prompt | llm | StrOutputParser()

    answer = chain.invoke(
        {"context": context, "question": question},
        config={"callbacks": [langfuse_handler]}
    )

    get_client().flush()

    return {
        "answer": answer,
        "sources": [
            {"title": doc.metadata.get("title", "Unknown"), "route": doc.metadata.get("route", "N/A")}
            for doc in documents
        ],
        "documents": documents  # For evals
    }


def interactive_mode():
    """Interactive Q&A session."""
    print("=" * 50)
    print("RAG Q&A (type 'quit' to exit)")
    print("=" * 50)

    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() in ['quit', 'exit', 'q']:
            break
        if not question:
            continue

        print("\nSearching...")
        result = generate_answer(question)

        print(f"\nAnswer: {result['answer']}")
        print(f"\nSources:")
        for s in result["sources"]:
            print(f"  - {s['title']} ({s['route']})")


def main():
    """Run interactive mode."""
    interactive_mode()


if __name__ == "__main__":
    main()
