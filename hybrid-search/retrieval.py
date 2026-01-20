"""
Hybrid Search Retrieval - BM25 (keyword) + Vector (semantic) + RRF fusion.

Why hybrid? Product docs need both:
- BM25 excels at exact matches like "GET /developer/events"
- Vector excels at semantic queries like "how do I configure webhooks?"
"""

import os
from collections import defaultdict
from dotenv import load_dotenv

from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from pymongo import MongoClient

from embeddings import NomicEmbeddings
from questions import TEST_QUERIES

load_dotenv()

# Configuration (must match ingestion.py)
MONGO_DB_URL = os.getenv("MONGO_DB_URL", "mongodb://localhost:27017")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "nomic-embed-text")

DB_NAME = "product_docs_rag"
COLLECTION_NAME = "hybrid_search"
INDEX_NAME = "vector_index"

# RRF constant k=60 is standard (from original RRF paper)
RRF_K = 60


def get_mongo_client():
    return MongoClient(MONGO_DB_URL)


def get_vector_store():
    """Connect to MongoDB vector store."""
    client = get_mongo_client()
    collection = client[DB_NAME][COLLECTION_NAME]
    embeddings = NomicEmbeddings(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
    vector_store = MongoDBAtlasVectorSearch(
        collection=collection, embedding=embeddings, index_name=INDEX_NAME
    )
    return vector_store, client


def load_documents_for_bm25() -> list[Document]:
    """Load all documents from MongoDB for BM25 (requires docs in memory)."""
    client = get_mongo_client()
    collection = client[DB_NAME][COLLECTION_NAME]

    try:
        documents = []
        for doc in collection.find():
            if not doc.get("text"):
                continue
            documents.append(Document(
                page_content=doc["text"],
                metadata={
                    "source_file": doc.get("source_file", "Unknown"),
                    "title": doc.get("title", "Unknown"),
                    "module": doc.get("module"),
                    "route": doc.get("route"),
                    "linked_apis": doc.get("linked_apis", []),
                }
            ))
        return documents
    finally:
        client.close()


def reciprocal_rank_fusion(result_lists: list[list[Document]], k: int = RRF_K) -> list[Document]:
    """
    Combine ranked lists using Reciprocal Rank Fusion.
    Score = sum(1 / (k + rank)) for each list where doc appears.
    Docs in multiple lists get boosted scores.
    """
    scores = defaultdict(float)
    doc_map = {}

    for doc_list in result_lists:
        for rank, doc in enumerate(doc_list):
            doc_key = hash(doc.page_content)
            scores[doc_key] += 1.0 / (k + rank + 1)
            if doc_key not in doc_map:
                doc_map[doc_key] = doc

    sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [doc_map[key] for key in sorted_keys]


def hybrid_search(query: str, k: int = 5, documents: list[Document] = None) -> list[Document]:
    """
    Hybrid search: BM25 + Vector + RRF fusion.
    Retrieves 3x candidates from each method, then RRF picks the best k.
    """
    candidate_k = k * 3

    if documents is None:
        documents = load_documents_for_bm25()
    if not documents:
        raise ValueError("No documents found. Run ingestion.py first!")

    # BM25 search (keyword matching)
    bm25_retriever = BM25Retriever.from_documents(documents, k=candidate_k)
    bm25_results = bm25_retriever.invoke(query)

    # Vector search (semantic)
    vector_store, client = get_vector_store()
    try:
        vector_results = vector_store.as_retriever(search_kwargs={"k": candidate_k}).invoke(query)
    finally:
        client.close()

    # Combine with RRF
    combined = reciprocal_rank_fusion([bm25_results, vector_results])
    return combined[:k]


def vector_search(query: str, k: int = 5) -> list[Document]:
    """Vector-only search (for comparison in evals)."""
    vector_store, client = get_vector_store()
    try:
        return vector_store.as_retriever(search_kwargs={"k": k}).invoke(query)
    finally:
        client.close()


def bm25_search(query: str, k: int = 5, documents: list[Document] = None) -> list[Document]:
    """BM25-only search (for comparison in evals)."""
    if documents is None:
        documents = load_documents_for_bm25()
    if not documents:
        raise ValueError("No documents found. Run ingestion.py first!")

    return BM25Retriever.from_documents(documents, k=k).invoke(query)


def format_retrieved_context(documents: list[Document]) -> str:
    """Format retrieved documents into context string for LLM."""
    parts = []
    for i, doc in enumerate(documents, 1):
        title = doc.metadata.get("title", "Unknown")
        route = doc.metadata.get("route", "N/A")
        apis = ", ".join(doc.metadata.get("linked_apis", [])) or "N/A"

        parts.append(
            f"[Document {i}]\n"
            f"Title: {title}\n"
            f"Route: {route}\n"
            f"APIs: {apis}\n"
            f"Content:\n{doc.page_content}\n"
        )
    return "\n---\n".join(parts)


def main():
    """Test and compare retrieval methods."""
    print("=" * 50)
    print("RETRIEVAL TEST")
    print("=" * 50)

    # Check we have data
    client = get_mongo_client()
    doc_count = client[DB_NAME][COLLECTION_NAME].count_documents({})
    client.close()

    if doc_count == 0:
        print("No documents found! Run ingestion.py first.")
        return

    print(f"Found {doc_count} documents\n")

    documents = load_documents_for_bm25()

    for query in TEST_QUERIES:
        print(f"\nQuery: {query}")
        print("-" * 40)

        bm25_results = bm25_search(query, k=3, documents=documents)
        vector_results = vector_search(query, k=3)
        hybrid_results = hybrid_search(query, k=3, documents=documents)

        print("BM25:  ", [d.metadata.get("title", "?")[:30] for d in bm25_results])
        print("Vector:", [d.metadata.get("title", "?")[:30] for d in vector_results])
        print("Hybrid:", [d.metadata.get("title", "?")[:30] for d in hybrid_results])


if __name__ == "__main__":
    main()
