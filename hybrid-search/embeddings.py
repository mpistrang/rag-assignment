"""
Custom embeddings wrapper for nomic-embed-text with proper prefixes.
"""

from langchain_ollama import OllamaEmbeddings
from langchain_core.embeddings import Embeddings


class NomicEmbeddings(Embeddings):
    """Wrapper for OllamaEmbeddings that adds nomic-embed-text prefixes.

    nomic-embed-text is a dual-encoder model that requires:
    - 'search_document:' prefix for documents being indexed
    - 'search_query:' prefix for queries during retrieval
    """

    def __init__(self, model: str, base_url: str):
        self._embeddings = OllamaEmbeddings(model=model, base_url=base_url)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed documents with 'search_document:' prefix."""
        # prefixed = [f"search_document: {text}" for text in texts]
        prefixed = [text for text in texts]
        return self._embeddings.embed_documents(prefixed)

    def embed_query(self, text: str) -> list[float]:
        """Embed query with 'search_query:' prefix."""
        #return self._embeddings.embed_query(f"search_query: {text}")
        return self._embeddings.embed_query(text)
