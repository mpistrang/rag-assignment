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

    # add boolean to control the prefix for testing purposes
    ADD_PREFIX = True

    def __init__(self, model: str, base_url: str):
        self._embeddings = OllamaEmbeddings(model=model, base_url=base_url)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed documents with 'search_document:' prefix."""
        if self.ADD_PREFIX:
            prefixed = [f"search_document: {text}" for text in texts]
        else:
            prefixed = texts
        return self._embeddings.embed_documents(prefixed)

    def embed_query(self, text: str) -> list[float]:
        """Embed query with 'search_query:' prefix."""
        if self.ADD_PREFIX:
            prefixed = f"search_query: {text}"
        else:
            prefixed = text
        return self._embeddings.embed_query(prefixed)