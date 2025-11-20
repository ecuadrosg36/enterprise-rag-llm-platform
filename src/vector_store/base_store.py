"""
Base vector store interface.

Defines the contract for vector database implementations.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from src.ingestion.base_loader import Document
from src.ingestion.text_chunker import Chunk
from src.core.logger import setup_logger


logger = setup_logger(__name__)


@dataclass
class SearchResult:
    """Container for search results."""

    document: Document
    score: float
    id: str


class BaseVectorStore(ABC):
    """Abstract base class for vector stores."""

    def __init__(self, collection_name: str):
        """
        Initialize vector store.

        Args:
            collection_name: Name of the collection/index
        """
        self.collection_name = collection_name
        logger.info(
            f"Initialized {self.__class__.__name__} for collection: {collection_name}"
        )

    @abstractmethod
    def add_documents(
        self, chunks: List[Chunk], embeddings: List[List[float]]
    ) -> List[str]:
        """
        Add documents to vector store.

        Args:
            chunks: List of text chunks
            embeddings: Corresponding embeddings

        Returns:
            List of generated document IDs
        """
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Search for similar documents.

        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            filter_metadata: Optional metadata filters

        Returns:
            List of SearchResult objects
        """
        pass

    @abstractmethod
    def delete(self, document_ids: List[str]):
        """
        Delete documents by ID.

        Args:
            document_ids: List of IDs to delete
        """
        pass

    @abstractmethod
    def count(self) -> int:
        """Get total number of documents in store."""
        pass

    @abstractmethod
    def reset(self):
        """Delete all documents in collection."""
        pass
