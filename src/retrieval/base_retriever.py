"""
Base retriever interface.

Defines the contract for document retrieval strategies.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from src.vector_store.base_store import SearchResult
from src.core.logger import setup_logger


logger = setup_logger(__name__)


class BaseRetriever(ABC):
    """Abstract base class for retrievers."""
    
    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query string
            top_k: Number of documents to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of SearchResult objects
        """
        pass
