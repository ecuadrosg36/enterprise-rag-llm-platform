"""
Base embedder interface.

All embedding providers implement this interface for consistent usage.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from src.core.logger import setup_logger


logger = setup_logger(__name__)


class BaseEmbedder(ABC):
    """Abstract base class for embedding providers."""
    
    def __init__(self, model_name: str):
        """
        Initialize embedder.
        
        Args:
            model_name: Name of the embedding model
        """
        self.model_name = model_name
        logger.info(f"Initialized {self.__class__.__name__} with model: {model_name}")
    
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        pass
    
    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts (batch processing).
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Get embedding dimension."""
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name}, dimension={self.dimension})"
