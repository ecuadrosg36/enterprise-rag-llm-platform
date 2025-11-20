"""
OpenAI embeddings provider.

Uses OpenAI's text-embedding models via the API.
"""

import os
from typing import List, Optional
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from .base_embedder import BaseEmbedder
from src.core.errors import EmbeddingError, RateLimitError
from src.core.logger import setup_logger


logger = setup_logger(__name__)


class OpenAIEmbedder(BaseEmbedder):
    """OpenAI embedding provider."""
    
    # Model dimensions mapping
    MODEL_DIMENSIONS = {
        'text-embedding-3-small': 1536,
        'text-embedding-3-large': 3072,
        'text-embedding-ada-002': 1536,
    }
    
    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        batch_size: int = 100,
        max_retries: int = 3
    ):
        """
        Initialize OpenAI embedder.
        
        Args:
            model_name: OpenAI embedding model name
            api_key: OpenAI API key (if None, uses OPENAI_API_KEY env var)
            batch_size: Maximum texts per batch request
            max_retries: Number of retries on failure
        """
        super().__init__(model_name)
        
        # Get API key
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise EmbeddingError(
                "OpenAI API key not provided. Set OPENAI_API_KEY environment variable.",
                details={'provider': 'openai'}
            )
        
        self.client = OpenAI(api_key=self.api_key)
        self.batch_size = batch_size
        self.max_retries = max_retries
        
        # Validate model
        if model_name not in self.MODEL_DIMENSIONS:
            logger.warning(f"Unknown OpenAI model: {model_name}, dimension may be incorrect")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        if not text or not text.strip():
            raise EmbeddingError("Cannot embed empty text")
        
        try:
            response = self.client.embeddings.create(
                model=self.model_name,
                input=text.strip()
            )
            
            embedding = response.data[0].embedding
            logger.debug(f"Generated embedding (dim={len(embedding)})")
            
            return embedding
            
        except Exception as e:
            logger.error(f"OpenAI embedding failed: {e}")
            
            if "rate_limit" in str(e).lower():
                raise RateLimitError(
                    f"OpenAI rate limit exceeded: {e}",
                    details={'model': self.model_name}
                )
            
            raise EmbeddingError(
                f"OpenAI embedding generation failed: {e}",
                details={'model': self.model_name, 'error': str(e)}
            )
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Batches requests to stay within API limits.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        # Filter empty texts
        valid_texts = [t.strip() for t in texts if t and t.strip()]
        
        if not valid_texts:
            raise EmbeddingError("No valid texts to embed")
        
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(valid_texts), self.batch_size):
            batch = valid_texts[i:i + self.batch_size]
            
            try:
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=batch
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                
                logger.debug(f"Generated {len(batch_embeddings)} embeddings (batch {i // self.batch_size + 1})")
                
            except Exception as e:
                logger.error(f"Batch embedding failed: {e}")
                raise EmbeddingError(
                    f"Batch embedding failed: {e}",
                    details={'batch_size': len(batch), 'error': str(e)}
                )
        
        logger.info(f"Generated {len(all_embeddings)} embeddings total")
        return all_embeddings
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self.MODEL_DIMENSIONS.get(self.model_name, 1536)
