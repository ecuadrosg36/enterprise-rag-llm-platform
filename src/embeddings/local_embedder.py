"""
Local embeddings using sentence-transformers.

Runs models locally without API calls (free but requires compute).
"""

from typing import List, Optional
import torch
from sentence_transformers import SentenceTransformer

from .base_embedder import BaseEmbedder
from src.core.errors import EmbeddingError
from src.core.logger import setup_logger


logger = setup_logger(__name__)


class LocalEmbedder(BaseEmbedder):
    """Local embedding provider using sentence-transformers."""

    # Popular models and their dimensions
    MODEL_DIMENSIONS = {
        "all-MiniLM-L6-v2": 384,  # Fast, small
        "all-mpnet-base-v2": 768,  # Good quality
        "all-MiniLM-L12-v2": 384,  # Balanced
        "paraphrase-multilingual-MiniLM-L12-v2": 384,  # Multilingual
    }

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        batch_size: int = 32,
    ):
        """
        Initialize local embedder.

        Args:
            model_name: Sentence-transformers model name
            device: Device to run on ('cpu', 'cuda', or None for auto)
            batch_size: Batch size for encoding
        """
        super().__init__(model_name)

        # Determine device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.batch_size = batch_size

        try:
            logger.info(f"Loading model '{model_name}' on device '{device}'...")
            self.model = SentenceTransformer(model_name, device=device)
            logger.info(f"✓ Model loaded successfully (dimension={self.dimension})")

        except Exception as e:
            logger.error(f"Failed to load sentence-transformers model: {e}")
            raise EmbeddingError(
                f"Failed to load model '{model_name}': {e}",
                details={"model": model_name, "device": device, "error": str(e)},
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
            # Encode returns numpy array, convert to list
            embedding = self.model.encode(
                text.strip(), convert_to_tensor=False, show_progress_bar=False
            )

            return embedding.tolist()

        except Exception as e:
            logger.error(f"Local embedding failed: {e}")
            raise EmbeddingError(
                f"Local embedding generation failed: {e}",
                details={"model": self.model_name, "error": str(e)},
            )

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts (batched for efficiency).

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

        try:
            # Encode batch (more efficient than individual encodes)
            embeddings = self.model.encode(
                valid_texts,
                batch_size=self.batch_size,
                convert_to_tensor=False,
                show_progress_bar=len(valid_texts)
                > 100,  # Show progress for large batches
            )

            logger.info(f"Generated {len(embeddings)} embeddings locally")

            # Convert numpy arrays to lists
            return [emb.tolist() for emb in embeddings]

        except Exception as e:
            logger.error(f"Batch embedding failed: {e}")
            raise EmbeddingError(
                f"Batch embedding failed: {e}",
                details={"batch_size": len(valid_texts), "error": str(e)},
            )

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        # Try to get from model directly
        if hasattr(self.model, "get_sentence_embedding_dimension"):
            return self.model.get_sentence_embedding_dimension()

        # Fallback to known dimensions
        return self.MODEL_DIMENSIONS.get(self.model_name, 384)
