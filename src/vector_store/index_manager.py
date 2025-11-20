"""
Index manager for high-level vector store operations.

Orchestrates the indexing process: chunking -> embedding -> storage.
"""

from typing import List, Optional
from tqdm import tqdm

from src.vector_store.base_store import BaseVectorStore
from src.ingestion.text_chunker import Chunk
from src.embeddings.base_embedder import BaseEmbedder
from src.core.logger import setup_logger


logger = setup_logger(__name__)


class IndexManager:
    """
    High-level manager for indexing documents.

    Handles batch processing of embeddings and storage.
    """

    def __init__(
        self,
        vector_store: BaseVectorStore,
        embedder: BaseEmbedder,
        batch_size: int = 100,
    ):
        """
        Initialize index manager.

        Args:
            vector_store: Vector store instance
            embedder: Embedder instance
            batch_size: Batch size for processing
        """
        self.vector_store = vector_store
        self.embedder = embedder
        self.batch_size = batch_size

    def index_chunks(self, chunks: List[Chunk], show_progress: bool = True) -> int:
        """
        Index text chunks into vector store.

        Args:
            chunks: List of text chunks
            show_progress: Show progress bar

        Returns:
            Number of chunks indexed
        """
        if not chunks:
            logger.warning("No chunks to index")
            return 0

        logger.info(f"Indexing {len(chunks)} chunks...")

        total_indexed = 0

        # Process in batches
        iterator = range(0, len(chunks), self.batch_size)
        if show_progress:
            iterator = tqdm(
                iterator,
                desc="Indexing batches",
                total=(len(chunks) + self.batch_size - 1) // self.batch_size,
            )

        for i in iterator:
            batch_chunks = chunks[i : i + self.batch_size]
            batch_texts = [chunk.text for chunk in batch_chunks]

            try:
                # Generate embeddings
                embeddings = self.embedder.embed_batch(batch_texts)

                # Store in vector DB
                self.vector_store.add_documents(batch_chunks, embeddings)

                total_indexed += len(batch_chunks)

            except Exception as e:
                logger.error(f"Failed to index batch {i}: {e}")
                # We continue with next batch instead of failing completely
                continue

        logger.info(f"Indexed {total_indexed}/{len(chunks)} chunks")
        return total_indexed

    def get_stats(self) -> dict:
        """Get index statistics."""
        return {
            "total_documents": self.vector_store.count(),
            "collection_name": self.vector_store.collection_name,
            "embedding_model": self.embedder.model_name,
            "embedding_dimension": self.embedder.dimension,
        }

    def reset_index(self):
        """Clear the entire index."""
        self.vector_store.reset()
        logger.warning("Index cleared")
