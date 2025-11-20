"""
ChromaDB vector store implementation.

Provides persistent vector storage with HNSW indexing using ChromaDB.
"""

import uuid
from typing import List, Dict, Any, Optional
from pathlib import Path
import chromadb
from chromadb.config import Settings

from .base_store import BaseVectorStore, SearchResult
from src.ingestion.base_loader import Document
from src.ingestion.text_chunker import Chunk
from src.core.config import Config
from src.core.errors import IndexingError, RetrievalError
from src.core.logger import setup_logger


logger = setup_logger(__name__)


class ChromaVectorStore(BaseVectorStore):
    """ChromaDB implementation of vector store."""

    def __init__(self, config: Config):
        """
        Initialize ChromaDB store.

        Args:
            config: Configuration object
        """
        collection_name = config.collection_name
        super().__init__(collection_name)

        self.persist_directory = config.vector_store_path
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        try:
            # Initialize persistent client
            self.client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=Settings(anonymized_telemetry=False, allow_reset=True),
            )

            # Get or create collection
            # Note: We don't pass an embedding function here because we handle
            # embedding generation explicitly in the pipeline.
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={
                    "hnsw:space": config.get("vector_store.hnsw.space", "cosine"),
                    "hnsw:construction_ef": config.get(
                        "vector_store.hnsw.construction_ef", 100
                    ),
                    "hnsw:search_ef": config.get("vector_store.hnsw.search_ef", 10),
                },
            )

            logger.info(
                f"ChromaDB initialized at {self.persist_directory} (collection={collection_name})"
            )

        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise IndexingError(
                f"ChromaDB initialization failed: {e}",
                details={"path": str(self.persist_directory)},
            )

    def add_documents(
        self, chunks: List[Chunk], embeddings: List[List[float]]
    ) -> List[str]:
        """
        Add documents to ChromaDB.

        Args:
            chunks: List of text chunks
            embeddings: Corresponding embeddings

        Returns:
            List of generated document IDs
        """
        if not chunks or not embeddings:
            return []

        if len(chunks) != len(embeddings):
            raise IndexingError(
                "Mismatch between chunks and embeddings count",
                details={"chunks": len(chunks), "embeddings": len(embeddings)},
            )

        try:
            # Prepare data for ChromaDB
            ids = [str(uuid.uuid4()) for _ in chunks]
            documents = [chunk.text for chunk in chunks]
            metadatas = [chunk.metadata for chunk in chunks]

            # ChromaDB expects metadata values to be str, int, float, or bool
            # We need to ensure complex objects are converted or removed
            sanitized_metadatas = [self._sanitize_metadata(m) for m in metadatas]

            # Add to collection
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=sanitized_metadatas,
            )

            logger.info(f"Added {len(ids)} documents to ChromaDB")
            return ids

        except Exception as e:
            logger.error(f"Failed to add documents to ChromaDB: {e}")
            raise IndexingError(f"Indexing failed: {e}", details={"count": len(chunks)})

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
        try:
            # Perform search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=filter_metadata,
                include=["documents", "metadatas", "distances"],
            )

            search_results = []

            # Parse results (ChromaDB returns list of lists)
            if results["ids"] and results["ids"][0]:
                num_results = len(results["ids"][0])

                for i in range(num_results):
                    doc_id = results["ids"][0][i]
                    text = results["documents"][0][i]
                    metadata = results["metadatas"][0][i]
                    distance = results["distances"][0][i]

                    # Convert distance to similarity score (assuming cosine distance)
                    # Cosine distance is in [0, 2], where 0 is identical
                    # We want a score in [0, 1] where 1 is identical
                    # For cosine distance: score = 1 - distance (approx)
                    # Note: ChromaDB returns distance, not similarity
                    score = 1.0 - distance

                    search_results.append(
                        SearchResult(
                            document=Document(text=text, metadata=metadata),
                            score=score,
                            id=doc_id,
                        )
                    )

            logger.debug(f"Found {len(search_results)} results in ChromaDB")
            return search_results

        except Exception as e:
            logger.error(f"ChromaDB search failed: {e}")
            raise RetrievalError(f"Vector search failed: {e}", details={"top_k": top_k})

    def delete(self, document_ids: List[str]):
        """Delete documents by ID."""
        try:
            self.collection.delete(ids=document_ids)
            logger.info(f"Deleted {len(document_ids)} documents from ChromaDB")
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            raise IndexingError(f"Deletion failed: {e}")

    def count(self) -> int:
        """Get total number of documents."""
        return self.collection.count()

    def reset(self):
        """Delete all documents."""
        try:
            # ChromaDB doesn't have a direct 'clear' for collection, so we delete all
            # Or we could delete and recreate the collection
            # Here we'll use the client's reset if allowed, or delete all items

            # Note: client.reset() is destructive for ALL collections
            # So we prefer deleting items in this collection

            count = self.count()
            if count > 0:
                # Get all IDs (limit might be needed for huge collections)
                result = self.collection.get()
                if result["ids"]:
                    self.collection.delete(ids=result["ids"])

            logger.info(f"Reset collection {self.collection_name}")

        except Exception as e:
            logger.error(f"Failed to reset collection: {e}")
            raise IndexingError(f"Reset failed: {e}")

    def _sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure metadata values are compatible with ChromaDB."""
        sanitized = {}
        for k, v in metadata.items():
            if isinstance(v, (str, int, float, bool)):
                sanitized[k] = v
            elif v is None:
                continue  # Skip None values
            else:
                # Convert complex types to string
                sanitized[k] = str(v)
        return sanitized
