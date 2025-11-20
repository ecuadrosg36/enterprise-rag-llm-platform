"""
Hybrid Retriever combining Vector Search and BM25.

Uses Reciprocal Rank Fusion (RRF) to merge results from semantic
and keyword searches for optimal retrieval performance.
"""

from typing import List, Dict, Any, Optional

from .base_retriever import BaseRetriever
from .bm25_retriever import BM25Retriever
from src.vector_store.base_store import BaseVectorStore, SearchResult
from src.embeddings.base_embedder import BaseEmbedder
from src.core.logger import setup_logger
from src.core.config import Config


logger = setup_logger(__name__)


class HybridRetriever(BaseRetriever):
    """
    Combines vector search (semantic) and BM25 (keyword).
    """

    def __init__(
        self,
        vector_store: BaseVectorStore,
        embedder: BaseEmbedder,
        bm25_retriever: Optional[BM25Retriever] = None,
        weights: tuple = (0.7, 0.3),  # (vector_weight, bm25_weight)
    ):
        """
        Initialize hybrid retriever.

        Args:
            vector_store: Vector database
            embedder: Embedding model
            bm25_retriever: Optional BM25 retriever
            weights: Weights for (vector, bm25) scores if using weighted sum
        """
        self.vector_store = vector_store
        self.embedder = embedder
        self.bm25 = bm25_retriever
        self.weights = weights

        logger.info("Initialized HybridRetriever")

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Retrieve documents using hybrid search.

        Args:
            query: Search query
            top_k: Number of final results
            filter_metadata: Metadata filters

        Returns:
            Merged list of SearchResult objects
        """
        # 1. Vector Search
        vector_results = []
        try:
            query_embedding = self.embedder.embed_text(query)
            vector_results = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=top_k * 2,  # Fetch more for re-ranking
                filter_metadata=filter_metadata,
            )
        except Exception as e:
            logger.error(f"Vector search failed in hybrid retrieval: {e}")

        # 2. BM25 Search (if available)
        bm25_results = []
        if self.bm25:
            try:
                bm25_results = self.bm25.retrieve(
                    query=query, top_k=top_k * 2, filter_metadata=filter_metadata
                )
            except Exception as e:
                logger.error(f"BM25 search failed in hybrid retrieval: {e}")

        # 3. Merge Results (Reciprocal Rank Fusion)
        if not self.bm25 or not bm25_results:
            return vector_results[:top_k]

        if not vector_results:
            return bm25_results[:top_k]

        return self._reciprocal_rank_fusion(vector_results, bm25_results, top_k)

    def _reciprocal_rank_fusion(
        self,
        vector_results: List[SearchResult],
        bm25_results: List[SearchResult],
        k: int = 60,
    ) -> List[SearchResult]:
        """
        Merge results using Reciprocal Rank Fusion (RRF).

        RRF score = 1 / (k + rank)

        Args:
            vector_results: Results from vector search
            bm25_results: Results from BM25
            k: RRF constant (usually 60)

        Returns:
            Re-ranked list of results
        """
        # Map content to results to deduplicate
        # Using text as key since IDs might differ (BM25 uses temp IDs)
        merged_scores = {}
        doc_map = {}

        # Process vector results
        for rank, result in enumerate(vector_results):
            doc_key = result.document.text
            if doc_key not in doc_map:
                doc_map[doc_key] = result

            score = 1.0 / (k + rank + 1)
            merged_scores[doc_key] = merged_scores.get(doc_key, 0) + score

        # Process BM25 results
        for rank, result in enumerate(bm25_results):
            doc_key = result.document.text
            if doc_key not in doc_map:
                doc_map[doc_key] = result

            score = 1.0 / (k + rank + 1)
            merged_scores[doc_key] = merged_scores.get(doc_key, 0) + score

        # Sort by merged score
        sorted_keys = sorted(
            merged_scores.keys(), key=lambda x: merged_scores[x], reverse=True
        )

        # Create final list
        final_results = []
        for key in sorted_keys:
            original_result = doc_map[key]
            # Update score to RRF score
            original_result.score = merged_scores[key]
            final_results.append(original_result)

        return final_results
