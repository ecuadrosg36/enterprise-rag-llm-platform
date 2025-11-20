"""
Unit tests for retrieval layer.
"""

import pytest
from unittest.mock import Mock, MagicMock
from pathlib import Path

from src.retrieval import (
    BM25Retriever,
    HybridRetriever,
    ContextAssembler,
    BaseRetriever,
)
from src.vector_store import SearchResult, BaseVectorStore
from src.ingestion.base_loader import Document
from src.ingestion.text_chunker import Chunk


@pytest.fixture
def sample_chunks():
    """Sample chunks for indexing."""
    return [
        Chunk(text="The quick brown fox jumps over the lazy dog", metadata={"id": 1}),
        Chunk(text="The lazy dog sleeps all day", metadata={"id": 2}),
        Chunk(text="Python is a programming language", metadata={"id": 3}),
    ]


class TestBM25Retriever:
    """Tests for BM25 retriever."""

    def test_indexing_and_retrieval(self, sample_chunks, temp_dir):
        """Test indexing and basic retrieval."""
        retriever = BM25Retriever(persist_dir=temp_dir)
        retriever.index_documents(sample_chunks)

        # Exact keyword match
        results = retriever.retrieve("fox", top_k=1)
        assert len(results) == 1
        assert "fox" in results[0].document.text

        # Another match
        results = retriever.retrieve("python", top_k=1)
        assert len(results) == 1
        assert "Python" in results[0].document.text

    def test_persistence(self, sample_chunks, temp_dir):
        """Test saving and loading index."""
        # Create and save
        retriever1 = BM25Retriever(persist_dir=temp_dir)
        retriever1.index_documents(sample_chunks)

        # Load in new instance
        retriever2 = BM25Retriever(persist_dir=temp_dir)

        assert len(retriever2.documents) == 3
        assert retriever2.bm25 is not None

        # Verify retrieval works
        results = retriever2.retrieve("fox")
        assert len(results) > 0


class TestHybridRetriever:
    """Tests for Hybrid retriever."""

    def test_hybrid_fusion(self):
        """Test RRF fusion logic."""
        # Mock vector store
        vector_store = Mock(spec=BaseVectorStore)
        vector_doc = Document(text="Vector Result", metadata={})
        vector_store.search.return_value = [
            SearchResult(document=vector_doc, score=0.9, id="v1")
        ]

        # Mock embedder
        embedder = Mock()
        embedder.embed_text.return_value = [0.1]

        # Mock BM25
        bm25 = Mock(spec=BM25Retriever)
        bm25_doc = Document(text="BM25 Result", metadata={})
        bm25.retrieve.return_value = [
            SearchResult(document=bm25_doc, score=0.5, id="b1")
        ]

        hybrid = HybridRetriever(vector_store, embedder, bm25)

        results = hybrid.retrieve("query", top_k=2)

        assert len(results) == 2
        # RRF logic:
        # Vector doc rank 0 -> score 1/61
        # BM25 doc rank 0 -> score 1/61
        # Both should be present
        texts = [r.document.text for r in results]
        assert "Vector Result" in texts
        assert "BM25 Result" in texts


class TestContextAssembler:
    """Tests for context assembler."""

    def test_assemble(self):
        """Test context formatting."""
        assembler = ContextAssembler()

        results = [
            SearchResult(
                document=Document(
                    text="Content 1", metadata={"filename": "doc1.pdf", "page": 1}
                ),
                score=0.9,
                id="1",
            ),
            SearchResult(
                document=Document(text="Content 2", metadata={"filename": "doc2.txt"}),
                score=0.8,
                id="2",
            ),
        ]

        context = assembler.assemble(results)

        assert "[Document 1]" in context
        assert "doc1.pdf" in context
        assert "Content 1" in context
        assert "[Document 2]" in context
        assert "Content 2" in context
