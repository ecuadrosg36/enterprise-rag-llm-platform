"""
Unit tests for vector store.
"""

import pytest
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock

from src.vector_store import ChromaVectorStore, IndexManager, SearchResult
from src.ingestion.text_chunker import Chunk
from src.core.config import Config
from src.core.errors import IndexingError


@pytest.fixture
def temp_chroma_dir(temp_dir):
    """Temporary directory for ChromaDB."""
    chroma_dir = temp_dir / "chromadb_test"
    yield chroma_dir
    # Cleanup
    if chroma_dir.exists():
        shutil.rmtree(chroma_dir)


@pytest.fixture
def mock_config(temp_chroma_dir):
    """Mock configuration."""
    config = Mock(spec=Config)
    config.collection_name = "test_collection"
    config.vector_store_path = temp_chroma_dir
    config.get.side_effect = lambda k, d=None: d  # Return default for config.get()
    return config


@pytest.fixture
def sample_chunks():
    """Sample chunks for testing."""
    return [
        Chunk(text="Chunk 1", metadata={"source": "doc1"}),
        Chunk(text="Chunk 2", metadata={"source": "doc2"}),
    ]


@pytest.fixture
def sample_embeddings():
    """Sample embeddings."""
    return [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]


class TestChromaVectorStore:
    """Tests for ChromaDB vector store."""
    
    def test_initialization(self, mock_config):
        """Test store initialization."""
        store = ChromaVectorStore(mock_config)
        assert store.collection_name == "test_collection"
        assert store.persist_directory == mock_config.vector_store_path
    
    def test_add_documents(self, mock_config, sample_chunks, sample_embeddings):
        """Test adding documents."""
        store = ChromaVectorStore(mock_config)
        
        ids = store.add_documents(sample_chunks, sample_embeddings)
        
        assert len(ids) == 2
        assert store.count() == 2
    
    def test_search(self, mock_config, sample_chunks, sample_embeddings):
        """Test searching documents."""
        store = ChromaVectorStore(mock_config)
        store.add_documents(sample_chunks, sample_embeddings)
        
        # Search with one of the embeddings
        results = store.search(query_embedding=[0.1, 0.2, 0.3], top_k=1)
        
        assert len(results) == 1
        assert isinstance(results[0], SearchResult)
        assert results[0].document.text == "Chunk 1"
        # Score should be close to 1.0 (identical vector)
        assert results[0].score > 0.99
    
    def test_metadata_filtering(self, mock_config, sample_chunks, sample_embeddings):
        """Test search with metadata filter."""
        store = ChromaVectorStore(mock_config)
        store.add_documents(sample_chunks, sample_embeddings)
        
        # Filter for doc2
        results = store.search(
            query_embedding=[0.1, 0.2, 0.3],
            top_k=2,
            filter_metadata={"source": "doc2"}
        )
        
        assert len(results) == 1
        assert results[0].document.metadata["source"] == "doc2"
    
    def test_delete(self, mock_config, sample_chunks, sample_embeddings):
        """Test deleting documents."""
        store = ChromaVectorStore(mock_config)
        ids = store.add_documents(sample_chunks, sample_embeddings)
        
        store.delete([ids[0]])
        
        assert store.count() == 1
    
    def test_reset(self, mock_config, sample_chunks, sample_embeddings):
        """Test resetting store."""
        store = ChromaVectorStore(mock_config)
        store.add_documents(sample_chunks, sample_embeddings)
        
        store.reset()
        
        assert store.count() == 0


class TestIndexManager:
    """Tests for IndexManager."""
    
    def test_index_chunks(self):
        """Test indexing workflow."""
        # Mock dependencies
        vector_store = Mock()
        embedder = Mock()
        embedder.embed_batch.return_value = [[0.1], [0.2]]
        
        manager = IndexManager(vector_store, embedder, batch_size=2)
        
        chunks = [
            Chunk(text="1", metadata={}),
            Chunk(text="2", metadata={})
        ]
        
        count = manager.index_chunks(chunks, show_progress=False)
        
        assert count == 2
        embedder.embed_batch.assert_called_once()
        vector_store.add_documents.assert_called_once()
    
    def test_index_empty(self):
        """Test indexing empty list."""
        manager = IndexManager(Mock(), Mock())
        count = manager.index_chunks([], show_progress=False)
        assert count == 0
