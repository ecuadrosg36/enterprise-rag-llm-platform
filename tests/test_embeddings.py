"""
Unit tests for embedding layer.
"""

import pytest
from unittest.mock import Mock, patch

from src.embeddings import (
    LocalEmbedder,
    EmbeddingFactory,
    EmbeddingCache,
)
from src.core.config import Config
from src.core.errors import EmbeddingError


class TestLocalEmbedder:
    """Tests for local sentence-transformers embedder."""
    
    def test_embed_text(self):
        """Test embedding single text."""
        embedder = LocalEmbedder(model_name="all-MiniLM-L6-v2")
        
        text = "This is a test sentence."
        embedding = embedder.embed_text(text)
        
        assert isinstance(embedding, list)
        assert len(embedding) == embedder.dimension
        assert all(isinstance(x, float) for x in embedding)
    
    def test_embed_batch(self):
        """Test batch embedding."""
        embedder = LocalEmbedder(model_name="all-MiniLM-L6-v2")
        
        texts = ["First sentence.", "Second sentence.", "Third sentence."]
        embeddings = embedder.embed_batch(texts)
        
        assert len(embeddings) == 3
        assert all(len(emb) == embedder.dimension for emb in embeddings)
    
    def test_empty_text_error(self):
        """Test error on empty text."""
        embedder = LocalEmbedder(model_name="all-MiniLM-L6-v2")
        
        with pytest.raises(EmbeddingError, match="Cannot embed empty text"):
            embedder.embed_text("")
    
    def test_dimension_property(self):
        """Test dimension property."""
        embedder = LocalEmbedder(model_name="all-MiniLM-L6-v2")
        
        assert embedder.dimension == 384


class TestEmbeddingFactory:
    """Tests for embedding factory."""
    
    def test_create_local_embedder(self, temp_dir):
        """Test creating local embedder from config."""
        # Create minimal config
        import yaml
        config_data = {
            'app': {'name': 'test'},
            'embedding': {
                'provider': 'local',
                'local': {'model': 'all-MiniLM-L6-v2'}
            },
            'vector_store': {'persist_directory': str(temp_dir)},
            'llm': {'provider': 'openai'},
            'paths': {'raw_documents': str(temp_dir)}
        }
        
        config_file = temp_dir / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        config = Config.load(env='dev', config_dir=temp_dir)
        
        embedder = EmbeddingFactory.create(config)
        
        assert isinstance(embedder, LocalEmbedder)
        assert embedder.model_name == 'all-MiniLM-L6-v2'
    
    def test_create_from_provider(self):
        """Test direct provider creation."""
        embedder = EmbeddingFactory.create_from_provider(
            provider='local',
            model_name='all-MiniLM-L6-v2'
        )
        
        assert isinstance(embedder, LocalEmbedder)


class TestEmbeddingCache:
    """Tests for embedding cache."""
    
    def test_cache_set_get(self, temp_dir):
        """Test setting and getting from cache."""
        cache = EmbeddingCache(
            enable_disk_cache=True,
            disk_cache_dir=temp_dir / "cache"
        )
        
        text = "Test text"
        model = "test-model"
        embedding = [0.1, 0.2, 0.3]
        
        # Set
        cache.set(text, model, embedding)
        
        # Get
        cached = cache.get(text, model)
        
        assert cached == embedding
    
    def test_cache_miss(self):
        """Test cache miss."""
        cache = EmbeddingCache(enable_disk_cache=False)
        
        result = cache.get("nonexistent", "model")
        
        assert result is None
        assert cache.misses == 1
    
    def test_cache_hit_rate(self):
        """Test hit rate calculation."""
        cache = EmbeddingCache(enable_disk_cache=False)
        
        # Set some entries
        cache.set("text1", "model", [0.1, 0.2])
        cache.set("text2", "model", [0.3, 0.4])
        
        # Hit
        cache.get("text1", "model")
        # Miss
        cache.get("text3", "model")
        
        # Hit rate should be 1/2 = 0.5
        assert cache.hit_rate == 0.5
    
    def test_cache_clear(self):
        """Test clearing cache."""
        cache = EmbeddingCache(enable_disk_cache=False)
        
        cache.set("text", "model", [0.1])
        cache.clear()
        
        assert len(cache._memory_cache) == 0
        assert cache.hits == 0
        assert cache.misses == 0
