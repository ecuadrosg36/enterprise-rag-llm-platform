"""
Embedding cache for performance optimization.

Caches embeddings to avoid recomputing for repeated texts.
"""

from typing import List, Optional, Dict
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import diskcache

from src.core.logger import setup_logger


logger = setup_logger(__name__)


class EmbeddingCache:
    """
    Two-tier caching system for embeddings.

    - Memory cache (LRU) for fast access
    - Disk cache for persistence across sessions
    """

    def __init__(
        self,
        enable_disk_cache: bool = True,
        disk_cache_dir: Optional[Path] = None,
        max_memory_size: int = 10000,
    ):
        """
        Initialize embedding cache.

        Args:
            enable_disk_cache: Enable persistent disk cache
            disk_cache_dir: Directory for disk cache
            max_memory_size: Maximum entries in memory cache
        """
        self.enable_disk_cache = enable_disk_cache
        self.max_memory_size = max_memory_size

        # Memory cache stats
        self.hits = 0
        self.misses = 0

        # Memory cache (simple dict, use LRU wrapper in get/set)
        self._memory_cache: Dict[str, List[float]] = {}

        # Disk cache
        self.disk_cache = None
        if enable_disk_cache:
            if disk_cache_dir is None:
                disk_cache_dir = Path("./data/cache/embeddings")

            disk_cache_dir.mkdir(parents=True, exist_ok=True)

            try:
                self.disk_cache = diskcache.Cache(str(disk_cache_dir))
                logger.info(f"Disk cache enabled at: {disk_cache_dir}")
            except Exception as e:
                logger.warning(f"Failed to initialize disk cache: {e}")
                self.enable_disk_cache = False

    def _get_key(self, text: str, model_name: str) -> str:
        """Generate cache key from text and model."""
        # Use hash of text + model for compact keys
        content = f"{model_name}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()

    def get(self, text: str, model_name: str) -> Optional[List[float]]:
        """
        Get embedding from cache.

        Args:
            text: Text that was embedded
            model_name: Model used for embedding

        Returns:
            Cached embedding or None if not found
        """
        key = self._get_key(text, model_name)

        # Try memory cache first
        if key in self._memory_cache:
            self.hits += 1
            logger.debug(f"Memory cache hit (hit_rate={self.hit_rate:.2%})")
            return self._memory_cache[key]

        # Try disk cache
        if self.enable_disk_cache and self.disk_cache:
            try:
                embedding = self.disk_cache.get(key)
                if embedding is not None:
                    # Promote to memory cache
                    self._set_memory(key, embedding)
                    self.hits += 1
                    logger.debug(f"Disk cache hit (hit_rate={self.hit_rate:.2%})")
                    return embedding
            except Exception as e:
                logger.warning(f"Disk cache read error: {e}")

        self.misses += 1
        return None

    def set(self, text: str, model_name: str, embedding: List[float]):
        """
        Store embedding in cache.

        Args:
            text: Text that was embedded
            model_name: Model used
            embedding: Embedding vector
        """
        key = self._get_key(text, model_name)

        # Store in memory cache
        self._set_memory(key, embedding)

        # Store in disk cache
        if self.enable_disk_cache and self.disk_cache:
            try:
                self.disk_cache.set(key, embedding)
            except Exception as e:
                logger.warning(f"Disk cache write error: {e}")

    def _set_memory(self, key: str, embedding: List[float]):
        """Set in memory cache with LRU eviction."""
        # Simple LRU: remove oldest if at capacity
        if len(self._memory_cache) >= self.max_memory_size:
            # Remove first key (oldest in dict order)
            oldest_key = next(iter(self._memory_cache))
            del self._memory_cache[oldest_key]

        self._memory_cache[key] = embedding

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def clear(self):
        """Clear all caches."""
        self._memory_cache.clear()

        if self.disk_cache:
            self.disk_cache.clear()

        self.hits = 0
        self.misses = 0

        logger.info("Cache cleared")

    def stats(self) -> Dict[str, any]:
        """Get cache statistics."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hit_rate,
            "memory_size": len(self._memory_cache),
            "disk_cache_enabled": self.enable_disk_cache,
        }
