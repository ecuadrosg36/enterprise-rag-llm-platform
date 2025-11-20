"""
Embedding factory for creating embedder instances.

Uses factory pattern to instantiate the correct embedder based on config.
"""

from typing import Optional

from .base_embedder import BaseEmbedder
from .openai_embedder import OpenAIEmbedder
from .local_embedder import LocalEmbedder
from src.core.config import Config
from src.core.errors import ConfigurationError
from src.core.logger import setup_logger


logger = setup_logger(__name__)


class EmbeddingFactory:
    """Factory for creating embedding providers."""
    
    @staticmethod
    def create(config: Config) -> BaseEmbedder:
        """
        Create embedder instance based on configuration.
        
        Args:
            config: Configuration object
            
        Returns:
            Configured embedder instance
            
        Raises:
            ConfigurationError: If provider is unknown
        """
        provider = config.embedding_provider
        
        logger.info(f"Creating embedder for provider: {provider}")
        
        if provider == "openai":
            return EmbeddingFactory._create_openai(config)
        elif provider == "local":
            return EmbeddingFactory._create_local(config)
        else:
            raise ConfigurationError(
                f"Unknown embedding provider: {provider}",
                details={'provider': provider, 'valid_providers': ['openai', 'local']}
            )
    
    @staticmethod
    def _create_openai(config: Config) -> OpenAIEmbedder:
        """Create OpenAI embedder."""
        model = config.get('embedding.openai.model', 'text-embedding-3-small')
        batch_size = config.get('embedding.openai.batch_size', 100)
        max_retries = config.get('embedding.openai.max_retries', 3)
        
        return OpenAIEmbedder(
            model_name=model,
            batch_size=batch_size,
            max_retries=max_retries
        )
    
    @staticmethod
    def _create_local(config: Config) -> LocalEmbedder:
        """Create local embedder."""
        model = config.get('embedding.local.model', 'all-MiniLM-L6-v2')
        device = config.get('embedding.local.device', None)
        batch_size = config.get('embedding.local.batch_size', 32)
        
        return LocalEmbedder(
            model_name=model,
            device=device,
            batch_size=batch_size
        )
    
    @staticmethod
    def create_from_provider(
        provider: str,
        model_name: Optional[str] = None,
        **kwargs
    ) -> BaseEmbedder:
        """
        Create embedder directly from provider name (for testing).
        
        Args:
            provider: Provider name ('openai' or 'local')
            model_name: Optional model name override
            **kwargs: Additional arguments passed to embedder
            
        Returns:
            Embedder instance
        """
        if provider == "openai":
            model = model_name or "text-embedding-3-small"
            return OpenAIEmbedder(model_name=model, **kwargs)
        elif provider == "local":
            model = model_name or "all-MiniLM-L6-v2"
            return LocalEmbedder(model_name=model, **kwargs)
        else:
            raise ConfigurationError(
                f"Unknown provider: {provider}",
                details={'provider': provider}
            )
