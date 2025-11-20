"""
Type-safe configuration loader with YAML support and environment overrides.

Loads config.yaml + config.{env}.yaml and provides typed accessors.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field

from .errors import ConfigurationError


@dataclass
class Config:
    """Type-safe configuration container."""

    _data: Dict[str, Any] = field(default_factory=dict)
    _env: str = "dev"

    @classmethod
    def load(
        cls, env: Optional[str] = None, config_dir: Optional[Path] = None
    ) -> "Config":
        """
        Load configuration from YAML files.

        Args:
            env: Environment name (dev, prod). Defaults to ENV environment variable or "dev"
            config_dir: Path to config directory. Defaults to ./config

        Returns:
            Loaded Config instance

        Raises:
            ConfigurationError: If config files not found or invalid
        """
        # Determine environment
        if env is None:
            env = os.getenv("ENV", "dev")

        # Determine config directory
        if config_dir is None:
            # Assume config dir is in project root
            project_root = Path(__file__).parent.parent.parent
            config_dir = project_root / "config"

        # Load base config
        base_config_path = config_dir / "config.yaml"
        if not base_config_path.exists():
            raise ConfigurationError(f"Base config file not found: {base_config_path}")

        try:
            with open(base_config_path, "r") as f:
                config_data = yaml.safe_load(f) or {}
        except Exception as e:
            raise ConfigurationError(
                f"Failed to load base config: {e}",
                details={"path": str(base_config_path)},
            )

        # Load environment-specific overrides
        env_config_path = config_dir / f"config.{env}.yaml"
        if env_config_path.exists():
            try:
                with open(env_config_path, "r") as f:
                    env_config = yaml.safe_load(f) or {}

                # Merge configs (env overrides base)
                config_data = cls._merge_dicts(config_data, env_config)
            except Exception as e:
                raise ConfigurationError(
                    f"Failed to load environment config: {e}",
                    details={"path": str(env_config_path)},
                )

        # Create config instance
        config = cls(_data=config_data, _env=env)

        # Validate required fields
        config._validate()

        return config

    @staticmethod
    def _merge_dicts(base: Dict, override: Dict) -> Dict:
        """Recursively merge two dictionaries."""
        result = base.copy()

        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = Config._merge_dicts(result[key], value)
            else:
                result[key] = value

        return result

    def _validate(self):
        """Validate required configuration fields."""
        required_sections = ["app", "embedding", "vector_store", "llm", "paths"]

        for section in required_sections:
            if section not in self._data:
                raise ConfigurationError(f"Missing required config section: {section}")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.

        Example:
            config.get('embedding.provider') -> 'openai'
            config.get('llm.openai.model') -> 'gpt-4-turbo-preview'
        """
        keys = key.split(".")
        value = self._data

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value

    # Typed property accessors for common configs

    @property
    def environment(self) -> str:
        """Current environment (dev, prod)."""
        return self._env

    @property
    def app_name(self) -> str:
        """Application name."""
        return self.get("app.name", "enterprise-rag-platform")

    @property
    def app_version(self) -> str:
        """Application version."""
        return self.get("app.version", "1.0.0")

    # Embedding config

    @property
    def embedding_provider(self) -> str:
        """Embedding provider (openai, local)."""
        return self.get("embedding.provider", "local")

    @property
    def embedding_model(self) -> str:
        """Embedding model name."""
        provider = self.embedding_provider
        return self.get(f"embedding.{provider}.model", "all-MiniLM-L6-v2")

    @property
    def embedding_dimension(self) -> int:
        """Embedding dimension (inferred from model)."""
        model = self.embedding_model

        # Map common models to dimensions
        dimension_map = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
            "all-MiniLM-L6-v2": 384,
            "all-mpnet-base-v2": 768,
        }

        return dimension_map.get(model, 384)

    # Vector store config

    @property
    def vector_store_path(self) -> Path:
        """Vector store persistence directory."""
        return Path(self.get("vector_store.persist_directory", "./data/chromadb"))

    @property
    def collection_name(self) -> str:
        """Vector store collection name."""
        return self.get("vector_store.collection_name", "enterprise_docs")

    # LLM config

    @property
    def llm_provider(self) -> str:
        """LLM provider (openai, groq)."""
        return self.get("llm.provider", "openai")

    @property
    def llm_model(self) -> str:
        """LLM model name."""
        provider = self.llm_provider
        return self.get(f"llm.{provider}.model", "gpt-3.5-turbo")

    @property
    def llm_temperature(self) -> float:
        """LLM temperature."""
        provider = self.llm_provider
        return self.get(f"llm.{provider}.temperature", 0.7)

    @property
    def llm_max_tokens(self) -> int:
        """LLM max tokens."""
        provider = self.llm_provider
        return self.get(f"llm.{provider}.max_tokens", 1000)

    # Retrieval config

    @property
    def top_k(self) -> int:
        """Number of documents to retrieve."""
        return self.get("retrieval.top_k", 5)

    @property
    def use_hybrid_search(self) -> bool:
        """Whether to use hybrid search."""
        return self.get("retrieval.use_hybrid", True)

    @property
    def vector_weight(self) -> float:
        """Vector search weight for hybrid retrieval."""
        return self.get("retrieval.vector_weight", 0.7)

    # Logging config

    @property
    def log_level(self) -> str:
        """Logging level."""
        return self.get("logging.level", "INFO")

    @property
    def log_to_file(self) -> bool:
        """Whether to log to file."""
        return self.get("logging.file.enabled", True)

    @property
    def log_file_path(self) -> Path:
        """Log file path."""
        return Path(self.get("logging.file.path", "./logs/rag.log"))

    # Paths

    @property
    def raw_documents_path(self) -> Path:
        """Raw documents directory."""
        return Path(self.get("paths.raw_documents", "./data/raw"))

    @property
    def processed_documents_path(self) -> Path:
        """Processed documents directory."""
        return Path(self.get("paths.processed_documents", "./data/processed"))

    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary."""
        return self._data.copy()


# Global config instance (lazy loaded)
_config: Optional[Config] = None


def get_config(env: Optional[str] = None, reload: bool = False) -> Config:
    """
    Get global config instance (singleton pattern).

    Args:
        env: Environment name (optional, defaults to ENV env variable)
        reload: Force reload configuration

    Returns:
        Config instance
    """
    global _config

    if _config is None or reload:
        _config = Config.load(env=env)

    return _config
