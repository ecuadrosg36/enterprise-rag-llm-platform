"""
Unit tests for core configuration loader.
"""

import pytest
import tempfile
from pathlib import Path
import yaml

from src.core.config import Config, get_config
from src.core.errors import ConfigurationError


@pytest.fixture
def temp_config_dir():
    """Create temporary config directory with test configs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir)

        # Create base config
        base_config = {
            "app": {"name": "test-app", "version": "1.0.0"},
            "embedding": {"provider": "local"},
            "vector_store": {"persist_directory": "./data/test"},
            "llm": {"provider": "openai"},
            "paths": {"raw_documents": "./data/raw"},
        }

        with open(config_dir / "config.yaml", "w") as f:
            yaml.dump(base_config, f)

        # Create dev config override
        dev_config = {
            "embedding": {"provider": "openai"},
            "llm": {"openai": {"model": "gpt-3.5-turbo"}},
        }

        with open(config_dir / "config.dev.yaml", "w") as f:
            yaml.dump(dev_config, f)

        yield config_dir


def test_config_load_base(temp_config_dir):
    """Test loading base configuration."""
    config = Config.load(env="prod", config_dir=temp_config_dir)

    assert config.app_name == "test-app"
    assert config.app_version == "1.0.0"
    assert config.embedding_provider == "local"


def test_config_env_override(temp_config_dir):
    """Test environment-specific overrides."""
    config = Config.load(env="dev", config_dir=temp_config_dir)

    # Should be overridden by dev config
    assert config.embedding_provider == "openai"
    assert config.get("llm.openai.model") == "gpt-3.5-turbo"


def test_config_dot_notation(temp_config_dir):
    """Test dot notation for nested configs."""
    config = Config.load(env="dev", config_dir=temp_config_dir)

    assert config.get("app.name") == "test-app"
    assert config.get("embedding.provider") == "openai"
    assert config.get("nonexistent.key", "default") == "default"


def test_config_typed_accessors(temp_config_dir):
    """Test typed property accessors."""
    config = Config.load(env="dev", config_dir=temp_config_dir)

    assert isinstance(config.vector_store_path, Path)
    assert isinstance(config.app_name, str)
    assert isinstance(config.environment, str)


def test_config_missing_base_file():
    """Test error when base config file missing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir)

        with pytest.raises(ConfigurationError, match="Base config file not found"):
            Config.load(config_dir=config_dir)


def test_config_missing_required_section():
    """Test validation of required sections."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir)

        # Create incomplete config
        incomplete_config = {"app": {"name": "test"}}

        with open(config_dir / "config.yaml", "w") as f:
            yaml.dump(incomplete_config, f)

        with pytest.raises(ConfigurationError, match="Missing required config section"):
            Config.load(config_dir=config_dir)


def test_config_singleton(temp_config_dir):
    """Test global config singleton."""
    config1 = get_config(env="dev")
    config2 = get_config()

    # Should return same instance
    assert config1 is config2


def test_config_to_dict(temp_config_dir):
    """Test converting config to dictionary."""
    config = Config.load(env="dev", config_dir=temp_config_dir)
    config_dict = config.to_dict()

    assert isinstance(config_dict, dict)
    assert "app" in config_dict
    assert "embedding" in config_dict
