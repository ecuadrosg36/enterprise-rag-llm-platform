"""
Base LLM interface.

Defines the contract for Large Language Model providers.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Generator

from src.core.logger import setup_logger


logger = setup_logger(__name__)


class BaseLLM(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, model_name: str):
        """
        Initialize LLM.

        Args:
            model_name: Name of the model
        """
        self.model_name = model_name
        logger.info(f"Initialized {self.__class__.__name__} with model: {model_name}")

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> str:
        """
        Generate text from prompt.

        Args:
            prompt: User prompt
            system_prompt: Optional system instruction
            temperature: Sampling temperature
            max_tokens: Max output tokens

        Returns:
            Generated text
        """
        pass

    @abstractmethod
    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> Generator[str, None, None]:
        """
        Stream generated text.

        Args:
            prompt: User prompt
            system_prompt: Optional system instruction
            temperature: Sampling temperature
            max_tokens: Max output tokens

        Yields:
            Text chunks
        """
        pass
