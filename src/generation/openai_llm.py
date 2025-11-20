"""
OpenAI/Groq LLM provider.

Supports any provider compatible with the OpenAI API format (OpenAI, Groq, etc.).
"""

import os
from typing import Optional, Generator
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from .base_llm import BaseLLM
from src.core.errors import GenerationError, RateLimitError
from src.core.logger import setup_logger


logger = setup_logger(__name__)


class OpenAILLM(BaseLLM):
    """OpenAI-compatible LLM provider (works with OpenAI, Groq, etc.)."""
    
    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_retries: int = 3
    ):
        """
        Initialize OpenAI LLM.
        
        Args:
            model_name: Model identifier
            api_key: API key (defaults to OPENAI_API_KEY or GROQ_API_KEY based on model)
            base_url: Optional API base URL (for local LLMs or proxies)
            max_retries: Number of retries
        """
        super().__init__(model_name)
        
        # Auto-detect API key if not provided
        if not api_key:
            if "groq" in model_name.lower() or "llama" in model_name.lower():
                api_key = os.getenv("GROQ_API_KEY")
                if not base_url:
                    base_url = "https://api.groq.com/openai/v1"
            else:
                api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            raise GenerationError(
                "API key not found. Set OPENAI_API_KEY or GROQ_API_KEY.",
                details={'model': model_name}
            )
        
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.max_retries = max_retries
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> str:
        """Generate text."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            content = response.choices[0].message.content
            logger.debug(f"Generated {len(content)} chars")
            return content
            
        except Exception as e:
            self._handle_error(e)
    
    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> Generator[str, None, None]:
        """Stream generated text."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            self._handle_error(e)
            
    def _handle_error(self, e: Exception):
        """Handle API errors."""
        logger.error(f"LLM generation failed: {e}")
        
        if "rate_limit" in str(e).lower():
            raise RateLimitError(
                f"Rate limit exceeded: {e}",
                details={'model': self.model_name}
            )
        
        raise GenerationError(
            f"Generation failed: {e}",
            details={'model': self.model_name, 'error': str(e)}
        )
