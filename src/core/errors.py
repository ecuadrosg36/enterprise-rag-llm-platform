"""
Centralized exception classes for Enterprise RAG Platform.

All custom exceptions inherit from RAGError base class for
easy exception handling and logging.
"""


class RAGError(Exception):
    """Base exception for all RAG-related errors."""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self):
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


class ConfigurationError(RAGError):
    """Configuration loading or validation error."""

    pass


class DocumentLoadError(RAGError):
    """Error loading or parsing documents."""

    pass


class ChunkingError(RAGError):
    """Error during text chunking."""

    pass


class EmbeddingError(RAGError):
    """Error generating embeddings."""

    pass


class IndexingError(RAGError):
    """Error indexing documents to vector store."""

    pass


class RetrievalError(RAGError):
    """Error retrieving documents from vector store."""

    pass


class GenerationError(RAGError):
    """Error generating LLM response."""

    pass


class ValidationError(RAGError):
    """Input validation error."""

    pass


class APIError(RAGError):
    """API-related error."""

    pass


class RateLimitError(RAGError):
    """Rate limit exceeded error (for external APIs)."""

    pass


class EvaluationError(RAGError):
    """Error during RAG evaluation."""

    pass
