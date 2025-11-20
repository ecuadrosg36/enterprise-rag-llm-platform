"""
API Schemas.

Pydantic models for request and response validation.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
    version: str
    environment: str


class RAGRequest(BaseModel):
    """RAG generation request."""

    query: str = Field(..., min_length=1, description="User query")
    top_k: int = Field(5, ge=1, le=20, description="Number of documents to retrieve")
    filter_metadata: Optional[Dict[str, Any]] = Field(
        None, description="Metadata filters"
    )


class SourceDocument(BaseModel):
    """Source document in response."""

    text: str
    metadata: Dict[str, Any]
    score: float
    id: str


class RAGResponse(BaseModel):
    """RAG generation response."""

    answer: str
    query: str
    source_documents: List[SourceDocument]
    processing_time_ms: Optional[float] = None


class EmbedRequest(BaseModel):
    """Embedding request."""

    text: str = Field(..., min_length=1, description="Text to embed")


class EmbedResponse(BaseModel):
    """Embedding response."""

    embedding: List[float]
    dimension: int
    model: str


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str
    detail: Optional[Dict[str, Any]] = None
