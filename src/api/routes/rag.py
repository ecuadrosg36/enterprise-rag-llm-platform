"""
RAG endpoints.
"""

import time
from fastapi import APIRouter, Depends, HTTPException
from src.api.schemas import (
    RAGRequest,
    RAGResponse,
    EmbedRequest,
    EmbedResponse,
    ErrorResponse,
)
from src.generation import RAGGenerator
from src.embeddings import BaseEmbedder
from src.api.dependencies import get_rag_generator, get_embedder
from src.core.logger import setup_logger

logger = setup_logger(__name__)
router = APIRouter(tags=["RAG"])


@router.post(
    "/rag", response_model=RAGResponse, responses={500: {"model": ErrorResponse}}
)
async def generate_rag_response(
    request: RAGRequest, generator: RAGGenerator = Depends(get_rag_generator)
):
    """
    Generate answer using RAG.
    """
    start_time = time.time()

    try:
        result = generator.generate(
            query=request.query,
            top_k=request.top_k,
            filter_metadata=request.filter_metadata,
        )

        processing_time = (time.time() - start_time) * 1000

        return RAGResponse(
            answer=result["answer"],
            query=result["query"],
            source_documents=result["source_documents"],
            processing_time_ms=processing_time,
        )

    except Exception as e:
        logger.error(f"RAG endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/embed", response_model=EmbedResponse)
async def generate_embedding(
    request: EmbedRequest, embedder: BaseEmbedder = Depends(get_embedder)
):
    """
    Generate text embedding.
    """
    try:
        embedding = embedder.embed_text(request.text)

        return EmbedResponse(
            embedding=embedding, dimension=embedder.dimension, model=embedder.model_name
        )

    except Exception as e:
        logger.error(f"Embedding endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
