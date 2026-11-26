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
)
from src.generation import RAGGenerator
from src.embeddings import BaseEmbedder
from src.api.dependencies import get_rag_generator, get_embedder, require_api_key
from src.core.logger import setup_logger, correlation_id_var

logger = setup_logger(__name__)
router = APIRouter(tags=["RAG"], dependencies=[Depends(require_api_key)])


def _internal_error(what: str, error: Exception) -> HTTPException:
    # Log the real error; the client gets only an ID to quote.
    logger.error(f"{what} failed: {error}", exc_info=True)
    return HTTPException(
        status_code=500,
        detail=f"{what} failed. request_id={correlation_id_var.get()}",
    )


# Plain `def`, not `async def`: retrieval and the LLM call block, so FastAPI runs these in a thread pool.
@router.post("/rag", response_model=RAGResponse)
def generate_rag_response(
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
    except Exception as e:
        raise _internal_error("RAG generation", e)

    return RAGResponse(
        answer=result["answer"],
        query=result["query"],
        source_documents=result["source_documents"],
        processing_time_ms=(time.time() - start_time) * 1000,
    )


@router.post("/embed", response_model=EmbedResponse)
def generate_embedding(
    request: EmbedRequest, embedder: BaseEmbedder = Depends(get_embedder)
):
    """
    Generate text embedding.
    """
    try:
        embedding = embedder.embed_text(request.text)
    except Exception as e:
        raise _internal_error("Embedding", e)

    return EmbedResponse(
        embedding=embedding, dimension=embedder.dimension, model=embedder.model_name
    )
