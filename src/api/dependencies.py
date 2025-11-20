"""
API Dependencies.

Dependency injection for RAG components.
"""

from functools import lru_cache
from fastapi import Request

from src.core.config import get_config, Config
from src.embeddings import EmbeddingFactory, BaseEmbedder, EmbeddingCache
from src.vector_store import ChromaVectorStore, BaseVectorStore
from src.retrieval import (
    BM25Retriever,
    HybridRetriever,
    ContextAssembler,
    BaseRetriever,
)
from src.generation import OpenAILLM, RAGGenerator, BaseLLM


"""
API Dependencies.

Dependency injection for RAG components.
"""

from functools import lru_cache
from fastapi import Request

from src.core.config import get_config, Config
from src.embeddings import EmbeddingFactory, BaseEmbedder, EmbeddingCache
from src.vector_store import ChromaVectorStore, BaseVectorStore
from src.retrieval import (
    BM25Retriever,
    HybridRetriever,
    ContextAssembler,
    BaseRetriever,
)
from src.generation import OpenAILLM, RAGGenerator, BaseLLM


@lru_cache()
def get_settings() -> Config:
    """Get global configuration."""
    return get_config()


def get_embedder() -> BaseEmbedder:
    """Get configured embedder instance."""
    config = get_settings()
    return EmbeddingFactory.create(config)


@lru_cache()
def get_vector_store() -> BaseVectorStore:
    """Get vector store instance."""
    config = get_settings()
    return ChromaVectorStore(config)


@lru_cache()
def get_bm25_retriever() -> BM25Retriever:
    """Get BM25 retriever instance."""
    config = get_settings()
    return BM25Retriever(persist_dir=config.vector_store_path)


@lru_cache()
def get_retriever() -> BaseRetriever:
    """Get hybrid retriever instance."""
    vector_store = get_vector_store()
    embedder = get_embedder()
    bm25_retriever = get_bm25_retriever()

    return HybridRetriever(
        vector_store=vector_store, embedder=embedder, bm25_retriever=bm25_retriever
    )


@lru_cache()
def get_llm() -> BaseLLM:
    """Get LLM instance."""
    config = get_settings()

    # Currently only OpenAI/Groq supported via OpenAILLM
    # Could expand factory logic here if needed
    return OpenAILLM(
        model_name=config.llm_model, api_key=None  # Auto-detected from env
    )


@lru_cache()
def get_rag_generator() -> RAGGenerator:
    """Get RAG generator instance."""
    llm = get_llm()
    retriever = get_retriever()

    assembler = ContextAssembler()

    return RAGGenerator(llm=llm, retriever=retriever, context_assembler=assembler)
