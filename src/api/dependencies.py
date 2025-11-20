"""
API Dependencies.

Dependency injection for RAG components.
"""

from functools import lru_cache
from fastapi import Request

from src.core.config import get_config, Config
from src.embeddings import EmbeddingFactory, BaseEmbedder, EmbeddingCache
from src.vector_store import ChromaVectorStore, BaseVectorStore
from src.retrieval import BM25Retriever, HybridRetriever, ContextAssembler, BaseRetriever
from src.generation import OpenAILLM, RAGGenerator, BaseLLM


@lru_cache()
def get_settings() -> Config:
    """Get global configuration."""
    return get_config()


@lru_cache()
def get_embedder(config: Config = None) -> BaseEmbedder:
    """Get configured embedder instance."""
    if config is None:
        config = get_settings()
    return EmbeddingFactory.create(config)


@lru_cache()
def get_vector_store(config: Config = None) -> BaseVectorStore:
    """Get vector store instance."""
    if config is None:
        config = get_settings()
    return ChromaVectorStore(config)


@lru_cache()
def get_bm25_retriever(config: Config = None) -> BM25Retriever:
    """Get BM25 retriever instance."""
    if config is None:
        config = get_settings()
    return BM25Retriever(persist_dir=config.vector_store_path)


@lru_cache()
def get_retriever(
    vector_store: BaseVectorStore = None,
    embedder: BaseEmbedder = None,
    bm25_retriever: BM25Retriever = None
) -> BaseRetriever:
    """Get hybrid retriever instance."""
    if vector_store is None:
        vector_store = get_vector_store()
    if embedder is None:
        embedder = get_embedder()
    if bm25_retriever is None:
        bm25_retriever = get_bm25_retriever()
        
    return HybridRetriever(
        vector_store=vector_store,
        embedder=embedder,
        bm25_retriever=bm25_retriever
    )


@lru_cache()
def get_llm(config: Config = None) -> BaseLLM:
    """Get LLM instance."""
    if config is None:
        config = get_settings()
        
    # Currently only OpenAI/Groq supported via OpenAILLM
    # Could expand factory logic here if needed
    return OpenAILLM(
        model_name=config.llm_model,
        api_key=None  # Auto-detected from env
    )


@lru_cache()
def get_rag_generator(
    llm: BaseLLM = None,
    retriever: BaseRetriever = None
) -> RAGGenerator:
    """Get RAG generator instance."""
    if llm is None:
        llm = get_llm()
    if retriever is None:
        retriever = get_retriever()
        
    assembler = ContextAssembler()
    
    return RAGGenerator(
        llm=llm,
        retriever=retriever,
        context_assembler=assembler
    )
