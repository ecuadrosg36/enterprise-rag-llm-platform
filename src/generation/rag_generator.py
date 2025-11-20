"""
RAG Generator.

Orchestrates the retrieval and generation process.
"""

from typing import Dict, Any, Optional, Generator

from .base_llm import BaseLLM
from .prompt_templates import RAG_SYSTEM_PROMPT, RAG_USER_PROMPT
from src.retrieval.base_retriever import BaseRetriever
from src.retrieval.context_assembler import ContextAssembler
from src.core.logger import setup_logger


logger = setup_logger(__name__)


class RAGGenerator:
    """
    RAG Generator component.

    Combines retrieval and generation to answer queries.
    """

    def __init__(
        self,
        llm: BaseLLM,
        retriever: BaseRetriever,
        context_assembler: ContextAssembler,
    ):
        """
        Initialize RAG Generator.

        Args:
            llm: LLM provider
            retriever: Document retriever
            context_assembler: Context formatter
        """
        self.llm = llm
        self.retriever = retriever
        self.assembler = context_assembler

    def generate(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generate answer for query.

        Args:
            query: User query
            top_k: Number of documents to retrieve
            filter_metadata: Metadata filters

        Returns:
            Dictionary containing answer, source documents, and metadata
        """
        logger.info(f"Processing RAG query: {query}")

        # 1. Retrieve documents
        results = self.retriever.retrieve(
            query=query, top_k=top_k, filter_metadata=filter_metadata
        )

        # 2. Assemble context
        context = self.assembler.assemble(results)

        # 3. Generate response
        if not context:
            logger.warning("No context found for query")
            answer = "I couldn't find any relevant information in the knowledge base to answer your question."
        else:
            prompt = RAG_USER_PROMPT.format(context=context, query=query)
            answer = self.llm.generate(prompt=prompt, system_prompt=RAG_SYSTEM_PROMPT)

        return {
            "answer": answer,
            "query": query,
            "source_documents": [
                {
                    "text": r.document.text[:200] + "...",
                    "metadata": r.document.metadata,
                    "score": r.score,
                    "id": r.id,
                }
                for r in results
            ],
        }
