"""
Context Assembler.

Formats retrieved documents into a context string for the LLM.
"""

from typing import List
from src.vector_store.base_store import SearchResult
from src.core.logger import setup_logger


logger = setup_logger(__name__)


class ContextAssembler:
    """
    Assembles context from retrieved documents.
    """
    
    def __init__(self, max_tokens: int = 4000):
        """
        Initialize assembler.
        
        Args:
            max_tokens: Maximum tokens for context (approximate)
        """
        self.max_tokens = max_tokens
    
    def assemble(self, results: List[SearchResult]) -> str:
        """
        Format search results into a single context string.
        
        Args:
            results: List of search results
            
        Returns:
            Formatted context string
        """
        if not results:
            return ""
        
        context_parts = []
        current_length = 0
        
        for i, result in enumerate(results):
            # Simple format:
            # [Document 1] (Source: file.pdf)
            # Content...
            
            source = result.document.metadata.get('filename', 'Unknown')
            page = result.document.metadata.get('page', '')
            source_info = f"{source} p.{page}" if page else source
            
            doc_text = (
                f"[Document {i+1}] (Source: {source_info})\n"
                f"{result.document.text}\n"
            )
            
            # Rough token estimation (4 chars per token)
            doc_tokens = len(doc_text) / 4
            
            if current_length + doc_tokens > self.max_tokens:
                logger.info(f"Context limit reached at document {i+1}")
                break
            
            context_parts.append(doc_text)
            current_length += doc_tokens
        
        return "\n".join(context_parts)
