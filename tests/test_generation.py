"""
Unit tests for generation layer.
"""

import pytest
from unittest.mock import Mock, MagicMock

from src.generation import OpenAILLM, RAGGenerator
from src.retrieval.base_retriever import BaseRetriever
from src.retrieval.context_assembler import ContextAssembler
from src.vector_store.base_store import SearchResult
from src.ingestion.base_loader import Document


class TestOpenAILLM:
    """Tests for OpenAI LLM wrapper."""

    def test_initialization(self):
        """Test init with API key."""
        llm = OpenAILLM(api_key="test-key", model_name="gpt-3.5-turbo")
        assert llm.model_name == "gpt-3.5-turbo"

    def test_generate(self):
        """Test generation (mocked)."""
        llm = OpenAILLM(api_key="test-key")

        # Mock OpenAI client
        llm.client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Test response"))]
        llm.client.chat.completions.create.return_value = mock_response

        response = llm.generate("Hello")
        assert response == "Test response"


class TestRAGGenerator:
    """Tests for RAG Generator."""

    def test_rag_flow(self):
        """Test full RAG flow."""
        # Mock dependencies
        llm = Mock()
        llm.generate.return_value = "Generated Answer"

        retriever = Mock(spec=BaseRetriever)
        retriever.retrieve.return_value = [
            SearchResult(
                document=Document(text="Context info", metadata={"source": "doc1"}),
                score=0.9,
                id="1",
            )
        ]

        assembler = ContextAssembler()

        generator = RAGGenerator(llm, retriever, assembler)

        result = generator.generate("Query")

        assert result["answer"] == "Generated Answer"
        assert len(result["source_documents"]) == 1
        assert result["source_documents"][0]["metadata"]["source"] == "doc1"

        # Verify calls
        retriever.retrieve.assert_called_once()
        llm.generate.assert_called_once()

    def test_no_context_flow(self):
        """Test flow when no documents found."""
        llm = Mock()
        retriever = Mock()
        retriever.retrieve.return_value = []  # No results
        assembler = ContextAssembler()

        generator = RAGGenerator(llm, retriever, assembler)

        result = generator.generate("Query")

        assert "couldn't find" in result["answer"]
        assert len(result["source_documents"]) == 0
        llm.generate.assert_not_called()  # Should skip LLM call
