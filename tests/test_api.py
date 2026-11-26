"""
Unit tests for API endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

from src.api.app import app
from src.api.dependencies import get_rag_generator, get_embedder
from src.generation import RAGGenerator
from src.embeddings import BaseEmbedder


@pytest.fixture
def client():
    """Test client fixture."""
    return TestClient(app)


@pytest.fixture
def mock_rag_generator():
    """Mock RAG generator."""
    generator = Mock(spec=RAGGenerator)
    generator.generate.return_value = {
        "answer": "Test Answer",
        "query": "Test Query",
        "source_documents": [
            {
                "text": "Source Text",
                "metadata": {"source": "doc1"},
                "score": 0.9,
                "id": "1",
            }
        ],
    }
    return generator


@pytest.fixture
def mock_embedder():
    """Mock embedder."""
    embedder = Mock(spec=BaseEmbedder)
    embedder.embed_text.return_value = [0.1, 0.2, 0.3]
    embedder.dimension = 3
    embedder.model_name = "test-model"
    return embedder


def test_health_check(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data


def test_rag_endpoint(client, mock_rag_generator):
    """Test RAG generation endpoint."""
    # Override dependency
    app.dependency_overrides[get_rag_generator] = lambda: mock_rag_generator

    response = client.post("/rag", json={"query": "Test Query"})

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Test Answer"
    assert len(data["source_documents"]) == 1

    # Clean up
    app.dependency_overrides = {}


def test_embed_endpoint(client, mock_embedder):
    """Test embedding endpoint."""
    # Override dependency
    app.dependency_overrides[get_embedder] = lambda: mock_embedder

    response = client.post("/embed", json={"text": "Test Text"})

    assert response.status_code == 200
    data = response.json()
    assert len(data["embedding"]) == 3
    assert data["model"] == "test-model"

    # Clean up
    app.dependency_overrides = {}


def test_rag_requires_api_key_when_set(client, mock_rag_generator, monkeypatch):
    """With RAG_API_KEY set, /rag rejects a missing or wrong key and accepts the right one."""
    monkeypatch.setenv("RAG_API_KEY", "secret-key")
    app.dependency_overrides[get_rag_generator] = lambda: mock_rag_generator

    assert client.post("/rag", json={"query": "q"}).status_code == 401
    assert (
        client.post(
            "/rag", json={"query": "q"}, headers={"X-API-Key": "wrong"}
        ).status_code
        == 401
    )
    assert (
        client.post(
            "/rag", json={"query": "q"}, headers={"X-API-Key": "secret-key"}
        ).status_code
        == 200
    )

    app.dependency_overrides = {}


def test_rag_rejects_oversized_query(client, mock_rag_generator):
    app.dependency_overrides[get_rag_generator] = lambda: mock_rag_generator

    assert client.post("/rag", json={"query": "x" * 2001}).status_code == 422

    app.dependency_overrides = {}


def test_rag_error_does_not_leak_details(client, mock_rag_generator):
    mock_rag_generator.generate.side_effect = RuntimeError(
        "sk-secret leaked in upstream error"
    )
    app.dependency_overrides[get_rag_generator] = lambda: mock_rag_generator

    response = client.post("/rag", json={"query": "q"})

    assert response.status_code == 500
    assert "sk-secret" not in response.text
    assert "request_id=" in response.json()["detail"]

    app.dependency_overrides = {}


def test_prod_refuses_to_start_without_api_key(monkeypatch):
    import importlib

    app_module = importlib.import_module("src.api.app")

    monkeypatch.delenv("RAG_API_KEY", raising=False)
    monkeypatch.setattr(app_module.config, "_env", "prod")

    with pytest.raises(RuntimeError, match="RAG_API_KEY"):
        app_module.create_app()
