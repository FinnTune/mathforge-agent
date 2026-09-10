"""Tests for server.py's retrieval logic and MCP tool wiring.

Uses qdrant-client's embedded (":memory:") mode and a fake, deterministic
bag-of-words embedder — no Docker and no VOYAGE_API_KEY required. The real
Qdrant HTTP client and MCP tool plumbing are still exercised for real; only
the embedding call is faked.
"""

from __future__ import annotations

import math

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from server import Config, build_server, search

EMBED_DIM = 16


def fake_embed(text: str) -> list[float]:
    """Deterministic hashing-trick bag-of-words embedding for tests only."""
    vector = [0.0] * EMBED_DIM
    for word in text.lower().split():
        vector[hash(word) % EMBED_DIM] += 1.0
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


@pytest.fixture
def qdrant_with_corpus() -> QdrantClient:
    client = QdrantClient(location=":memory:")
    client.create_collection(
        "test-notes", vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE)
    )
    docs = [
        ("linear_algebra.md", "Eigenvalues", "eigenvalue matrix numpy linalg eig dominant"),
        ("plotting_matplotlib.md", "Saving a figure", "matplotlib savefig plots agg backend"),
        ("calculus_and_integration.md", "Definite integrals", "scipy integrate quad abserr"),
    ]
    points = [
        PointStruct(
            id=i,
            vector=fake_embed(text),
            payload={"source": source, "heading": heading, "text": text},
        )
        for i, (source, heading, text) in enumerate(docs)
    ]
    client.upsert("test-notes", points=points)
    return client


def test_search_returns_most_relevant_chunk_first(qdrant_with_corpus: QdrantClient) -> None:
    result = search(
        qdrant_with_corpus, "test-notes", fake_embed, "eigenvalue matrix numpy", top_k=3
    )
    lines = result.splitlines()
    assert "linear_algebra.md" in lines[0]
    assert "Eigenvalues" in lines[0]


def test_search_respects_top_k(qdrant_with_corpus: QdrantClient) -> None:
    result = search(qdrant_with_corpus, "test-notes", fake_embed, "matplotlib savefig", top_k=1)
    assert result.count("source:") == 1


def test_search_empty_collection_returns_message() -> None:
    client = QdrantClient(location=":memory:")
    client.create_collection(
        "empty", vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE)
    )
    result = search(client, "empty", fake_embed, "anything", top_k=3)
    assert "No matching notes" in result


@pytest.mark.asyncio
async def test_build_server_registers_search_tool(qdrant_with_corpus: QdrantClient) -> None:
    config = Config(
        qdrant_url="unused",
        qdrant_collection="test-notes",
        voyage_api_key="unused",
        voyage_model="voyage-3-lite",
    )
    mcp = build_server(config, qdrant=qdrant_with_corpus, embed_query=fake_embed)

    tools = await mcp.list_tools()
    assert [t.name for t in tools] == ["search_math_knowledge"]

    content = await mcp.call_tool(
        "search_math_knowledge", {"query": "scipy quad integrate", "top_k": 1}
    )
    text = content[0].text if isinstance(content, list) else str(content)
    assert "calculus_and_integration.md" in text
