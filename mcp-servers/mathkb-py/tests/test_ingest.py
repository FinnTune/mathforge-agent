"""Tests for ingest.py against an embedded (":memory:") Qdrant instance.

No VOYAGE_API_KEY / network needed — embedding is faked with a stand-in
object exposing the same `.embed(...)` shape as `voyageai.Client`.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from qdrant_client import QdrantClient

from chunking import Chunk
from ingest import (
    EMBEDDING_DIMENSIONS,
    _embedding_dimension,
    embed_chunks,
    ensure_collection,
    upsert_chunks,
)
from server import Config


@dataclass
class _FakeEmbeddingsResult:
    embeddings: list[list[float]]


class _FakeVoyageClient:
    """Duck-types voyageai.Client: returns a fixed-size zero vector per text."""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim
        self.calls: list[tuple[list[str], str, str]] = []

    def embed(self, texts: list[str], model: str, input_type: str) -> _FakeEmbeddingsResult:
        self.calls.append((texts, model, input_type))
        vectors = [[float(len(t))] + [0.0] * (self.dim - 1) for t in texts]
        return _FakeEmbeddingsResult(embeddings=vectors)


def _config(collection: str = "test-notes") -> Config:
    return Config(
        qdrant_url="unused",
        qdrant_collection=collection,
        voyage_api_key="unused",
        voyage_model="voyage-3-lite",
    )


def test_embedding_dimension_known_model() -> None:
    assert _embedding_dimension("voyage-3-lite") == EMBEDDING_DIMENSIONS["voyage-3-lite"]


def test_embedding_dimension_unknown_model_raises() -> None:
    with pytest.raises(ValueError, match="Unknown VOYAGE_MODEL"):
        _embedding_dimension("not-a-real-model")


def test_embed_chunks_batches_and_uses_document_input_type() -> None:
    fake = _FakeVoyageClient(dim=4)
    chunks = [
        Chunk(text="alpha", source="a.md", heading="A"),
        Chunk(text="beta", source="a.md", heading="B"),
    ]

    vectors = embed_chunks(fake, "voyage-3-lite", chunks)

    assert len(vectors) == 2
    assert fake.calls == [(["alpha", "beta"], "voyage-3-lite", "document")]


def test_embed_chunks_empty_list_skips_api_call() -> None:
    fake = _FakeVoyageClient()
    assert embed_chunks(fake, "voyage-3-lite", []) == []
    assert fake.calls == []


def test_ensure_collection_creates_once_and_is_idempotent() -> None:
    client = QdrantClient(location=":memory:")
    config = _config()
    assert not client.collection_exists(config.qdrant_collection)

    ensure_collection(client, config)
    assert client.collection_exists(config.qdrant_collection)

    ensure_collection(client, config)  # second call must not raise
    assert client.collection_exists(config.qdrant_collection)


def test_upsert_chunks_is_idempotent_on_unchanged_corpus() -> None:
    client = QdrantClient(location=":memory:")
    config = _config()
    ensure_collection(client, config)

    chunks = [
        Chunk(text="eigenvalues note", source="linear_algebra.md", heading="Eigenvalues"),
        Chunk(text="quad note", source="calculus_and_integration.md", heading="Definite integrals"),
    ]
    vectors = [[1.0] + [0.0] * 511, [0.0, 1.0] + [0.0] * 510]

    upsert_chunks(client, config, chunks, vectors)
    first_count = client.count(config.qdrant_collection).count
    assert first_count == 2

    # Re-ingesting the same corpus should overwrite in place, not duplicate.
    upsert_chunks(client, config, chunks, vectors)
    assert client.count(config.qdrant_collection).count == 2
