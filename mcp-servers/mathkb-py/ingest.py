"""Embeds ``corpus/*.md`` and upserts them into Qdrant.

Run manually (not on server start — ingestion is a separate, explicit step
from serving, same as any RAG pipeline):

    python ingest.py

Reads ``VOYAGE_API_KEY`` (required), and ``QDRANT_URL``/``QDRANT_COLLECTION``/
``VOYAGE_MODEL`` (optional, see defaults in ``server.py``'s ``Config``).
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

import voyageai
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from chunking import Chunk, chunk_corpus
from server import Config, config_from_env

CORPUS_DIR = Path(__file__).resolve().parent / "corpus"

# voyage-3-lite embeds at 512 dimensions; keep in sync with VOYAGE_MODEL default.
EMBEDDING_DIMENSIONS = {
    "voyage-3-lite": 512,
    "voyage-3": 1024,
    "voyage-3-large": 1024,
}


def _embedding_dimension(model: str) -> int:
    try:
        return EMBEDDING_DIMENSIONS[model]
    except KeyError:
        msg = (
            f"Unknown VOYAGE_MODEL {model!r}; add its embedding dimension to "
            "EMBEDDING_DIMENSIONS in ingest.py."
        )
        raise ValueError(msg) from None


def ensure_collection(client: QdrantClient, config: Config) -> None:
    if client.collection_exists(config.qdrant_collection):
        return
    client.create_collection(
        collection_name=config.qdrant_collection,
        vectors_config=VectorParams(
            size=_embedding_dimension(config.voyage_model),
            distance=Distance.COSINE,
        ),
    )


def embed_chunks(voyage: voyageai.Client, model: str, chunks: list[Chunk]) -> list[list[float]]:
    if not chunks:
        return []
    result = voyage.embed([c.text for c in chunks], model=model, input_type="document")
    return result.embeddings


def upsert_chunks(
    client: QdrantClient, config: Config, chunks: list[Chunk], vectors: list[list[float]]
) -> None:
    points = [
        PointStruct(
            # Deterministic id from source+heading+index so re-running ingest
            # on unchanged corpus content overwrites in place instead of
            # accumulating duplicates.
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{chunk.source}#{chunk.heading}#{i}")),
            vector=vector,
            payload={"text": chunk.text, "source": chunk.source, "heading": chunk.heading},
        )
        for i, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True))
    ]
    client.upsert(collection_name=config.qdrant_collection, points=points)


def main() -> int:
    config = config_from_env()
    if not config.voyage_api_key:
        print("VOYAGE_API_KEY is not set.", file=sys.stderr)
        return 1

    chunks = chunk_corpus(CORPUS_DIR)
    if not chunks:
        print(f"No chunks found under {CORPUS_DIR}.", file=sys.stderr)
        return 1

    voyage = voyageai.Client(api_key=config.voyage_api_key)
    vectors = embed_chunks(voyage, config.voyage_model, chunks)

    client = QdrantClient(url=config.qdrant_url)
    ensure_collection(client, config)
    upsert_chunks(client, config, chunks, vectors)

    print(
        f"Upserted {len(chunks)} chunks into {config.qdrant_collection!r} at {config.qdrant_url}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
