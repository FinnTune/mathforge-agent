"""MCP server exposing ``search_math_knowledge`` — retrieval over the corpus
embedded by ``ingest.py``.

Retrieval logic (``search``) is factored out from the FastMCP tool wiring and
takes an injectable ``embed_query`` function and Qdrant client, so
``tests/test_server.py`` can exercise the real query path against an
embedded (``:memory:``) Qdrant instance with a fake embedder — no Docker, no
Voyage API key needed to run the test suite.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass

import voyageai
from mcp.server.fastmcp import FastMCP
from qdrant_client import QdrantClient
from qdrant_client.models import ScoredPoint

DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_QDRANT_COLLECTION = "mathforge-math-notes"
DEFAULT_VOYAGE_MODEL = "voyage-3-lite"
DEFAULT_TOP_K = 3

EmbedQueryFn = Callable[[str], list[float]]


@dataclass(frozen=True)
class Config:
    qdrant_url: str
    qdrant_collection: str
    voyage_api_key: str
    voyage_model: str


def config_from_env() -> Config:
    return Config(
        qdrant_url=os.getenv("QDRANT_URL", DEFAULT_QDRANT_URL),
        qdrant_collection=os.getenv("QDRANT_COLLECTION", DEFAULT_QDRANT_COLLECTION),
        voyage_api_key=os.getenv("VOYAGE_API_KEY", ""),
        voyage_model=os.getenv("VOYAGE_MODEL", DEFAULT_VOYAGE_MODEL),
    )


def format_results(points: list[ScoredPoint]) -> str:
    if not points:
        return "No matching notes found in the knowledge base."
    blocks = []
    for point in points:
        payload = point.payload or {}
        blocks.append(
            f"[source: {payload.get('source', '?')} | section: {payload.get('heading', '?')} "
            f"| score: {point.score:.3f}]\n{payload.get('text', '')}"
        )
    return "\n\n".join(blocks)


def search(
    qdrant: QdrantClient,
    collection: str,
    embed_query: EmbedQueryFn,
    query: str,
    top_k: int,
) -> str:
    vector = embed_query(query)
    response = qdrant.query_points(collection_name=collection, query=vector, limit=top_k)
    return format_results(response.points)


def make_voyage_embed_query(voyage: voyageai.Client, model: str) -> EmbedQueryFn:
    def embed_query(query: str) -> list[float]:
        result = voyage.embed([query], model=model, input_type="query")
        return result.embeddings[0]

    return embed_query


def build_server(
    config: Config,
    *,
    qdrant: QdrantClient | None = None,
    embed_query: EmbedQueryFn | None = None,
) -> FastMCP:
    """Build the FastMCP server. ``qdrant``/``embed_query`` overrides are test-only hooks."""
    qdrant = qdrant or QdrantClient(url=config.qdrant_url)
    if embed_query is None:
        voyage = voyageai.Client(api_key=config.voyage_api_key)
        embed_query = make_voyage_embed_query(voyage, config.voyage_model)

    mcp = FastMCP("mathforge-mathkb")

    @mcp.tool()
    def search_math_knowledge(query: str, top_k: int = DEFAULT_TOP_K) -> str:
        """Search MathForge's curated reference notes (linear algebra, SciPy
        integration, SymPy, Matplotlib, numerical methods) for snippets
        relevant to `query`. Returns the top matches with their source file
        and section so you can cite them in your answer."""
        return search(qdrant, config.qdrant_collection, embed_query, query, top_k)

    return mcp


def main() -> None:
    config = config_from_env()
    mcp = build_server(config)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
