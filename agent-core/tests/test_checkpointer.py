"""Tests for ``checkpointer.build_checkpointer`` against a real (tmp_path) SQLite file.

No mocking needed — ``AsyncSqliteSaver`` against a temp file is fast and
fully self-contained; this is what actually proves persistence works, not
just that the graph accepts a ``BaseCheckpointSaver``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from checkpointer import build_checkpointer
from config import Settings


def _settings_with_db(db_path: Path) -> Settings:
    return Settings(
        anthropic_api_key="test-key-not-used",
        model="claude-sonnet-4-6",
        temperature=0.0,
        max_tokens=None,
        recursion_limit=15,
        code_timeout_sec=5.0,
        workspace_root=".",
        log_level="DEBUG",
        sandbox_mcp_bin="/nonexistent/mathforge-sandbox-mcp",
        sandbox_python="python3",
        sandbox_max_memory_mb=512,
        sandbox_max_output_bytes=256_000,
        mathkb_mcp_python="/nonexistent/mathkb-python",
        mathkb_mcp_script="/nonexistent/server.py",
        qdrant_url="http://localhost:6333",
        qdrant_collection="test-notes",
        voyage_api_key="",
        voyage_model="voyage-3-lite",
        checkpoint_db_path=str(db_path),
        verification_max_attempts=2,
        grpc_host="127.0.0.1",
        grpc_port=50051,
        tls_cert_path=None,
        tls_key_path=None,
        otlp_endpoint=None,
    )


@pytest.mark.asyncio
async def test_build_checkpointer_creates_db_file_and_parent_dir(tmp_path: Path) -> None:
    db_path = tmp_path / "nested" / "checkpoints.sqlite3"
    settings = _settings_with_db(db_path)

    async with build_checkpointer(settings):
        pass

    assert db_path.is_file()


@pytest.mark.asyncio
async def test_checkpointer_persists_across_separate_sessions(tmp_path: Path) -> None:
    """A second `build_checkpointer` against the same path can read what the first wrote."""
    from langgraph.checkpoint.base import empty_checkpoint

    db_path = tmp_path / "checkpoints.sqlite3"
    settings = _settings_with_db(db_path)
    config = {"configurable": {"thread_id": "t1", "checkpoint_ns": ""}}

    async with build_checkpointer(settings) as first_session:
        checkpoint = empty_checkpoint()
        checkpoint["channel_values"] = {"marker": "written-by-first-session"}
        await first_session.aput(config, checkpoint, {}, {})

    async with build_checkpointer(settings) as second_session:
        tuple_ = await second_session.aget_tuple(config)

    assert tuple_ is not None
    assert tuple_.checkpoint["channel_values"]["marker"] == "written-by-first-session"
