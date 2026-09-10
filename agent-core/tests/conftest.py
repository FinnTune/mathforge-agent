"""Pytest fixtures shared across test modules."""

from __future__ import annotations

import pytest

from config import Settings


@pytest.fixture
def dummy_settings(tmp_path) -> Settings:
    """Valid ``Settings`` for building an agent with a fake tool (no MCP server involved)."""
    return Settings(
        anthropic_api_key="test-key-not-used",
        model="claude-sonnet-4-6",
        temperature=0.0,
        max_tokens=None,
        recursion_limit=15,
        code_timeout_sec=5.0,
        workspace_root=str(tmp_path),
        log_level="DEBUG",
        sandbox_mcp_bin="/nonexistent/mathforge-sandbox-mcp",
        sandbox_python="python3",
        sandbox_max_memory_mb=512,
        sandbox_max_output_bytes=256_000,
    )
