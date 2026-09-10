"""Loads tools from MathForge's MCP servers as LangChain tools.

Two servers today, both spawned over stdio by ``langchain-mcp-adapters``:

- ``sandbox`` (Rust, ``mcp-servers/sandbox-rs``) — ``execute_python``.
- ``mathkb`` (Python, ``mcp-servers/mathkb-py``) — ``search_math_knowledge``.

Per ``langchain-mcp-adapters``' own docs, a fresh session (and subprocess) is
opened for *each* tool call and torn down afterward, so callers just need the
combined tool list from ``load_mcp_tools()`` once; there is no client
lifecycle to manage across the app's lifetime.
"""

from __future__ import annotations

import os

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from config import Settings


def _require_file(path: str, *, fix: str) -> None:
    if not os.path.isfile(path):
        msg = f"Expected a file at {path!r} but found none. {fix}"
        raise FileNotFoundError(msg)


def _sandbox_connection(settings: Settings) -> dict:
    _require_file(
        settings.sandbox_mcp_bin,
        fix=(
            "Build it first: cargo build --release "
            "--manifest-path mcp-servers/sandbox-rs/Cargo.toml"
        ),
    )
    env = {
        "PATH": os.environ.get("PATH", ""),
        "MATHFORGE_WORKSPACE_ROOT": settings.workspace_root,
        "MATHFORGE_CODE_TIMEOUT_SEC": str(settings.code_timeout_sec),
        "MATHFORGE_SANDBOX_PYTHON": settings.sandbox_python,
        "MATHFORGE_SANDBOX_MAX_MEMORY_MB": str(settings.sandbox_max_memory_mb),
        "MATHFORGE_SANDBOX_MAX_OUTPUT_BYTES": str(settings.sandbox_max_output_bytes),
    }
    return {"transport": "stdio", "command": settings.sandbox_mcp_bin, "args": [], "env": env}


def _mathkb_connection(settings: Settings) -> dict:
    _require_file(
        settings.mathkb_mcp_python,
        fix=(
            "Set up its venv first: python3 -m venv mcp-servers/mathkb-py/.venv && "
            "mcp-servers/mathkb-py/.venv/bin/pip install -e mcp-servers/mathkb-py"
        ),
    )
    _require_file(
        settings.mathkb_mcp_script, fix="Expected mcp-servers/mathkb-py/server.py to exist."
    )
    env = {
        "PATH": os.environ.get("PATH", ""),
        "QDRANT_URL": settings.qdrant_url,
        "QDRANT_COLLECTION": settings.qdrant_collection,
        "VOYAGE_API_KEY": settings.voyage_api_key,
        "VOYAGE_MODEL": settings.voyage_model,
    }
    return {
        "transport": "stdio",
        "command": settings.mathkb_mcp_python,
        "args": [settings.mathkb_mcp_script],
        "env": env,
    }


async def load_mcp_tools(settings: Settings) -> list[BaseTool]:
    """Connect to both MCP servers and return their combined tool list.

    Raises:
        FileNotFoundError: If either server's interpreter/binary is missing —
            the common cause is skipping a build/setup step from the README
            Quick Start (the error message says which one).
    """
    client = MultiServerMCPClient(
        {
            "sandbox": _sandbox_connection(settings),
            "mathkb": _mathkb_connection(settings),
        }
    )
    return await client.get_tools()
