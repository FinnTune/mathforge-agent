"""Loads the sandbox tool from the Rust MCP server as a LangChain tool.

Replaces the old in-process ``tools.execute_python_code``. The MCP server
(``mcp-servers/sandbox-rs``) is spawned over stdio by ``langchain-mcp-adapters``
— per its own docs, a fresh session (and subprocess) is opened for *each* tool
call and torn down afterward, so callers just need the tool list from
``load_sandbox_tools()`` once; there is no client lifecycle to manage across
the app's lifetime.
"""

from __future__ import annotations

import os

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from config import Settings


def _sandbox_env(settings: Settings) -> dict[str, str]:
    """Env passed to the sandbox server subprocess (stdio ``env`` replaces, not merges)."""
    return {
        "PATH": os.environ.get("PATH", ""),
        "MATHFORGE_WORKSPACE_ROOT": settings.workspace_root,
        "MATHFORGE_CODE_TIMEOUT_SEC": str(settings.code_timeout_sec),
        "MATHFORGE_SANDBOX_PYTHON": settings.sandbox_python,
        "MATHFORGE_SANDBOX_MAX_MEMORY_MB": str(settings.sandbox_max_memory_mb),
        "MATHFORGE_SANDBOX_MAX_OUTPUT_BYTES": str(settings.sandbox_max_output_bytes),
    }


async def load_sandbox_tools(settings: Settings) -> list[BaseTool]:
    """Connect to the compiled sandbox MCP server and return its tools.

    Raises:
        FileNotFoundError: If ``settings.sandbox_mcp_bin`` doesn't exist — the
            most common cause is forgetting to ``cargo build --release`` the
            server first (see README Quick Start).
    """
    if not os.path.isfile(settings.sandbox_mcp_bin):
        msg = (
            f"Sandbox MCP server binary not found at {settings.sandbox_mcp_bin!r}. "
            "Build it first: cargo build --release "
            "--manifest-path mcp-servers/sandbox-rs/Cargo.toml"
        )
        raise FileNotFoundError(msg)

    client = MultiServerMCPClient(
        {
            "sandbox": {
                "transport": "stdio",
                "command": settings.sandbox_mcp_bin,
                "args": [],
                "env": _sandbox_env(settings),
            }
        }
    )
    return await client.get_tools()
