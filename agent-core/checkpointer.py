"""Builds the persistent (SQLite-backed) LangGraph checkpointer.

Replaces ``InMemorySaver`` in production: conversation memory now survives a
process restart. Centralized here because both ``main.py`` and
``discord_bot.py`` need the same connection lifecycle — unlike the MCP tool
sessions from ``mcp_client.py`` (a fresh session per call, no held-open
state), a checkpointer's DB connection must stay open for the whole process
lifetime, so callers use this as an ``async with`` around their entire
run loop.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from config import Settings


@asynccontextmanager
async def build_checkpointer(settings: Settings) -> AsyncIterator[AsyncSqliteSaver]:
    """Open (creating if needed) the SQLite checkpointer at ``settings.checkpoint_db_path``."""
    db_path = Path(settings.checkpoint_db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(str(db_path)) as checkpointer:
        await checkpointer.setup()
        yield checkpointer
