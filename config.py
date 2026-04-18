"""Application settings loaded from the process environment.

``python-dotenv`` (called from ``main.async_main``) populates ``os.environ`` from
``.env`` before ``load_settings()`` runs, so the same variable names work locally
and in production.

``Settings`` is immutable (``frozen=True``) so configuration stays stable for the
lifetime of a ``Settings`` instance—handy for tests and clarity.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_int(key: str, default: int) -> int:
    """Parse ``key`` as int; empty or missing → ``default``."""
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    return int(raw)


def _env_float(key: str, default: float) -> float:
    """Parse ``key`` as float; empty or missing → ``default``."""
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    return float(raw)


@dataclass(frozen=True)
class Settings:
    """Resolved configuration for the chat model, agent graph, and sandbox."""

    anthropic_api_key: str
    model: str
    temperature: float
    max_tokens: int | None
    # LangGraph ``recursion_limit`` — caps agent/tool loop depth (each step counts).
    recursion_limit: int
    # Passed to ``subprocess.run(..., timeout=...)`` in the code sandbox.
    code_timeout_sec: float
    # Working directory for executed code and relative paths like ./plots/.
    workspace_root: str
    log_level: str


def load_settings() -> Settings:
    """Read settings from the environment.

    Raises:
        ValueError: If ``ANTHROPIC_API_KEY`` is missing or whitespace-only.
    """
    key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not key:
        msg = "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key."
        raise ValueError(msg)

    max_tokens_raw = os.getenv("MATHFORGE_MAX_TOKENS", "").strip()
    max_tokens = int(max_tokens_raw) if max_tokens_raw else None

    return Settings(
        anthropic_api_key=key,
        model=os.getenv("MATHFORGE_MODEL", "claude-sonnet-4-6"),
        temperature=_env_float("MATHFORGE_TEMPERATURE", 0.0),
        max_tokens=max_tokens,
        recursion_limit=_env_int("MATHFORGE_RECURSION_LIMIT", 40),
        code_timeout_sec=_env_float("MATHFORGE_CODE_TIMEOUT_SEC", 30.0),
        workspace_root=os.getenv("MATHFORGE_WORKSPACE_ROOT", "."),
        log_level=os.getenv("MATHFORGE_LOG_LEVEL", "INFO").upper(),
    )
