"""Load settings from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    return int(raw)


def _env_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    return float(raw)


@dataclass(frozen=True)
class Settings:
    anthropic_api_key: str
    model: str
    temperature: float
    max_tokens: int | None
    recursion_limit: int
    code_timeout_sec: float
    workspace_root: str
    log_level: str


def load_settings() -> Settings:
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
