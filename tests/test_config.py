"""Tests for ``config.load_settings`` and ``Settings``.

Covers required API key validation, env parsing edge cases (empty strings,
whitespace), and direct dataclass construction for non-env scenarios.
"""

from __future__ import annotations

import pytest

from config import Settings, load_settings


def test_load_settings_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        load_settings()


def test_load_settings_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("MATHFORGE_MODEL", "claude-test")
    monkeypatch.setenv("MATHFORGE_TEMPERATURE", "0.5")
    monkeypatch.setenv("MATHFORGE_RECURSION_LIMIT", "99")
    monkeypatch.setenv("MATHFORGE_CODE_TIMEOUT_SEC", "12")
    monkeypatch.setenv("MATHFORGE_WORKSPACE_ROOT", "/tmp/ws")
    monkeypatch.setenv("MATHFORGE_LOG_LEVEL", "warning")

    s = load_settings()
    assert s.anthropic_api_key == "sk-test"
    assert s.model == "claude-test"
    assert s.temperature == 0.5
    assert s.recursion_limit == 99
    assert s.code_timeout_sec == 12.0
    assert s.workspace_root == "/tmp/ws"
    assert s.log_level == "WARNING"


def test_settings_dataclass_instantiation() -> None:
    s = Settings(
        anthropic_api_key="k",
        model="m",
        temperature=0.0,
        max_tokens=100,
        recursion_limit=10,
        code_timeout_sec=1.0,
        workspace_root=".",
        log_level="INFO",
    )
    assert s.max_tokens == 100


def test_max_tokens_optional_empty_string(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setenv("MATHFORGE_MAX_TOKENS", "")
    s = load_settings()
    assert s.max_tokens is None


def test_strips_api_key_whitespace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "  sk-abc  ")
    s = load_settings()
    assert s.anthropic_api_key == "sk-abc"
