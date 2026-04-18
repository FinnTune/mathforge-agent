"""Pytest fixtures shared across test modules.

``dummy_settings`` mirrors production: ``execute_python_code`` reads workspace and
timeout from ``MATHFORGE_*`` env vars, so we set those whenever we build an agent
that will invoke the real sandbox tool.
"""

from __future__ import annotations

import pytest

from config import Settings


@pytest.fixture
def dummy_settings(tmp_path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Valid ``Settings`` plus env vars aligned with ``tools._workspace_and_timeout``."""
    monkeypatch.setenv("MATHFORGE_WORKSPACE_ROOT", str(tmp_path))
    monkeypatch.setenv("MATHFORGE_CODE_TIMEOUT_SEC", "10")
    return Settings(
        anthropic_api_key="test-key-not-used",
        model="claude-sonnet-4-6",
        temperature=0.0,
        max_tokens=None,
        recursion_limit=15,
        code_timeout_sec=5.0,
        workspace_root=str(tmp_path),
        log_level="DEBUG",
    )
