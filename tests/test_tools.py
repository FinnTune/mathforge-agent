"""Tests for ``tools.execute_python_code`` (LangChain tool wrapper).

Imports the tool inside tests after env patching so ``_workspace_and_timeout``
sees the test ``tmp_path`` (import order matters for some patterns; here the
tool reads env at invoke time, so late import is defensive).
"""

from __future__ import annotations

import pytest


@pytest.fixture
def tool_workspace(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """Point sandbox execution at ``tmp_path`` with a generous default timeout."""
    monkeypatch.setenv("MATHFORGE_WORKSPACE_ROOT", str(tmp_path))
    monkeypatch.setenv("MATHFORGE_CODE_TIMEOUT_SEC", "10")


def test_tool_runs_print(tool_workspace, tmp_path) -> None:
    from tools import execute_python_code

    raw = execute_python_code.invoke({"code": 'print(40 + 2)'})
    assert "42" in raw
    assert str(tmp_path) in raw


def test_tool_respects_timeout(tool_workspace, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MATHFORGE_CODE_TIMEOUT_SEC", "0.5")
    from tools import execute_python_code

    raw = execute_python_code.invoke({"code": "while True: pass"})
    assert "timed out" in raw.lower()
