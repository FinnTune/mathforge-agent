"""LangChain tools exposed to the ReAct agent.

``execute_python_code`` is the only tool today. It reads ``MATHFORGE_*`` env vars
for workspace and timeout so that:

1. ``main.py`` can push values from ``Settings`` into the environment after
   ``load_settings()``, keeping a single source of truth.
2. Tests can ``monkeypatch`` those env vars without importing the full app stack.

The docstring on the tool is shown to the model as part of the tool schema—keep
it accurate (pre-imported names, plot paths).
"""

from __future__ import annotations

import os
from pathlib import Path

from langchain_core.tools import tool

from sandbox import run_python_code_isolated


def _workspace_and_timeout() -> tuple[Path, float]:
    """Resolve execution root and timeout from the environment (see README / .env.example)."""
    root = Path(os.getenv("MATHFORGE_WORKSPACE_ROOT", ".")).resolve()
    timeout = float(os.getenv("MATHFORGE_CODE_TIMEOUT_SEC", "30"))
    return root, timeout


@tool
def execute_python_code(code: str) -> str:
    """Execute Python in an isolated subprocess (fresh interpreter, no API keys in env).

    Pre-imported names: `sp` (SymPy), `np`, `plt`, `scipy`, `matplotlib` (Agg backend).
    For plots save to './plots/figure.png' (or similar) under the workspace.

    Returns combined stdout/stderr from the child process."""
    workspace, timeout_sec = _workspace_and_timeout()
    result = run_python_code_isolated(code, workspace=workspace, timeout_sec=timeout_sec)
    return (
        f"Execution result:\n{result}\n"
        f"(Workspace: {workspace}; plots directory: {workspace / 'plots'})"
    )
