"""Isolated execution of model-generated Python in a subprocess.

This module is the trust boundary for *code* (as opposed to the LLM, which sees
your API key in the parent process). Design goals:

- **Process isolation:** User/model code never runs in the same interpreter as
  the agent. A crash or infinite loop in code does not take down the CLI (loops
  are bounded by ``timeout_sec``).
- **Minimal environment:** The child does not inherit ``ANTHROPIC_API_KEY`` or
  other secrets from the parent. Only a small allowlist of variables is passed
  (see ``_minimal_env``). This reduces accidental exfiltration via ``os.environ``.
- **Workspace discipline:** ``cwd`` is the configured workspace; plots and temp
  Matplotlib config live under that tree.

This is **not** a hard security sandbox (no seccomp, no VM). Malicious code still
runs as your OS user inside ``workspace``. See README for stronger options.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

# Code prepended to every run so the model can assume common aliases exist.
# - ``matplotlib.use("Agg")``: non-interactive backend suitable for servers/CI.
# - ``plots/`` is created early so savefig paths like ./plots/x.png work.
# - ``_os`` avoids shadowing if user code assigns ``os``.
SANDBOX_PREAMBLE = """
import os as _os
_os.makedirs("plots", exist_ok=True)

import sympy as sp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy
"""

# Cap captured stdout/stderr so a chatty script cannot allocate unbounded memory
# in the parent when reading ``proc.stdout``.
MAX_CAPTURE_BYTES = 256_000


def _minimal_env(workspace: Path) -> dict[str, str]:
    """Build ``env`` for ``subprocess.run``.

    We intentionally avoid copying ``os.environ`` wholesale: that would leak
    API keys and other credentials into the child. ``HOME`` points at the
    workspace so libraries that write under $HOME stay inside the project tree.
    ``MPLCONFIGDIR`` must be writable for Matplotlib font/cache behavior.
    """
    mpl_dir = workspace / ".matplotlib"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    env: dict[str, str] = {
        # Needed to find the same Python toolchain / system binaries as the parent.
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(workspace),
        "PYTHONHASHSEED": "random",
        "PYTHONUNBUFFERED": "1",
        # Prevent user site-packages from altering the interpreter profile.
        "PYTHONNOUSERSITE": "1",
        "MPLCONFIGDIR": str(mpl_dir),
        "TMPDIR": os.environ.get("TMPDIR", "/tmp"),
        "LANG": os.environ.get("LANG", "C.UTF-8"),
    }
    # Optional passthroughs for reproducible numeric behavior (e.g. timezone).
    for key in ("TZ",):
        if key in os.environ:
            env[key] = os.environ[key]
    return env


def run_python_code_isolated(
    code: str,
    *,
    workspace: Path,
    timeout_sec: float,
) -> str:
    """Run ``code`` in a new Python process and return a single string for the LLM.

    The script is written to a file under ``workspace`` (not ``python -c``) so
    we avoid shell length limits and escaping issues for large multiline snippets.

    Returns:
        Combined stdout/stderr, possibly prefixed with exit code on failure, or
        a timeout message if ``timeout_sec`` elapses.
    """
    workspace = workspace.resolve()
    plots = workspace / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    script = SANDBOX_PREAMBLE + "\n" + code
    env = _minimal_env(workspace)

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix="_mathforge_sandbox.py",
        dir=workspace,
        delete=False,
        encoding="utf-8",
    ) as tmp:
        tmp.write(script)
        path = Path(tmp.name)

    proc: subprocess.CompletedProcess[str] | None = None
    try:
        proc = subprocess.run(
            [sys.executable, str(path)],
            cwd=str(workspace),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired:
        return f"Execution timed out after {timeout_sec} seconds."
    finally:
        # Always remove the script; user code should not rely on it existing.
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass

    if proc is None:
        return "Execution finished without a process result."

    out = (proc.stdout or "") + (proc.stderr or "")
    if len(out.encode("utf-8")) > MAX_CAPTURE_BYTES:
        out = out[:MAX_CAPTURE_BYTES] + "\n… (output truncated)"

    if proc.returncode != 0 and not out.strip():
        return f"Process exited with code {proc.returncode} (no output captured)."

    if proc.returncode != 0:
        return f"(exit {proc.returncode})\n{out}"

    return out
