"""Run untrusted Python code in a separate process with a minimal environment."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

# Pre-import common math stack so the model can use np, plt, etc. directly.
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

MAX_CAPTURE_BYTES = 256_000

def _minimal_env(workspace: Path) -> dict[str, str]:
    """Environment for the child process: no API keys, writable Matplotlib config."""
    mpl_dir = workspace / ".matplotlib"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    env: dict[str, str] = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(workspace),
        "PYTHONHASHSEED": "random",
        "PYTHONUNBUFFERED": "1",
        "PYTHONNOUSERSITE": "1",
        "MPLCONFIGDIR": str(mpl_dir),
        "TMPDIR": os.environ.get("TMPDIR", "/tmp"),
        "LANG": os.environ.get("LANG", "C.UTF-8"),
    }
    # Allow optional extra safe vars (e.g. timezone) without copying the full environ.
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
    """Execute `code` under `python` in `workspace` with timeout. Returns stdout/stderr text."""
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
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass

    out = (proc.stdout or "") + (proc.stderr or "")
    if len(out.encode("utf-8")) > MAX_CAPTURE_BYTES:
        out = out[:MAX_CAPTURE_BYTES] + "\n… (output truncated)"

    if proc.returncode != 0 and not out.strip():
        return f"Process exited with code {proc.returncode} (no output captured)."

    if proc.returncode != 0:
        return f"(exit {proc.returncode})\n{out}"

    return out
