"""Integration-style tests for ``sandbox.run_python_code_isolated``.

Each test exercises a real subprocess (same interpreter as pytest). Covers
happy path, timeouts, stderr on failure, env isolation from the parent, Matplotlib
Agg saves, and output truncation.
"""

from __future__ import annotations

import pytest

from sandbox import MAX_CAPTURE_BYTES, run_python_code_isolated


def test_print_stdout(tmp_path) -> None:
    out = run_python_code_isolated('print("hello")', workspace=tmp_path, timeout_sec=5)
    assert "hello" in out


def test_numpy_available(tmp_path) -> None:
    out = run_python_code_isolated(
        "import numpy as np; print(np.sum([1,2,3]))",
        workspace=tmp_path,
        timeout_sec=5,
    )
    assert "6" in out.replace("\n", "")


def test_syntax_error_reports_nonzero(tmp_path) -> None:
    out = run_python_code_isolated("def broken(", workspace=tmp_path, timeout_sec=5)
    assert "exit" in out.lower() or "error" in out.lower() or "traceback" in out.lower()


def test_timeout_stops_infinite_loop(tmp_path) -> None:
    out = run_python_code_isolated("while True: pass", workspace=tmp_path, timeout_sec=0.5)
    assert "timed out" in out.lower()


def test_child_cannot_see_parent_anthropic_key(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "super-secret-key")
    code = (
        "import os\n"
        "keys = [k for k in os.environ if 'ANTHROPIC' in k.upper()]\n"
        "print('FOUND:' + ','.join(sorted(keys)))"
    )
    out = run_python_code_isolated(code, workspace=tmp_path, timeout_sec=5)
    assert "FOUND:" in out
    assert "ANTHROPIC" not in out.split("FOUND:", 1)[-1]


def test_plots_directory_created(tmp_path) -> None:
    code = "import pathlib; p = pathlib.Path('plots'); p.mkdir(exist_ok=True); print(p.exists())"
    out = run_python_code_isolated(code, workspace=tmp_path, timeout_sec=5)
    assert "True" in out
    assert (tmp_path / "plots").is_dir()


def test_matplotlib_agg_save(tmp_path) -> None:
    code = """
import matplotlib.pyplot as plt
plt.plot([1, 2], [3, 4])
plt.savefig('plots/t.png')
print('saved')
"""
    out = run_python_code_isolated(code, workspace=tmp_path, timeout_sec=15)
    assert "saved" in out
    assert (tmp_path / "plots" / "t.png").is_file()


def test_truncates_huge_output(tmp_path) -> None:
    code = f"print('x' * {MAX_CAPTURE_BYTES + 5000})"
    out = run_python_code_isolated(code, workspace=tmp_path, timeout_sec=10)
    assert "truncated" in out
