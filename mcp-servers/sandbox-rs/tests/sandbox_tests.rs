//! Integration tests for `runner::run_python_code_isolated`, the actual trust
//! boundary of this server. Ported 1:1 from the Python sandbox's test suite
//! (`agent-core/tests/test_sandbox.py`, prior to its removal) plus new cases
//! for the rlimit- and process-group-based hardening this Rust rewrite adds.
//!
//! Requires `MATHFORGE_SANDBOX_PYTHON` to point at a Python 3 interpreter with
//! numpy/sympy/matplotlib/scipy installed (see `requirements.txt`) — every
//! script goes through the same preamble that imports them unconditionally.

use std::time::Duration;

use mathforge_sandbox_mcp::config::Config;
use mathforge_sandbox_mcp::runner::run_python_code_isolated;

fn test_config(workspace: &std::path::Path) -> Config {
    Config {
        workspace_root: workspace.to_path_buf(),
        timeout: Duration::from_secs(10),
        max_memory_mb: 512,
        max_output_bytes: 256_000,
        max_processes: 64,
    }
}

#[tokio::test]
async fn print_stdout() {
    let dir = tempfile::tempdir().unwrap();
    let out = run_python_code_isolated(r#"print("hello")"#, &test_config(dir.path())).await;
    assert!(out.contains("hello"), "unexpected output: {out}");
}

#[tokio::test]
async fn numpy_available() {
    let dir = tempfile::tempdir().unwrap();
    let out = run_python_code_isolated(
        "import numpy as np; print(np.sum([1,2,3]))",
        &test_config(dir.path()),
    )
    .await;
    assert!(out.replace('\n', "").contains('6'), "unexpected output: {out}");
}

#[tokio::test]
async fn syntax_error_reports_nonzero() {
    let dir = tempfile::tempdir().unwrap();
    let out = run_python_code_isolated("def broken(", &test_config(dir.path())).await;
    let lower = out.to_lowercase();
    assert!(
        lower.contains("exit") || lower.contains("error") || lower.contains("traceback"),
        "unexpected output: {out}"
    );
}

#[tokio::test]
async fn timeout_stops_infinite_loop() {
    let dir = tempfile::tempdir().unwrap();
    let mut config = test_config(dir.path());
    config.timeout = Duration::from_millis(500);
    let out = run_python_code_isolated("while True: pass", &config).await;
    assert!(out.to_lowercase().contains("timed out"), "unexpected output: {out}");
}

#[tokio::test]
async fn child_cannot_see_parent_anthropic_key() {
    // SAFETY: test-only, single-threaded w.r.t. this env var within this process.
    unsafe {
        std::env::set_var("ANTHROPIC_API_KEY", "super-secret-key");
    }
    let dir = tempfile::tempdir().unwrap();
    let code = "import os\n\
                keys = [k for k in os.environ if 'ANTHROPIC' in k.upper()]\n\
                print('FOUND:' + ','.join(sorted(keys)))";
    let out = run_python_code_isolated(code, &test_config(dir.path())).await;
    assert!(out.contains("FOUND:"), "unexpected output: {out}");
    let after = out.split("FOUND:").nth(1).unwrap_or_default();
    assert!(!after.contains("ANTHROPIC"), "key leaked into child env: {out}");
}

#[tokio::test]
async fn plots_directory_created() {
    let dir = tempfile::tempdir().unwrap();
    let code = "import pathlib; p = pathlib.Path('plots'); p.mkdir(exist_ok=True); print(p.exists())";
    let out = run_python_code_isolated(code, &test_config(dir.path())).await;
    assert!(out.contains("True"), "unexpected output: {out}");
    assert!(dir.path().join("plots").is_dir());
}

#[tokio::test]
async fn matplotlib_agg_save() {
    let dir = tempfile::tempdir().unwrap();
    let code = "import matplotlib.pyplot as plt\n\
                plt.plot([1, 2], [3, 4])\n\
                plt.savefig('plots/t.png')\n\
                print('saved')";
    let mut config = test_config(dir.path());
    config.timeout = Duration::from_secs(15);
    let out = run_python_code_isolated(code, &config).await;
    assert!(out.contains("saved"), "unexpected output: {out}");
    assert!(dir.path().join("plots").join("t.png").is_file());
}

#[tokio::test]
async fn truncates_huge_output() {
    let dir = tempfile::tempdir().unwrap();
    let config = test_config(dir.path());
    let code = format!("print('x' * {})", config.max_output_bytes + 5000);
    let out = run_python_code_isolated(&code, &config).await;
    assert!(out.contains("truncated"), "unexpected output: {out}");
}

/// New vs. the Python sandbox: a script that tries to allocate well past the
/// memory rlimit is killed by the kernel (MemoryError / SIGKILL), not left to
/// run until it exhausts host memory.
#[tokio::test]
async fn memory_rlimit_kills_oversized_allocation() {
    let dir = tempfile::tempdir().unwrap();
    let mut config = test_config(dir.path());
    config.max_memory_mb = 64;
    config.timeout = Duration::from_secs(10);
    // Try to allocate ~1GB, far past the 64MB rlimit.
    let code = "x = bytearray(1_000_000_000)\nprint('should not get here')";
    let out = run_python_code_isolated(code, &config).await;
    assert!(
        !out.contains("should not get here"),
        "allocation past rlimit was not stopped: {out}"
    );
}

/// New vs. the Python sandbox: on timeout, the whole process group is killed,
/// so a script's own child processes don't outlive it.
#[tokio::test]
async fn timeout_kills_grandchild_processes() {
    let dir = tempfile::tempdir().unwrap();
    let mut config = test_config(dir.path());
    config.timeout = Duration::from_millis(500);
    let marker = dir.path().join("grandchild_alive.txt");
    let code = format!(
        "import subprocess, sys\n\
         subprocess.Popen([sys.executable, '-c', \
         \"import time; time.sleep(5); open('{}', 'w').close()\"])\n\
         import time; time.sleep(5)",
        marker.display()
    );
    let out = run_python_code_isolated(&code, &config).await;
    assert!(out.to_lowercase().contains("timed out"), "unexpected output: {out}");

    // Give a leaked grandchild a moment to have written the marker if it
    // survived the kill; it shouldn't have.
    tokio::time::sleep(Duration::from_secs(2)).await;
    assert!(
        !marker.exists(),
        "grandchild process survived the timeout kill"
    );
}

/// New vs. the Python sandbox and vs. `RLIMIT_NPROC` (see `limits.rs`'s doc
/// comment for why that was dropped): a real per-subtree fork-bomb guard via
/// cgroup v2 `pids.max`. Skips (doesn't fail) on hosts without a delegated
/// cgroup v2 subtree — see `cgroup.rs` module docs.
#[tokio::test]
async fn fork_bomb_is_stopped_by_cgroup_pids_max() {
    if mathforge_sandbox_mcp::cgroup::SandboxCgroup::create(4).is_none() {
        eprintln!(
            "cgroup v2 not available/delegated in this environment — \
             skipping fork_bomb_is_stopped_by_cgroup_pids_max"
        );
        return;
    }

    let dir = tempfile::tempdir().unwrap();
    let mut config = test_config(dir.path());
    config.max_processes = 10;
    config.timeout = Duration::from_secs(10);
    let code = "import multiprocessing, time\n\
                spawned = 0\n\
                try:\n\
                    for _ in range(200):\n\
                        p = multiprocessing.Process(target=time.sleep, args=(5,))\n\
                        p.start()\n\
                        spawned += 1\n\
                    print(f'should not reach here: spawned {spawned}')\n\
                except Exception as e:\n\
                    print(f'blocked as expected after {spawned} spawned: {e}')\n";

    let start = std::time::Instant::now();
    let out = run_python_code_isolated(code, &config).await;
    let elapsed = start.elapsed();

    assert!(
        !out.contains("should not reach here"),
        "fork bomb was not stopped by pids.max: {out}"
    );
    assert!(
        elapsed < Duration::from_secs(8),
        "took {elapsed:?} for a 10s timeout — looks like it ran to the wall-clock \
         timeout instead of being stopped early by the cgroup pids limit: {out}"
    );
}
