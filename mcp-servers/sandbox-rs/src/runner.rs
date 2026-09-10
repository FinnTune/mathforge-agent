//! Runs one script in an isolated Python subprocess and returns a single
//! string for the model — the actual trust boundary of this server.
//!
//! Compared to the previous `subprocess.run(..., timeout=...)` implementation:
//! - Resource limits ([`ResourceLimits`]) are applied in the child before
//!   `exec`, not just a wall-clock timeout.
//! - On timeout, the whole process group is killed (not just the direct
//!   child), so a script that spawns its own children can't outlive it.
//! - Output is read with a running cap instead of buffered to EOF then
//!   truncated, so a child can't grow parent memory unboundedly by writing
//!   output faster than the cap even before it's killed.

use std::io::Write;
use std::path::Path;
use std::process::Stdio;
use std::time::Duration;

use tokio::io::AsyncReadExt;
use tokio::process::Command;
use tokio::time::timeout;

use crate::config::Config;
use crate::env::{SANDBOX_PREAMBLE, minimal_env};
use crate::limits::ResourceLimits;

pub async fn run_python_code_isolated(code: &str, config: &Config) -> String {
    let workspace = &config.workspace_root;
    let plots = workspace.join("plots");
    if let Err(e) = std::fs::create_dir_all(&plots) {
        return format!("Failed to prepare workspace: {e}");
    }

    let env = match minimal_env(workspace) {
        Ok(e) => e,
        Err(e) => return format!("Failed to prepare sandbox environment: {e}"),
    };

    let script = format!("{SANDBOX_PREAMBLE}\n{code}");
    let script_path = match write_script(workspace, &script) {
        Ok(p) => p,
        Err(e) => return format!("Failed to write sandbox script: {e}"),
    };

    let limits = ResourceLimits {
        cpu_seconds: config.timeout.as_secs().max(1),
        max_memory_bytes: config.max_memory_mb * 1024 * 1024,
        max_open_files: 256,
    };

    let python_bin = std::env::var("MATHFORGE_SANDBOX_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let mut cmd = Command::new(python_bin);
    cmd.arg(&script_path)
        .current_dir(workspace)
        .env_clear()
        .envs(&env)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .process_group(0);

    // Safety: `apply()` only calls `setrlimit`, which is async-signal-safe and
    // runs in the forked child before `execve` — it never touches the parent.
    unsafe {
        cmd.pre_exec(move || limits.apply());
    }

    let result = run_with_timeout(cmd, config.timeout, config.max_output_bytes).await;
    let _ = std::fs::remove_file(&script_path);
    result
}

fn write_script(workspace: &Path, script: &str) -> std::io::Result<std::path::PathBuf> {
    let mut tmp = tempfile::Builder::new()
        .prefix(".mathforge_sandbox_")
        .suffix(".py")
        .tempfile_in(workspace)?;
    tmp.write_all(script.as_bytes())?;
    let (_file, path) = tmp.keep().map_err(|e| e.error)?;
    Ok(path)
}

async fn run_with_timeout(mut cmd: Command, dur: Duration, max_bytes: usize) -> String {
    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => return format!("Failed to start sandbox process: {e}"),
    };
    let pid = child.id();
    let mut stdout = child.stdout.take().expect("stdout was piped");
    let mut stderr = child.stderr.take().expect("stderr was piped");

    let stdout_task = tokio::spawn(async move { read_capped(&mut stdout, max_bytes).await });
    let stderr_task = tokio::spawn(async move { read_capped(&mut stderr, max_bytes).await });

    match timeout(dur, child.wait()).await {
        Err(_) => {
            // Negative pid == kill the whole process group, so grandchildren
            // spawned by the script die too, not just the direct child.
            if let Some(pid) = pid {
                unsafe {
                    libc::kill(-(pid as libc::pid_t), libc::SIGKILL);
                }
            }
            let _ = child.wait().await;
            stdout_task.abort();
            stderr_task.abort();
            format!("Execution timed out after {:.1} seconds.", dur.as_secs_f64())
        }
        Ok(Err(e)) => format!("Execution failed: {e}"),
        Ok(Ok(status)) => {
            let (out, out_truncated) = stdout_task.await.unwrap_or_default();
            let (err, err_truncated) = stderr_task.await.unwrap_or_default();
            let mut combined = out;
            combined.push_str(&err);
            if out_truncated || err_truncated {
                combined.push_str("\n… (output truncated)");
            }

            if !status.success() && combined.trim().is_empty() {
                return format!(
                    "Process exited with code {} (no output captured).",
                    status.code().unwrap_or(-1)
                );
            }
            if !status.success() {
                return format!("(exit {})\n{combined}", status.code().unwrap_or(-1));
            }
            combined
        }
    }
}

/// Reads `reader` to EOF, keeping at most `max_bytes` in memory. Bytes past
/// the cap are still drained (so the child never blocks writing into a full
/// pipe) but discarded — the child's own rlimits/timeout are what actually
/// bound how much it can produce, this just bounds *our* memory regardless.
async fn read_capped<R: tokio::io::AsyncRead + Unpin>(reader: &mut R, max_bytes: usize) -> (String, bool) {
    let mut buf = Vec::new();
    let mut scratch = [0u8; 8192];
    let mut truncated = false;
    loop {
        match reader.read(&mut scratch).await {
            Ok(0) => break,
            Ok(n) => {
                if buf.len() < max_bytes {
                    let take = (max_bytes - buf.len()).min(n);
                    buf.extend_from_slice(&scratch[..take]);
                    if take < n {
                        truncated = true;
                    }
                } else {
                    truncated = true;
                }
            }
            Err(_) => break,
        }
    }
    (String::from_utf8_lossy(&buf).into_owned(), truncated)
}
