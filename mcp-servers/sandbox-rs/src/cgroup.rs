//! cgroup v2 `pids.max` fork-bomb guard — the correct per-subtree limit
//! `RLIMIT_NPROC` can't provide (see `limits.rs`'s doc comment for why that
//! was tried and dropped: it caps the real UID's total process count
//! system-wide, not the sandboxed child's subtree).
//!
//! Opportunistic, not required: creating a cgroup here depends on the
//! current process already sitting in a *delegated* cgroup v2 subtree
//! (true under a normal systemd user session — confirmed by hand against
//! `/sys/fs/cgroup` on the dev machine this was built on — but **not**
//! guaranteed in containers without delegation, non-systemd init, or
//! non-Linux). Every function here degrades to `None`/a no-op rather than
//! erroring, so a sandbox run is never *worse* than before this existed —
//! just not extra-hardened against fork bombs on hosts where cgroups aren't
//! usable.

use std::fs::{self, File, OpenOptions};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

const CGROUP_ROOT: &str = "/sys/fs/cgroup";

/// Disambiguates cgroup names for multiple invocations within one process.
/// In production this binary is spawned fresh per invocation (see module
/// docs), so `std::process::id()` alone is already unique there — but the
/// test suite calls `run_python_code_isolated` many times *concurrently*
/// within one test-binary process, where the pid alone collides.
static INVOCATION_COUNTER: AtomicU64 = AtomicU64::new(0);

pub struct SandboxCgroup {
    path: PathBuf,
}

impl SandboxCgroup {
    /// Creates a fresh child cgroup under the current process's own cgroup
    /// with `pids.max` set to `pids_max`. `None` on any failure along the
    /// way — see module docs.
    ///
    /// Names the child `mathforge-sandbox-<own pid>-<invocation counter>` —
    /// unique per invocation even when many invocations happen concurrently
    /// within one process (the test suite), not just across the fresh
    /// process per invocation this binary normally runs as (see `main.rs`).
    pub fn create(pids_max: u64) -> Option<Self> {
        let own = own_cgroup_dir().ok()?;
        // Best effort: the controller may already be enabled (harmless), or
        // enabling it may fail (e.g. no delegation) — either way, the next
        // step (creating the child dir and checking for pids.max) is the
        // real test of whether this is usable.
        let _ = fs::write(own.join("cgroup.subtree_control"), "+pids");

        let n = INVOCATION_COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = own.join(format!("mathforge-sandbox-{}-{n}", std::process::id()));
        fs::create_dir(&path).ok()?;

        // Confirmed by hand: on a host where the parent cgroup is itself
        // "threaded", a freshly created child defaults to the unusable
        // "domain invalid" type until explicitly marked "threaded" too —
        // process migration into an "invalid" cgroup fails with ENOTSUP.
        // Fix it up if needed; if that fails too, this cgroup can't be used
        // for process migration at all, so bail out rather than hand back a
        // cgroup whose `cgroup.procs` write will fail deep inside `pre_exec`.
        if let Ok(cgroup_type) = fs::read_to_string(path.join("cgroup.type"))
            && cgroup_type.trim() == "domain invalid"
            && fs::write(path.join("cgroup.type"), "threaded").is_err()
        {
            let _ = fs::remove_dir(&path);
            return None;
        }

        if fs::write(path.join("pids.max"), pids_max.to_string()).is_err() {
            let _ = fs::remove_dir(&path);
            return None;
        }
        Some(Self { path })
    }

    /// Opens `cgroup.procs` for writing. Call this in the *parent* before
    /// spawning, so the raw fd can be moved into a `pre_exec` closure —
    /// opening by path from inside `pre_exec` itself (post-fork, pre-exec)
    /// would do unnecessary extra work in that restricted context.
    pub fn procs_file(&self) -> io::Result<File> {
        OpenOptions::new().write(true).open(self.path.join("cgroup.procs"))
    }

    /// Immediately SIGKILLs every process in this cgroup. Used on timeout
    /// *in addition to* the existing process-group kill in `runner.rs` —
    /// strictly stronger, since it also catches anything that escaped the
    /// process group (e.g. via `setsid()`), which the process-group kill can't.
    pub fn kill_all(&self) {
        let _ = fs::write(self.path.join("cgroup.kill"), "1");
    }
}

impl Drop for SandboxCgroup {
    fn drop(&mut self) {
        // Only succeeds once empty (the child has exited/been reaped). A
        // failed cleanup just leaves a harmless empty directory behind,
        // bounded by how many sandbox invocations hit this race.
        let _ = fs::remove_dir(&self.path);
    }
}

/// Resolves the current process's own cgroup v2 path under `/sys/fs/cgroup`.
fn own_cgroup_dir() -> io::Result<PathBuf> {
    let contents = fs::read_to_string("/proc/self/cgroup")?;
    let line = contents
        .lines()
        .find(|l| l.starts_with("0::"))
        .ok_or_else(|| io::Error::other("no cgroup v2 (unified hierarchy) entry in /proc/self/cgroup"))?;
    let relative = line.trim_start_matches("0::").trim_start_matches('/');
    Ok(Path::new(CGROUP_ROOT).join(relative))
}

/// Formats `pid` as ASCII decimal into `buf`, returning the used slice — no
/// heap allocation, safe to call from within a `pre_exec` closure (runs in
/// the forked child, post-fork, pre-`execve`).
pub fn format_pid(mut pid: u32, buf: &mut [u8; 10]) -> &[u8] {
    if pid == 0 {
        buf[0] = b'0';
        return &buf[..1];
    }
    let mut i = buf.len();
    while pid > 0 {
        i -= 1;
        buf[i] = b'0' + (pid % 10) as u8;
        pid /= 10;
    }
    &buf[i..]
}
