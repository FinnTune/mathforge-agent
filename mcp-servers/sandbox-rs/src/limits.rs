//! OS-level resource limits applied to the sandboxed child *before* `exec`.
//!
//! This is the main hardening this Rust server adds over the previous Python
//! `subprocess.run`-based sandbox, which only had a wall-clock timeout: a
//! runaway or malicious script could still allocate unbounded memory or open
//! unbounded file descriptors. `setrlimit` here closes those gaps at the
//! kernel level, independent of anything the script does.
//!
//! **Deliberately not enforcing `RLIMIT_NPROC`** as a fork-bomb guard: on
//! Linux it caps the *real UID's* total process/thread count system-wide,
//! not the child's own subtree — on a machine where the service account is
//! already running other work (routine here: 90+ processes under a normal
//! dev/CI user), a low value fails almost every `pthread_create` immediately
//! (numpy/matplotlib spawn threads on import) and a high value stops
//! stopping anything. A correct per-subtree fork-bomb guard needs a cgroup
//! with `pids.max`, which is real future work, not a `setrlimit` call.

use rlimit::Resource;
use std::io;

#[derive(Debug, Clone, Copy)]
pub struct ResourceLimits {
    /// CPU time cap in seconds (kernel sends SIGXCPU/SIGKILL past this,
    /// independent of our own wall-clock timeout).
    pub cpu_seconds: u64,
    pub max_memory_bytes: u64,
    pub max_open_files: u64,
}

impl ResourceLimits {
    /// Applies all limits to the *current* process. Must only be called from
    /// within a `pre_exec` closure, i.e. in the forked child right before
    /// `execve`, never from the parent.
    pub fn apply(&self) -> io::Result<()> {
        rlimit::setrlimit(Resource::CPU, self.cpu_seconds, self.cpu_seconds)?;
        rlimit::setrlimit(Resource::AS, self.max_memory_bytes, self.max_memory_bytes)?;
        rlimit::setrlimit(Resource::NOFILE, self.max_open_files, self.max_open_files)?;
        Ok(())
    }
}
