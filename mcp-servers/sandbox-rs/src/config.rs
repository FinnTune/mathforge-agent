//! Server configuration, read from `MATHFORGE_*` env vars at startup.
//!
//! Mirrors the env var names the Python side already uses
//! (`MATHFORGE_WORKSPACE_ROOT`, `MATHFORGE_CODE_TIMEOUT_SEC`) plus two new
//! ones for the rlimit-based hardening this Rust server adds.

use std::path::PathBuf;
use std::time::Duration;

#[derive(Debug, Clone)]
pub struct Config {
    pub workspace_root: PathBuf,
    pub timeout: Duration,
    pub max_memory_mb: u64,
    pub max_output_bytes: usize,
}

fn env_or<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key)
        .ok()
        .filter(|v| !v.is_empty())
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

impl Config {
    pub fn from_env() -> Self {
        let workspace_root: String = env_or("MATHFORGE_WORKSPACE_ROOT", ".".to_string());
        let timeout_sec: f64 = env_or("MATHFORGE_CODE_TIMEOUT_SEC", 30.0);
        let max_memory_mb: u64 = env_or("MATHFORGE_SANDBOX_MAX_MEMORY_MB", 512);
        let max_output_bytes: usize = env_or("MATHFORGE_SANDBOX_MAX_OUTPUT_BYTES", 256_000);

        Self {
            workspace_root: PathBuf::from(workspace_root)
                .canonicalize()
                .unwrap_or_else(|_| PathBuf::from(".")),
            timeout: Duration::from_secs_f64(timeout_sec.max(0.1)),
            max_memory_mb,
            max_output_bytes,
        }
    }
}
