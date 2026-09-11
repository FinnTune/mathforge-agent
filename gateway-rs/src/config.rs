//! Gateway configuration, read from `MATHFORGE_GATEWAY_*` env vars.

use std::env;

fn env_or(key: &str, default: &str) -> String {
    env::var(key).ok().filter(|v| !v.is_empty()).unwrap_or_else(|| default.to_string())
}

#[derive(Debug, Clone)]
pub struct Config {
    pub bind_addr: String,
    pub upstream: String,
    /// Required — the gateway refuses to start without at least one key.
    /// An auth gateway that ships "auth optional by default" would
    /// undercut the point of building it.
    pub api_keys: Vec<String>,
    pub rate_limit_per_minute: u32,
}

impl Config {
    pub fn from_env() -> Result<Self, String> {
        let host = env_or("MATHFORGE_GATEWAY_HOST", "127.0.0.1");
        let port = env_or("MATHFORGE_GATEWAY_PORT", "50052");
        let upstream = env_or("MATHFORGE_GATEWAY_UPSTREAM", "http://127.0.0.1:50051");
        let rate_limit_per_minute: u32 = env_or("MATHFORGE_GATEWAY_RATE_LIMIT_PER_MINUTE", "30")
            .parse()
            .map_err(|_| "MATHFORGE_GATEWAY_RATE_LIMIT_PER_MINUTE must be a positive integer".to_string())?;

        let api_keys: Vec<String> = env::var("MATHFORGE_GATEWAY_API_KEYS")
            .unwrap_or_default()
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .collect();
        if api_keys.is_empty() {
            return Err(
                "MATHFORGE_GATEWAY_API_KEYS is required (comma-separated) — the gateway \
                 will not start without at least one API key configured."
                    .to_string(),
            );
        }

        Ok(Self {
            bind_addr: format!("{host}:{port}"),
            upstream,
            api_keys,
            rate_limit_per_minute,
        })
    }
}
