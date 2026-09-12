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
    /// TLS identity for the gateway's own listening port. Both unset
    /// (default) means plaintext, matching today's behavior; either alone
    /// is a startup error.
    pub tls_cert_path: Option<String>,
    pub tls_key_path: Option<String>,
    /// CA certificate to verify the upstream (`grpc_server.py`) with. Unset
    /// (default) means the upstream connection stays plaintext.
    pub upstream_tls_ca_path: Option<String>,
}

fn env_opt(key: &str) -> Option<String> {
    env::var(key).ok().filter(|v| !v.is_empty())
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

        let tls_cert_path = env_opt("MATHFORGE_GATEWAY_TLS_CERT");
        let tls_key_path = env_opt("MATHFORGE_GATEWAY_TLS_KEY");
        if tls_cert_path.is_some() != tls_key_path.is_some() {
            return Err(
                "MATHFORGE_GATEWAY_TLS_CERT and MATHFORGE_GATEWAY_TLS_KEY must be set together"
                    .to_string(),
            );
        }

        let upstream_tls_ca_path = env_opt("MATHFORGE_GATEWAY_UPSTREAM_TLS_CA");
        // tonic only performs the TLS handshake when the endpoint URI's
        // scheme is literally "https" (see `Connector::call` upstream) —
        // a `tls_config()` set on an "http://" endpoint is silently never
        // used, which would look like a working plaintext connection right
        // up until it hits a TLS-only listener. Catch that mismatch here
        // rather than let it surface as a confusing connect-time failure.
        if upstream_tls_ca_path.is_some() && !upstream.starts_with("https://") {
            return Err(
                "MATHFORGE_GATEWAY_UPSTREAM_TLS_CA is set, so MATHFORGE_GATEWAY_UPSTREAM must \
                 use an https:// scheme for the TLS handshake to actually happen."
                    .to_string(),
            );
        }

        Ok(Self {
            bind_addr: format!("{host}:{port}"),
            upstream,
            api_keys,
            rate_limit_per_minute,
            tls_cert_path,
            tls_key_path,
            upstream_tls_ca_path,
        })
    }
}
