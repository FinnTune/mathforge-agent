//! `mathforge-gateway` — a `tonic` gRPC gateway in front of `agent-core`'s
//! `grpc_server.py`. Implements the *same* `MathForgeChat` service and
//! transparently forwards to the real server, but requires an API key
//! (`x-api-key` metadata) and enforces a per-key rate limit first — the
//! systems-level concerns (auth, rate limiting) live here in Rust,
//! orchestration stays in Python, same split as the sandbox server.
//!
//! Split from `main.rs` so `tests/` can build a `Gateway` directly against a
//! fake in-process upstream, without going through the real binary.

pub mod config;
pub mod gateway;
pub mod tls;

pub mod pb {
    tonic::include_proto!("mathforge");
}
