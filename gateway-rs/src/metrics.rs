//! A tiny hand-rolled Prometheus `/metrics` endpoint — no framework needed
//! for one read-only endpoint with a handful of counters (verified: a plain
//! `tokio::net::TcpListener` responder is enough; see the Phase 9 plan for
//! the throwaway-crate check this is based on). Every path on this listener
//! returns the same body — there's only one thing to scrape.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

#[derive(Default)]
pub struct Metrics {
    pub requests_total: AtomicU64,
    pub unauthenticated_total: AtomicU64,
    pub rate_limited_total: AtomicU64,
}

impl Metrics {
    fn render(&self) -> String {
        format!(
            "# HELP mathforge_gateway_requests_total Total Chat RPCs received.\n\
             # TYPE mathforge_gateway_requests_total counter\n\
             mathforge_gateway_requests_total {}\n\
             # HELP mathforge_gateway_unauthenticated_total Chat RPCs rejected for a missing/invalid API key.\n\
             # TYPE mathforge_gateway_unauthenticated_total counter\n\
             mathforge_gateway_unauthenticated_total {}\n\
             # HELP mathforge_gateway_rate_limited_total Chat RPCs rejected for exceeding the per-key rate limit.\n\
             # TYPE mathforge_gateway_rate_limited_total counter\n\
             mathforge_gateway_rate_limited_total {}\n",
            self.requests_total.load(Ordering::Relaxed),
            self.unauthenticated_total.load(Ordering::Relaxed),
            self.rate_limited_total.load(Ordering::Relaxed),
        )
    }
}

/// Serves `metrics.render()` on every connection accepted by `listener`,
/// forever. Meant to be `tokio::spawn`ed alongside the gRPC server. Takes an
/// already-bound listener (rather than binding an address itself) so tests
/// can bind an ephemeral port and learn the real address, same pattern as
/// the gRPC integration tests' `spawn_server` helper.
pub async fn serve_metrics(listener: TcpListener, metrics: Arc<Metrics>) -> std::io::Result<()> {
    tracing::info!(addr = ?listener.local_addr(), "metrics endpoint listening");
    loop {
        let (mut socket, _) = listener.accept().await?;
        let metrics = metrics.clone();
        tokio::spawn(async move {
            let mut buf = [0u8; 1024];
            // Best-effort: a scrape is a bare GET with no body, one read is
            // enough to drain the request before we respond.
            let _ = socket.read(&mut buf).await;
            let body = metrics.render();
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/plain; version=0.0.4\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            let _ = socket.write_all(response.as_bytes()).await;
        });
    }
}
