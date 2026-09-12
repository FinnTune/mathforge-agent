//! TLS wiring for the gateway's upstream connection (gateway -> `grpc_server.py`).
//! Listening-side TLS (what clients connect to) is simple enough to stay
//! inline in `main.rs` (`ServerTlsConfig`); this module exists so both
//! `main.rs` and `tests/gateway_tests.rs` can build the same kind of
//! upstream channel without duplicating the `Endpoint`/`ClientTlsConfig`
//! wiring.

use tonic::transport::{Certificate, Channel, ClientTlsConfig, Endpoint};

use crate::pb::math_forge_chat_client::MathForgeChatClient;

/// Connects to `upstream`, over TLS trusting `ca_pem` when given (verifying
/// the peer against `domain_name`, defaulting to the URI's own host),
/// plaintext otherwise — mirrors the opt-in-by-presence-of-a-CA behavior of
/// every other TLS surface in this codebase.
pub async fn connect_upstream(
    upstream: &str,
    ca_pem: Option<&[u8]>,
    domain_name: Option<&str>,
) -> Result<MathForgeChatClient<Channel>, tonic::transport::Error> {
    let mut endpoint = Endpoint::from_shared(upstream.to_string())?;
    if let Some(ca_pem) = ca_pem {
        let domain = domain_name.map(str::to_string).unwrap_or_else(|| {
            upstream
                .parse::<tonic::codegen::http::Uri>()
                .ok()
                .and_then(|uri| uri.host().map(str::to_string))
                .unwrap_or_else(|| "localhost".to_string())
        });
        endpoint = endpoint
            .tls_config(ClientTlsConfig::new().ca_certificate(Certificate::from_pem(ca_pem)).domain_name(domain))?;
    }
    let channel = endpoint.connect().await?;
    Ok(MathForgeChatClient::new(channel))
}
