//! Integration tests for the gateway: a fake `MathForgeChat` upstream server
//! (pure Rust, no Python) + the real [`Gateway`] + a real client, all over
//! real sockets — the same "no mocking of the wire format" discipline used
//! for the Rust sandbox server and the Python gRPC server's own tests.

use std::net::SocketAddr;
use std::pin::Pin;

use mathforge_gateway::gateway::Gateway;
use mathforge_gateway::pb::math_forge_chat_client::MathForgeChatClient;
use mathforge_gateway::pb::math_forge_chat_server::{MathForgeChat, MathForgeChatServer};
use mathforge_gateway::pb::{
    ChatEvent, ChatRequest, Done, HealthCheckRequest, HealthCheckResponse, TextDelta,
};
use mathforge_gateway::tls::connect_upstream;
use tokio::net::TcpListener;
use tokio_stream::wrappers::TcpListenerStream;
use tonic::codegen::tokio_stream::Stream;
use tonic::transport::{Certificate, ClientTlsConfig, Endpoint, Identity, Server, ServerTlsConfig};
use tonic::{Request, Response, Status};

/// A throwaway self-signed cert/key pair covering `localhost`/`127.0.0.1`,
/// generated fresh per test via `rcgen` — no secret material is ever
/// checked into the repo for these tests.
struct DevCert {
    cert_pem: String,
    key_pem: String,
}

fn generate_dev_cert() -> DevCert {
    let cert = rcgen::generate_simple_self_signed(vec!["localhost".to_string(), "127.0.0.1".to_string()])
        .expect("generate self-signed cert");
    DevCert { cert_pem: cert.cert.pem(), key_pem: cert.signing_key.serialize_pem() }
}

/// Canned upstream: always replies with one TextDelta ("upstream reply") + Done.
struct FakeUpstream;

#[tonic::async_trait]
impl MathForgeChat for FakeUpstream {
    type ChatStream = Pin<Box<dyn Stream<Item = Result<ChatEvent, Status>> + Send + 'static>>;

    async fn chat(
        &self,
        _request: Request<ChatRequest>,
    ) -> Result<Response<Self::ChatStream>, Status> {
        let events = vec![
            Ok(ChatEvent {
                event: Some(mathforge_gateway::pb::chat_event::Event::TextDelta(TextDelta {
                    text: "upstream reply".to_string(),
                })),
            }),
            Ok(ChatEvent {
                event: Some(mathforge_gateway::pb::chat_event::Event::Done(Done {})),
            }),
        ];
        let stream = tokio_stream::iter(events);
        Ok(Response::new(Box::pin(stream) as Self::ChatStream))
    }

    async fn health_check(
        &self,
        _request: Request<HealthCheckRequest>,
    ) -> Result<Response<HealthCheckResponse>, Status> {
        Ok(Response::new(HealthCheckResponse {
            ok: true,
            model: "fake-upstream-model".to_string(),
        }))
    }
}

/// Binds an ephemeral port, serves `router` on it in a background task, and
/// returns the actual bound address.
async fn spawn_server(router: tonic::transport::server::Router) -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind ephemeral port");
    let addr = listener.local_addr().expect("local_addr");
    tokio::spawn(async move {
        router
            .serve_with_incoming(TcpListenerStream::new(listener))
            .await
            .expect("server crashed");
    });
    addr
}

async fn spawn_fake_upstream() -> SocketAddr {
    let router = Server::builder().add_service(MathForgeChatServer::new(FakeUpstream));
    spawn_server(router).await
}

/// Same as `spawn_fake_upstream`, but serving TLS with `cert`.
async fn spawn_fake_upstream_tls(cert: &DevCert) -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind ephemeral port");
    let addr = listener.local_addr().expect("local_addr");
    let identity = Identity::from_pem(cert.cert_pem.clone(), cert.key_pem.clone());
    let router = Server::builder()
        .tls_config(ServerTlsConfig::new().identity(identity))
        .expect("valid tls config")
        .add_service(MathForgeChatServer::new(FakeUpstream));
    tokio::spawn(async move {
        router
            .serve_with_incoming(TcpListenerStream::new(listener))
            .await
            .expect("server crashed");
    });
    addr
}

async fn spawn_gateway(upstream_addr: SocketAddr, api_keys: Vec<String>, rpm: u32) -> SocketAddr {
    let upstream = MathForgeChatClient::connect(format!("http://{upstream_addr}"))
        .await
        .expect("connect to fake upstream");
    let gateway = Gateway::new(upstream, api_keys, rpm);
    let router = Server::builder().add_service(MathForgeChatServer::new(gateway));
    spawn_server(router).await
}

/// A gateway whose *upstream* connection is TLS (the fake upstream from
/// `spawn_fake_upstream_tls`), listening itself in plaintext.
async fn spawn_gateway_with_tls_upstream(
    upstream_addr: SocketAddr,
    upstream_ca_pem: &[u8],
    api_keys: Vec<String>,
    rpm: u32,
) -> SocketAddr {
    let upstream = connect_upstream(&format!("https://{upstream_addr}"), Some(upstream_ca_pem), Some("localhost"))
        .await
        .expect("connect to TLS fake upstream");
    let gateway = Gateway::new(upstream, api_keys, rpm);
    let router = Server::builder().add_service(MathForgeChatServer::new(gateway));
    spawn_server(router).await
}

/// A gateway that itself listens over TLS with `cert`, forwarding to a
/// plaintext fake upstream.
async fn spawn_gateway_with_tls_listener(
    upstream_addr: SocketAddr,
    cert: &DevCert,
    api_keys: Vec<String>,
    rpm: u32,
) -> SocketAddr {
    let upstream = MathForgeChatClient::connect(format!("http://{upstream_addr}"))
        .await
        .expect("connect to fake upstream");
    let gateway = Gateway::new(upstream, api_keys, rpm);
    let identity = Identity::from_pem(cert.cert_pem.clone(), cert.key_pem.clone());
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind ephemeral port");
    let addr = listener.local_addr().expect("local_addr");
    let router = Server::builder()
        .tls_config(ServerTlsConfig::new().identity(identity))
        .expect("valid tls config")
        .add_service(MathForgeChatServer::new(gateway));
    tokio::spawn(async move {
        router
            .serve_with_incoming(TcpListenerStream::new(listener))
            .await
            .expect("server crashed");
    });
    addr
}

fn request_with_key(query: &str, key: Option<&str>) -> Request<ChatRequest> {
    let mut request = Request::new(ChatRequest {
        thread_id: "t1".to_string(),
        query: query.to_string(),
    });
    if let Some(key) = key {
        request.metadata_mut().insert("x-api-key", key.parse().unwrap());
    }
    request
}

#[tokio::test]
async fn missing_api_key_is_rejected_without_reaching_upstream() {
    let upstream_addr = spawn_fake_upstream().await;
    let gateway_addr = spawn_gateway(upstream_addr, vec!["good-key".to_string()], 60).await;
    let mut client = MathForgeChatClient::connect(format!("http://{gateway_addr}"))
        .await
        .unwrap();

    let result = client.chat(request_with_key("hi", None)).await;

    let status = result.unwrap_err();
    assert_eq!(status.code(), tonic::Code::Unauthenticated);
}

#[tokio::test]
async fn invalid_api_key_is_rejected() {
    let upstream_addr = spawn_fake_upstream().await;
    let gateway_addr = spawn_gateway(upstream_addr, vec!["good-key".to_string()], 60).await;
    let mut client = MathForgeChatClient::connect(format!("http://{gateway_addr}"))
        .await
        .unwrap();

    let result = client.chat(request_with_key("hi", Some("wrong-key"))).await;

    assert_eq!(result.unwrap_err().code(), tonic::Code::Unauthenticated);
}

#[tokio::test]
async fn valid_api_key_forwards_the_real_upstream_response() {
    let upstream_addr = spawn_fake_upstream().await;
    let gateway_addr = spawn_gateway(upstream_addr, vec!["good-key".to_string()], 60).await;
    let mut client = MathForgeChatClient::connect(format!("http://{gateway_addr}"))
        .await
        .unwrap();

    let mut stream = client
        .chat(request_with_key("hi", Some("good-key")))
        .await
        .unwrap()
        .into_inner();

    let mut texts = Vec::new();
    while let Some(event) = tonic::codegen::tokio_stream::StreamExt::next(&mut stream).await {
        let event = event.unwrap();
        if let Some(mathforge_gateway::pb::chat_event::Event::TextDelta(t)) = event.event {
            texts.push(t.text);
        }
    }
    assert_eq!(texts, vec!["upstream reply".to_string()]);
}

#[tokio::test]
async fn health_check_forwards_without_requiring_a_key() {
    let upstream_addr = spawn_fake_upstream().await;
    let gateway_addr = spawn_gateway(upstream_addr, vec!["good-key".to_string()], 60).await;
    let mut client = MathForgeChatClient::connect(format!("http://{gateway_addr}"))
        .await
        .unwrap();

    let response = client.health_check(Request::new(HealthCheckRequest {})).await.unwrap();

    assert!(response.get_ref().ok);
    assert_eq!(response.get_ref().model, "fake-upstream-model");
}

#[tokio::test]
async fn exceeding_the_rate_limit_returns_resource_exhausted() {
    let upstream_addr = spawn_fake_upstream().await;
    // 1 request/minute — the second call in this test must be rejected.
    let gateway_addr = spawn_gateway(upstream_addr, vec!["good-key".to_string()], 1).await;
    let mut client = MathForgeChatClient::connect(format!("http://{gateway_addr}"))
        .await
        .unwrap();

    let first = client.chat(request_with_key("hi", Some("good-key"))).await;
    assert!(first.is_ok());

    let second = client.chat(request_with_key("hi again", Some("good-key"))).await;
    assert_eq!(second.unwrap_err().code(), tonic::Code::ResourceExhausted);
}

/// Gateway's *upstream* connection encrypted with TLS (gateway ->
/// `grpc_server.py`) — the real handshake happens, not a stubbed-out check.
#[tokio::test]
async fn gateway_connects_to_upstream_over_tls() {
    let cert = generate_dev_cert();
    let upstream_addr = spawn_fake_upstream_tls(&cert).await;
    let gateway_addr =
        spawn_gateway_with_tls_upstream(upstream_addr, cert.cert_pem.as_bytes(), vec!["good-key".to_string()], 60)
            .await;
    let mut client = MathForgeChatClient::connect(format!("http://{gateway_addr}")).await.unwrap();

    let mut stream = client.chat(request_with_key("hi", Some("good-key"))).await.unwrap().into_inner();

    let mut texts = Vec::new();
    while let Some(event) = tonic::codegen::tokio_stream::StreamExt::next(&mut stream).await {
        let event = event.unwrap();
        if let Some(mathforge_gateway::pb::chat_event::Event::TextDelta(t)) = event.event {
            texts.push(t.text);
        }
    }
    assert_eq!(texts, vec!["upstream reply".to_string()]);
}

/// Gateway's *listening* side encrypted with TLS (client -> gateway) — a
/// client without the right CA is rejected by the handshake itself.
#[tokio::test]
async fn gateway_listener_accepts_tls_clients_and_rejects_untrusted_ones() {
    let cert = generate_dev_cert();
    let upstream_addr = spawn_fake_upstream().await;
    let gateway_addr =
        spawn_gateway_with_tls_listener(upstream_addr, &cert, vec!["good-key".to_string()], 60).await;

    let trusted_endpoint = Endpoint::from_shared(format!("https://{gateway_addr}"))
        .unwrap()
        .tls_config(
            ClientTlsConfig::new()
                .ca_certificate(Certificate::from_pem(cert.cert_pem.clone()))
                .domain_name("localhost"),
        )
        .unwrap();
    let mut trusted_client =
        MathForgeChatClient::new(trusted_endpoint.connect().await.expect("TLS handshake with the right CA"));
    let response = trusted_client.health_check(Request::new(HealthCheckRequest {})).await.unwrap();
    assert!(response.get_ref().ok);

    let other_cert = generate_dev_cert();
    let untrusted_endpoint = Endpoint::from_shared(format!("https://{gateway_addr}"))
        .unwrap()
        .tls_config(
            ClientTlsConfig::new()
                .ca_certificate(Certificate::from_pem(other_cert.cert_pem))
                .domain_name("localhost"),
        )
        .unwrap();
    let result = untrusted_endpoint.connect().await;
    assert!(result.is_err(), "connecting with the wrong CA should fail the TLS handshake");
}
