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
use tokio::net::TcpListener;
use tokio_stream::wrappers::TcpListenerStream;
use tonic::codegen::tokio_stream::Stream;
use tonic::transport::Server;
use tonic::{Request, Response, Status};

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

async fn spawn_gateway(upstream_addr: SocketAddr, api_keys: Vec<String>, rpm: u32) -> SocketAddr {
    let upstream = MathForgeChatClient::connect(format!("http://{upstream_addr}"))
        .await
        .expect("connect to fake upstream");
    let gateway = Gateway::new(upstream, api_keys, rpm);
    let router = Server::builder().add_service(MathForgeChatServer::new(gateway));
    spawn_server(router).await
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
