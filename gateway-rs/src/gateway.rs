//! The `MathForgeChat` service implementation: authenticate, rate-limit,
//! forward to the real upstream (`grpc_server.py`).

use std::num::NonZeroU32;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use governor::state::keyed::DefaultKeyedStateStore;
use governor::{Quota, RateLimiter};
use opentelemetry::global;
use tonic::codegen::tokio_stream::Stream;
use tonic::transport::Channel;
use tonic::{Request, Response, Status};
use tracing_opentelemetry::OpenTelemetrySpanExt;

use crate::metrics::Metrics;
use crate::pb::math_forge_chat_client::MathForgeChatClient;
use crate::pb::math_forge_chat_server::MathForgeChat;
use crate::pb::{ChatEvent, ChatRequest, HealthCheckRequest, HealthCheckResponse};

type KeyedLimiter =
    RateLimiter<String, DefaultKeyedStateStore<String>, governor::clock::DefaultClock>;

pub struct Gateway {
    upstream: MathForgeChatClient<Channel>,
    api_keys: Vec<String>,
    limiter: Arc<KeyedLimiter>,
    pub metrics: Arc<Metrics>,
}

/// Stamps the outgoing request's gRPC metadata with the *current* tracing
/// span's OTel context as a W3C `traceparent` (+`tracestate` if non-empty),
/// so `grpc_server.py` on the other end can join the same trace as a child
/// span instead of starting a disconnected one. A no-op when tracing isn't
/// configured (`telemetry::init_tracing` was never given an OTLP endpoint) —
/// there's no active span to inject in that case, and the propagator
/// defaults to a no-op too.
fn inject_trace_context<T>(request: &mut Request<T>) {
    let cx = tracing::Span::current().context();
    let mut carrier = std::collections::HashMap::new();
    global::get_text_map_propagator(|propagator| propagator.inject_context(&cx, &mut carrier));
    for (key, value) in carrier {
        if let (Ok(key), Ok(value)) = (
            key.parse::<tonic::metadata::MetadataKey<tonic::metadata::Ascii>>(),
            value.parse::<tonic::metadata::MetadataValue<tonic::metadata::Ascii>>(),
        ) {
            request.metadata_mut().insert(key, value);
        }
    }
}

impl Gateway {
    pub fn new(
        upstream: MathForgeChatClient<Channel>,
        api_keys: Vec<String>,
        rate_limit_per_minute: u32,
    ) -> Self {
        let quota = Quota::per_minute(NonZeroU32::new(rate_limit_per_minute.max(1)).unwrap());
        Self {
            upstream,
            api_keys,
            limiter: Arc::new(RateLimiter::keyed(quota)),
            metrics: Arc::new(Metrics::default()),
        }
    }

    /// Returns the caller's API key if valid, else an `Unauthenticated` status.
    fn authenticate<T>(&self, request: &Request<T>) -> Result<String, Status> {
        let key = request
            .metadata()
            .get("x-api-key")
            .ok_or_else(|| Status::unauthenticated("missing x-api-key metadata"))?
            .to_str()
            .map_err(|_| Status::unauthenticated("x-api-key metadata is not valid UTF-8"))?
            .to_string();
        if !self.api_keys.iter().any(|k| k == &key) {
            return Err(Status::unauthenticated("invalid x-api-key"));
        }
        Ok(key)
    }

    /// Checked *after* `authenticate` so unauthenticated traffic can't burn
    /// a legitimate key's budget.
    fn check_rate_limit(&self, key: &str) -> Result<(), Status> {
        self.limiter
            .check_key(&key.to_owned())
            .map_err(|_| Status::resource_exhausted("rate limit exceeded"))
    }
}

#[tonic::async_trait]
impl MathForgeChat for Gateway {
    type ChatStream =
        Pin<Box<dyn Stream<Item = Result<ChatEvent, Status>> + Send + 'static>>;

    #[tracing::instrument(name = "gateway.chat", skip(self, request), fields(thread_id = %request.get_ref().thread_id))]
    async fn chat(
        &self,
        request: Request<ChatRequest>,
    ) -> Result<Response<Self::ChatStream>, Status> {
        self.metrics.requests_total.fetch_add(1, Ordering::Relaxed);

        let key = match self.authenticate(&request) {
            Ok(key) => key,
            Err(status) => {
                self.metrics.unauthenticated_total.fetch_add(1, Ordering::Relaxed);
                return Err(status);
            }
        };
        if let Err(status) = self.check_rate_limit(&key) {
            self.metrics.rate_limited_total.fetch_add(1, Ordering::Relaxed);
            return Err(status);
        }

        let mut upstream = self.upstream.clone();
        let mut outgoing = Request::new(request.into_inner());
        inject_trace_context(&mut outgoing);
        let response = upstream.chat(outgoing).await?;
        Ok(Response::new(Box::pin(response.into_inner()) as Self::ChatStream))
    }

    #[tracing::instrument(name = "gateway.health_check", skip(self, request))]
    async fn health_check(
        &self,
        request: Request<HealthCheckRequest>,
    ) -> Result<Response<HealthCheckResponse>, Status> {
        // Unauthenticated on purpose — a probe/load-balancer can check the
        // gateway itself is up without needing a key.
        let mut upstream = self.upstream.clone();
        let mut outgoing = Request::new(request.into_inner());
        inject_trace_context(&mut outgoing);
        let response = upstream.health_check(outgoing).await?;
        Ok(Response::new(response.into_inner()))
    }
}
