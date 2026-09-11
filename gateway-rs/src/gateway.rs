//! The `MathForgeChat` service implementation: authenticate, rate-limit,
//! forward to the real upstream (`grpc_server.py`).

use std::num::NonZeroU32;
use std::pin::Pin;
use std::sync::Arc;

use governor::state::keyed::DefaultKeyedStateStore;
use governor::{Quota, RateLimiter};
use tonic::codegen::tokio_stream::Stream;
use tonic::transport::Channel;
use tonic::{Request, Response, Status};

use crate::pb::math_forge_chat_client::MathForgeChatClient;
use crate::pb::math_forge_chat_server::MathForgeChat;
use crate::pb::{ChatEvent, ChatRequest, HealthCheckRequest, HealthCheckResponse};

type KeyedLimiter =
    RateLimiter<String, DefaultKeyedStateStore<String>, governor::clock::DefaultClock>;

pub struct Gateway {
    upstream: MathForgeChatClient<Channel>,
    api_keys: Vec<String>,
    limiter: Arc<KeyedLimiter>,
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

    async fn chat(
        &self,
        request: Request<ChatRequest>,
    ) -> Result<Response<Self::ChatStream>, Status> {
        let key = self.authenticate(&request)?;
        self.check_rate_limit(&key)?;

        let mut upstream = self.upstream.clone();
        let response = upstream.chat(request.into_inner()).await?;
        Ok(Response::new(Box::pin(response.into_inner()) as Self::ChatStream))
    }

    async fn health_check(
        &self,
        request: Request<HealthCheckRequest>,
    ) -> Result<Response<HealthCheckResponse>, Status> {
        // Unauthenticated on purpose — a probe/load-balancer can check the
        // gateway itself is up without needing a key.
        let mut upstream = self.upstream.clone();
        let response = upstream.health_check(request.into_inner()).await?;
        Ok(Response::new(response.into_inner()))
    }
}
