//! `mathforge-gateway` binary entrypoint: connects to the upstream
//! `grpc_server.py`, builds the [`Gateway`], and serves it over gRPC until
//! interrupted (SIGINT/SIGTERM), mirroring `grpc_server.py`'s own shutdown.

use mathforge_gateway::config::Config;
use mathforge_gateway::gateway::Gateway;
use mathforge_gateway::pb::math_forge_chat_server::MathForgeChatServer;
use mathforge_gateway::tls::connect_upstream;
use mathforge_gateway::{metrics, telemetry};
use tonic::transport::{Identity, Server, ServerTlsConfig};

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c().await.expect("failed to install SIGINT handler");
    };
    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let config = match Config::from_env() {
        Ok(config) => config,
        Err(msg) => {
            eprintln!("{msg}");
            std::process::exit(1);
        }
    };

    let tracer_provider = telemetry::init_tracing(config.otlp_endpoint.as_deref());

    tracing::info!(upstream = %config.upstream, "connecting to upstream MathForge server");
    let upstream_ca_pem = config.upstream_tls_ca_path.as_deref().map(std::fs::read).transpose()?;
    let upstream = connect_upstream(&config.upstream, upstream_ca_pem.as_deref(), None).await?;
    let gateway = Gateway::new(upstream, config.api_keys.clone(), config.rate_limit_per_minute);
    let gateway_metrics = gateway.metrics.clone();

    let metrics_listener = tokio::net::TcpListener::bind(&config.metrics_addr).await?;
    tokio::spawn(async move {
        if let Err(e) = metrics::serve_metrics(metrics_listener, gateway_metrics).await {
            tracing::error!(error = %e, "metrics endpoint failed");
        }
    });

    let addr = config.bind_addr.parse()?;

    let mut server_builder = Server::builder();
    if let (Some(cert_path), Some(key_path)) = (&config.tls_cert_path, &config.tls_key_path) {
        let cert = std::fs::read(cert_path)?;
        let key = std::fs::read(key_path)?;
        server_builder = server_builder.tls_config(ServerTlsConfig::new().identity(Identity::from_pem(cert, key)))?;
        tracing::info!(%addr, rate_limit_per_minute = config.rate_limit_per_minute, "MathForge gRPC gateway listening (TLS)");
    } else {
        tracing::info!(%addr, rate_limit_per_minute = config.rate_limit_per_minute, "MathForge gRPC gateway listening (plaintext)");
    }

    server_builder
        .add_service(MathForgeChatServer::new(gateway))
        .serve_with_shutdown(addr, shutdown_signal())
        .await?;

    tracing::info!("MathForge gRPC gateway shut down");
    // Flushes any batched-but-not-yet-exported spans; a clean process exit
    // would otherwise silently drop them.
    if let Some(provider) = tracer_provider {
        let _ = provider.shutdown();
    }
    Ok(())
}
