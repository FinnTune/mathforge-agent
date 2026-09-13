//! Tracing setup: always the plain `fmt` layer (stderr), plus — only when
//! `otlp_endpoint` is given — an OpenTelemetry layer exporting spans via
//! OTLP/gRPC (e.g. to Jaeger) and a W3C trace-context propagator, so
//! `tracing::info_span!`s created in `gateway.rs` become real distributed
//! trace spans a caller (or `agent-core`, downstream) can join onto.
//!
//! Unset `otlp_endpoint` (the default) means tracing behaves exactly as
//! before this module existed: local `fmt` logging only, no OTLP traffic.

use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::propagation::TraceContextPropagator;
use opentelemetry_sdk::trace::SdkTracerProvider;
use tracing_subscriber::Layer;
use tracing_subscriber::filter::Targets;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

/// Installs the global `tracing` subscriber. Returns the `SdkTracerProvider`
/// when OTLP export was enabled, so `main.rs` can flush/shut it down on exit
/// (batched spans would otherwise be lost on a clean process exit).
///
/// `RUST_LOG` (`EnvFilter::from_default_env()`) is applied only to the `fmt`
/// (terminal) layer, not the OTel layer — otherwise, with `RUST_LOG` unset
/// (`EnvFilter`'s restrictive default), spans would silently never reach the
/// exporter either, since a bare `.with(env_filter)` on the registry filters
/// *every* layer beneath it, not just `fmt`. Trace export shouldn't depend
/// on terminal verbosity.
pub fn init_tracing(otlp_endpoint: Option<&str>) -> Option<SdkTracerProvider> {
    let fmt_layer =
        tracing_subscriber::fmt::layer().with_filter(tracing_subscriber::EnvFilter::from_default_env());

    let Some(endpoint) = otlp_endpoint else {
        tracing_subscriber::registry().with(fmt_layer).init();
        return None;
    };

    opentelemetry::global::set_text_map_propagator(TraceContextPropagator::new());
    let exporter = opentelemetry_otlp::SpanExporter::builder()
        .with_tonic()
        .with_endpoint(endpoint)
        .build()
        .expect("build OTLP span exporter");
    let resource = opentelemetry_sdk::Resource::builder().with_service_name("mathforge-gateway").build();
    let provider =
        SdkTracerProvider::builder().with_resource(resource).with_batch_exporter(exporter).build();
    let tracer = {
        use opentelemetry::trace::TracerProvider as _;
        provider.tracer("mathforge-gateway")
    };
    // `tonic`/`hyper`/`h2` all emit their own (very chatty, transport-level)
    // `tracing` spans — without this, the OTel layer exports those too,
    // burying our own `gateway.chat`/`gateway.health_check` spans under
    // hundreds of irrelevant `h2::proto::...` spans per request. Scope
    // export to this crate's own spans only.
    let otel_layer = tracing_opentelemetry::layer()
        .with_tracer(tracer)
        .with_filter(Targets::new().with_target(env!("CARGO_PKG_NAME").replace('-', "_"), tracing::Level::TRACE));

    tracing_subscriber::registry().with(fmt_layer).with(otel_layer).init();
    Some(provider)
}
