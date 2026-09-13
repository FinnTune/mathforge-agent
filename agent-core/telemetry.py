"""Distributed tracing setup (OpenTelemetry).

Unset ``MATHFORGE_OTLP_ENDPOINT`` (the default, e.g. every non-Docker manual
setup) means ``init_tracing`` is never called with a real endpoint and the
``opentelemetry-api``'s default global tracer stays a no-op — every
``tracer.start_as_current_span(...)`` call site in ``grpc_server.py`` works
unconditionally, with zero overhead and no OTLP traffic, when tracing isn't
configured. Same opt-in-by-presence pattern as TLS.
"""

from __future__ import annotations

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.propagate import set_global_textmap
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator


def init_tracing(service_name: str, otlp_endpoint: str | None) -> None:
    """Installs a real OTLP-exporting tracer provider, if ``otlp_endpoint`` is set."""
    if not otlp_endpoint:
        return
    set_global_textmap(TraceContextTextMapPropagator())
    provider = TracerProvider(resource=Resource.create({SERVICE_NAME: service_name}))
    exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
