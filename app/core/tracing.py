"""Distributed tracing (OpenTelemetry), off by default.

Enabling this requires an OTLP collector to send spans to (Jaeger, Tempo,
the vendor of your choice) — there's none bundled with this template, so
tracing stays inert until APP_OTEL_EXPORTER_OTLP_ENDPOINT is set, matching
the opt-in pattern used elsewhere (CORS origins, admin token, predict API
key all default to "off" too).
"""

from fastapi import FastAPI
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def setup_tracing(app: FastAPI) -> None:
    """Wire up OpenTelemetry tracing for ``app``, if configured.

    Parameters
    ----------
    app : FastAPI

    Notes
    -----
    No-op unless ``settings.otel_exporter_otlp_endpoint`` is set.
    """
    settings = get_settings()
    if not settings.otel_exporter_otlp_endpoint:
        return

    provider = TracerProvider(
        resource=Resource.create({"service.name": settings.otel_service_name})
    )
    exporter = OTLPSpanExporter(endpoint=settings.otel_exporter_otlp_endpoint)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    FastAPIInstrumentor.instrument_app(app)
    logger.info(f"Tracing enabled, exporting to {settings.otel_exporter_otlp_endpoint}")
