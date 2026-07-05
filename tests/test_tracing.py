from types import SimpleNamespace
from unittest.mock import MagicMock

from fastapi import FastAPI

import app.core.tracing as tracing_module
from app.core.tracing import setup_tracing


def test_setup_tracing_is_a_noop_when_endpoint_not_configured(monkeypatch) -> None:
    monkeypatch.setattr(
        tracing_module,
        "get_settings",
        lambda: SimpleNamespace(otel_exporter_otlp_endpoint="", otel_service_name="x"),
    )
    mock_instrument = MagicMock()
    monkeypatch.setattr(
        tracing_module.FastAPIInstrumentor, "instrument_app", mock_instrument
    )

    setup_tracing(FastAPI())

    mock_instrument.assert_not_called()


def test_setup_tracing_instruments_app_when_endpoint_configured(monkeypatch) -> None:
    monkeypatch.setattr(
        tracing_module,
        "get_settings",
        lambda: SimpleNamespace(
            otel_exporter_otlp_endpoint="http://localhost:4318",
            otel_service_name="test-svc",
        ),
    )
    mock_instrument = MagicMock()
    monkeypatch.setattr(
        tracing_module.FastAPIInstrumentor, "instrument_app", mock_instrument
    )

    app = FastAPI()
    setup_tracing(app)

    mock_instrument.assert_called_once_with(app)
