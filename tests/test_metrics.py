from fastapi.testclient import TestClient

import app.main as main_module
from app.main import app


def test_metrics_endpoint_exposes_prometheus_format() -> None:
    with TestClient(app) as client:
        client.get("/health")
        response = client.get("/metrics")

    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    assert "http_requests_total" in response.text


def test_metrics_rejects_missing_token_when_configured(monkeypatch) -> None:
    monkeypatch.setattr(main_module.settings, "metrics_token", "secret")

    with TestClient(app) as client:
        response = client.get("/metrics")

    assert response.status_code == 401


def test_metrics_accepts_correct_token(monkeypatch) -> None:
    monkeypatch.setattr(main_module.settings, "metrics_token", "secret")

    with TestClient(app) as client:
        response = client.get("/metrics", headers={"X-Metrics-Token": "secret"})

    assert response.status_code == 200
