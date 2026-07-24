import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import app.api.routes.predict as predict_module
from app.main import app


def _png_bytes() -> bytes:
    image = Image.new("L", (28, 28), color=128)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_predict_unprotected_by_default(client: TestClient) -> None:
    response = client.post(
        "/v1/predict", files={"file": ("digit.png", _png_bytes(), "image/png")}
    )
    assert response.status_code == 200


def test_predict_rejects_missing_key_when_configured(
    client: TestClient, monkeypatch
) -> None:
    monkeypatch.setattr(predict_module.settings, "predict_api_key", "secret")

    response = client.post(
        "/v1/predict", files={"file": ("digit.png", _png_bytes(), "image/png")}
    )

    assert response.status_code == 401


def test_predict_accepts_correct_key(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(predict_module.settings, "predict_api_key", "secret")

    response = client.post(
        "/v1/predict",
        files={"file": ("digit.png", _png_bytes(), "image/png")},
        headers={"X-API-Key": "secret"},
    )

    assert response.status_code == 200
