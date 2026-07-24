import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.api.routes.predict import MAX_UPLOAD_BYTES
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


def test_predict_rejects_unsupported_content_type(client: TestClient) -> None:
    response = client.post(
        "/v1/predict", files={"file": ("digit.txt", b"not an image", "text/plain")}
    )
    assert response.status_code == 415


def test_predict_rejects_unparseable_image(client: TestClient) -> None:
    response = client.post(
        "/v1/predict", files={"file": ("digit.png", b"not a real png", "image/png")}
    )
    assert response.status_code == 400


def test_predict_rejects_oversized_upload(client: TestClient) -> None:
    oversized = b"\x00" * (MAX_UPLOAD_BYTES + 1)
    response = client.post(
        "/v1/predict", files={"file": ("digit.png", oversized, "image/png")}
    )
    assert response.status_code == 413


def test_predict_returns_valid_prediction_for_real_image(client: TestClient) -> None:
    response = client.post(
        "/v1/predict", files={"file": ("digit.png", _png_bytes(), "image/png")}
    )

    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["predicted_digit"] <= 9
    assert 0.0 <= body["confidence"] <= 1.0
    assert len(body["probabilities"]) == 10
    assert sum(body["probabilities"]) == pytest.approx(1.0, abs=1e-4)
