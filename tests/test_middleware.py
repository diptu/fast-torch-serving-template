import io

from fastapi.testclient import TestClient
from PIL import Image

from app.api.dependencies import get_inference_service
from app.main import app


def _png_bytes() -> bytes:
    image = Image.new("L", (28, 28), color=128)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def test_request_id_header_present_and_stable() -> None:
    with TestClient(app) as client:
        response = client.get("/health")

    assert "x-request-id" in response.headers


def test_disallowed_origin_gets_no_cors_allow_origin_header() -> None:
    with TestClient(app) as client:
        response = client.get("/health", headers={"Origin": "http://evil.example.com"})

    assert "access-control-allow-origin" not in response.headers


def test_unhandled_exception_returns_structured_500() -> None:
    class _BrokenInferenceService:
        def predict_image(self, image_bytes: bytes) -> tuple[int, list[float]]:
            raise RuntimeError("boom")

    app.dependency_overrides[get_inference_service] = _BrokenInferenceService
    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post(
                "/v1/predict", files={"file": ("digit.png", _png_bytes(), "image/png")}
            )
    finally:
        app.dependency_overrides.pop(get_inference_service, None)

    assert response.status_code == 500
    body = response.json()
    assert body["detail"] == "Internal server error"
    assert body["request_id"] == response.headers["x-request-id"]
