import io

from fastapi.testclient import TestClient
from PIL import Image
from starlette.requests import Request

from app.core.rate_limit import _rate_limit_key
from app.main import app


def _png_bytes() -> bytes:
    image = Image.new("L", (28, 28), color=128)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _request(headers: dict[str, str]) -> Request:
    return Request(
        {
            "type": "http",
            "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
            "client": ("1.2.3.4", 12345),
        }
    )


def test_rate_limit_key_uses_api_key_header_when_present() -> None:
    assert _rate_limit_key(_request({"x-api-key": "abc"})) == "abc"


def test_rate_limit_key_falls_back_to_remote_address() -> None:
    assert _rate_limit_key(_request({})) == "1.2.3.4"


def test_predict_rate_limit_returns_429_after_default_quota(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.services.inference_service.InferenceService.predict_image",
        lambda self, image_bytes: (7, [0.0] * 7 + [1.0] + [0.0] * 2),
    )
    # A distinct X-API-Key gives this test its own rate-limit bucket (see
    # _rate_limit_key) so it can't be pushed over/under quota by other
    # tests sharing the default "testclient" remote-address bucket.
    headers = {"X-API-Key": "rate-limit-test-bucket"}
    files = {"file": ("digit.png", _png_bytes(), "image/png")}

    with TestClient(app) as client:
        responses = [
            client.post("/v1/predict", files=files, headers=headers) for _ in range(61)
        ]

    assert [r.status_code for r in responses[:60]] == [200] * 60
    assert responses[60].status_code == 429
