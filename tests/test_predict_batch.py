import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.api.routes.predict import MAX_BATCH_SIZE
from app.main import app


def _png_bytes(color: int = 128) -> bytes:
    image = Image.new("L", (28, 28), color=color)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_batch_predict_returns_one_result_per_file(client: TestClient) -> None:
    files = [
        ("files", ("a.png", _png_bytes(50), "image/png")),
        ("files", ("b.png", _png_bytes(200), "image/png")),
        ("files", ("c.png", _png_bytes(100), "image/png")),
    ]

    response = client.post("/v1/predict/batch", files=files)

    assert response.status_code == 200
    predictions = response.json()["predictions"]
    assert len(predictions) == 3
    for p in predictions:
        assert 0 <= p["predicted_digit"] <= 9
        assert len(p["probabilities"]) == 10
        assert sum(p["probabilities"]) == pytest.approx(1.0, abs=1e-4)


def test_batch_predict_rejects_empty_file_list(client: TestClient) -> None:
    # FastAPI's own validation rejects this (422) before the route body
    # runs, since `files` is required and no parts are sent.
    response = client.post("/v1/predict/batch", files=[])
    assert response.status_code == 422


def test_batch_predict_rejects_too_many_files(client: TestClient) -> None:
    files = [
        ("files", (f"{i}.png", _png_bytes(), "image/png"))
        for i in range(MAX_BATCH_SIZE + 1)
    ]

    response = client.post("/v1/predict/batch", files=files)

    assert response.status_code == 400


def test_batch_predict_rejects_bad_content_type(client: TestClient) -> None:
    files = [
        ("files", ("a.png", _png_bytes(), "image/png")),
        ("files", ("b.txt", b"not an image", "text/plain")),
    ]

    response = client.post("/v1/predict/batch", files=files)

    assert response.status_code == 415


def test_batch_predict_rejects_unparseable_image(client: TestClient) -> None:
    files = [
        ("files", ("a.png", _png_bytes(), "image/png")),
        ("files", ("b.png", b"not a real png", "image/png")),
    ]

    response = client.post("/v1/predict/batch", files=files)

    assert response.status_code == 400
