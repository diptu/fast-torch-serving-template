from fastapi.testclient import TestClient

from app import __version__
from app.api.dependencies import get_inference_service
from app.main import app


def test_version_endpoint() -> None:
    with TestClient(app) as client:
        response = client.get("/version")

    assert response.status_code == 200
    assert response.json() == {"version": __version__, "git_sha": "unknown"}


def test_health_endpoint() -> None:
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert isinstance(body["model_loaded"], bool)


def test_ready_returns_503_when_model_not_loaded() -> None:
    class _NotLoaded:
        checkpoint_loaded = False

    app.dependency_overrides[get_inference_service] = _NotLoaded
    try:
        with TestClient(app) as client:
            response = client.get("/ready")
    finally:
        app.dependency_overrides.pop(get_inference_service, None)

    assert response.status_code == 503


def test_ready_returns_200_when_model_loaded() -> None:
    class _Loaded:
        checkpoint_loaded = True

    app.dependency_overrides[get_inference_service] = _Loaded
    try:
        with TestClient(app) as client:
            response = client.get("/ready")
    finally:
        app.dependency_overrides.pop(get_inference_service, None)

    assert response.status_code == 200
    assert response.json() == {"ready": True}
