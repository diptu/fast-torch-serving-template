from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_reload_disabled_by_default(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(
        "app.api.routes.admin.get_settings",
        lambda: SimpleNamespace(admin_token=""),
    )

    response = client.post("/admin/reload-model")

    assert response.status_code == 503


def test_reload_rejects_missing_or_wrong_token(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(
        "app.api.routes.admin.get_settings",
        lambda: SimpleNamespace(admin_token="secret"),
    )

    no_token = client.post("/admin/reload-model")
    wrong_token = client.post("/admin/reload-model", headers={"X-Admin-Token": "wrong"})

    assert no_token.status_code == 401
    assert wrong_token.status_code == 401


def test_reload_succeeds_with_correct_token(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(
        "app.api.routes.admin.get_settings",
        lambda: SimpleNamespace(admin_token="secret"),
    )

    response = client.post("/admin/reload-model", headers={"X-Admin-Token": "secret"})

    assert response.status_code == 200
    body = response.json()
    assert body["reloaded"] is True
    assert isinstance(body["checkpoint_loaded"], bool)
    assert isinstance(body["shadow_loaded"], bool)
