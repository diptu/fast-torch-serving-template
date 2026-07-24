import asyncio
import time

import httpx
import pytest

import app.api.routes.predict as predict_module
from app.main import app


def _slow_predict_image(sleep_seconds: float):
    def _predict(self, image_bytes: bytes) -> tuple[int, list[float]]:
        time.sleep(sleep_seconds)
        return 7, [0.0] * 7 + [1.0] + [0.0] * 2

    return _predict


async def _post_digit(client: httpx.AsyncClient) -> httpx.Response:
    return await client.post(
        "/v1/predict",
        files={"file": ("digit.png", b"\x89PNG\r\n\x1a\n" + b"\x00" * 32, "image/png")},
    )


@pytest.mark.anyio
async def test_predict_serializes_requests_at_concurrency_limit(monkeypatch) -> None:
    monkeypatch.setattr(predict_module, "_inference_semaphore", asyncio.Semaphore(1))
    monkeypatch.setattr(
        predict_module.settings, "prediction_queue_timeout_seconds", 2.0
    )
    monkeypatch.setattr(
        "app.services.inference_service.InferenceService.predict_image",
        _slow_predict_image(0.2),
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        start = time.monotonic()
        responses = await asyncio.gather(_post_digit(client), _post_digit(client))
        elapsed = time.monotonic() - start

    assert all(r.status_code == 200 for r in responses)
    # With a concurrency limit of 1 and two 0.2s calls, they must run
    # back-to-back rather than in parallel.
    assert elapsed >= 0.4


@pytest.mark.anyio
async def test_predict_returns_503_when_capacity_exceeded(monkeypatch) -> None:
    monkeypatch.setattr(predict_module, "_inference_semaphore", asyncio.Semaphore(1))
    # Budget comfortably covers one 0.15s inference but not two back-to-back
    # (one waiting on the other's slot) — so the loser times out while the
    # winner finishes fine.
    monkeypatch.setattr(
        predict_module.settings, "prediction_queue_timeout_seconds", 0.25
    )
    monkeypatch.setattr(
        "app.services.inference_service.InferenceService.predict_image",
        _slow_predict_image(0.15),
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        responses = await asyncio.gather(_post_digit(client), _post_digit(client))

    statuses = sorted(r.status_code for r in responses)
    assert statuses == [200, 503]

    busy_response = next(r for r in responses if r.status_code == 503)
    assert busy_response.headers["retry-after"] == "2"
