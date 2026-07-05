# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A template for serving PyTorch models through FastAPI. The concrete example wired up so far is an MNIST digit classifier: model, dataset loader, training loop, inference, a full FastAPI app (routes, middleware, schemas, service layer), Docker/Compose packaging, CI/CD, and Kubernetes manifests — see "Known gaps / backlog" below for what's actually still open.

Package manager is **uv**; do not use pip/poetry directly.

## Commands

```bash
uv sync                    # install dependencies (creates .venv)
uv run uvicorn app.main:app --reload   # dev server

make lint                  # ruff check --fix + ruff format
make check                 # mypy app/ (strict) + bandit security audit (medium+ severity)
make test                  # pytest with coverage, fails if coverage < 80%; HTML report in coverage_html/

uv run pytest tests/test_mnist.py::test_mnist_inference_shape   # run a single test
uv run mypy app/<path>                                          # type-check a single path

make train                                     # runs scripts/train_dispatch.py (Colab GPU > local CUDA/MPS > CPU), which calls app/ml/train/train.py
make evaluate                                  # runs `python -m app.ml.train.evaluate --checkpoint=checkpoints/model_latest.pth`
```

`make scaffold` / `make init` (re)create the directory/file layout below and `__init__.py` files — this is how the empty placeholders came to exist. `make purge` deletes `app/`, `tests/`, `docker/`, `scripts/`, `.github/`, and top-level project files; treat it as destructive and never run it without explicit user confirmation.

CI (`.github/workflows/ci.yml`) runs `make lint`, `make check`, `make test`, then a `docker build` on push/PR to `main`.

## Architecture

```
app/
├── api/
│   ├── routes/     # predict.py (/v1/predict, /v1/predict/batch), admin.py (/admin/reload-model)
│   ├── docs/       # OpenAPI summary/description/responses text, kept out of routes/ for readability
│   ├── dependencies.py
│   └── middleware.py   # RequestIDMiddleware
├── core/           # config.py (Settings), logging.py, security.py (secrets_match), tracing.py (OTel), rate_limit.py (slowapi Limiter)
├── ml/
│   ├── models/     # nn.Module definitions (e.g. MNISTModel)
│   ├── datasets/   # DataLoader factories, transform.py (canonical MNIST_MEAN/STD + dataset_fingerprint(), shared by training and serving)
│   ├── inference/  # inference-time helpers (predict.py)
│   ├── train/      # train.py (registers each run in the MLflow Model Registry, embeds the transform in each checkpoint), engine.py (train_one_epoch/evaluate + evaluate_detailed's confusion matrix/per-class/calibration report), evaluate.py CLI, promote.py (gates on val_accuracy + per-class recall + calibration vs. the champion, then points the registry's "champion"/"shadow" alias + model_latest.pth/model_shadow.pth at a run)
│   └── utils/      # device.py (CUDA/MPS/CPU selection)
├── schemas/        # prediction.py (PredictionResponse, BatchPredictionResponse)
├── services/       # inference_service.py — the service layer between routes and app/ml; also loads an optional shadow model (model_shadow.pth) scored on every prediction, best-effort, never affecting the response
└── main.py         # FastAPI app: routes, CORS, RequestIDMiddleware, Prometheus /metrics, OTel, /health, /ready
```

Request flow: route → Pydantic validation → `InferenceService` (`app/services/`) → `app/ml` model → response. Every layer in the diagram above is implemented.

`docs/external/` is a separate thing from `app/api/docs/` above — it's third-party-facing API documentation (getting started, auth, endpoint reference, errors, changelog) meant to be shared outside the team, with no internal implementation/deployment detail. Keep it in sync with `app/api/routes/*.py` and `app/schemas/prediction.py` when the API contract changes; it is not auto-generated from the OpenAPI schema.

Configuration is centralized in `app/core/config.py` via a pydantic-settings `Settings` class (env prefix `APP_`, loads from `.env`), exposed through the cached `get_settings()` accessor — don't instantiate `Settings()` directly, use `get_settings()` so config stays a singleton. Field naming is intentionally mixed (`snake_case` like `device`/`num_classes`/`data_dir` for general settings, `UPPER_CASE` like `BATCH_SIZE`/`LEARNING_RATE`/`EPOCHS`/`SEED` for ML hyperparameters) — check the actual attribute name in `config.py` before using it, don't assume a single casing convention applies everywhere.

Structured logging goes through `app/core/logging.py`: call `setup_logging()` once at process start, then `get_logger(__name__)` everywhere else. The formatter emits JSON lines suitable for log aggregation.

### Remote GPU training bridge (`scripts/`)

`scripts/colab_server.py`, `scripts/run_remote.py`, and `scripts/gpu_sanity_check.py` form a system for running training scripts on a free Colab GPU from a local machine, not just standalone utilities:
1. `colab_server.py` runs inside a Colab notebook cell — starts Jupyter, opens a tunnel (cloudflared/ngrok/localhostrun), and publishes the URL+token to an ntfy.sh topic.
2. `run_remote.py`, run locally, polls that same ntfy.sh topic for connection info, then executes a target local `.py` file on the remote Colab kernel over the Jupyter websocket protocol, streaming output back to the terminal.
3. `gpu_sanity_check.py` is an example self-contained target script for step 2 (verifies CUDA is actually available on the remote kernel).

Scripts run via `run_remote.py` execute as raw source with no access to the `app` package — only genuinely self-contained scripts work here. The two scripts must agree on `NTFY_TOPIC`, set via the `NTFY_TOPIC` environment variable on both ends (a Colab secret of the same name works for `colab_server.py`) — there's no committed default, since ntfy.sh topics are unauthenticated and a shared constant would be a standing secret leak.

## Known gaps / backlog

There are no known "doesn't run" seams in the codebase right now — `make lint`, `make check`, and `make test` all pass, and `main.py`/`Dockerfile`/`docker-compose.yml` are fully implemented. Remaining work is tracked as a prioritized backlog in [`TODO.md`](TODO.md) (full JWT/OAuth to replace the opt-in shared-secret tokens, GPU-accelerated serving, Redis/Celery for async batch inference) — check there before assuming something is missing, and check the actual file before trusting either TODO.md or this file, since both can drift out of date as the code changes.
