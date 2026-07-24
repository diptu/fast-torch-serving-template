
<div align="center">

# 🚀 Fast-torch-serving-template

*A template for building and serving a PyTorch model behind a FastAPI API, with MLflow experiment tracking, Docker packaging, and CI/CD wired up end to end.*

<p>

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.139+-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

</p>

**Build ML APIs—not boilerplate.**

</div>

---

# ✨ Features

- ⚡ FastAPI backend serving a PyTorch CNN (MNIST digit classifier, `POST /v1/predict`)
- 📊 MLflow experiment tracking + Model Registry — params, metrics, confusion matrix/per-class/calibration reports, model + checkpoint artifacts, a `"champion"` alias gated on accuracy/recall/calibration (`make promote`)
- 🌗 Shadow deployment — stage a candidate (`make promote-shadow`), score it against live traffic without affecting responses, then commit it through the same gate as a normal promotion (`make promote-commit`)
- 🎓 Training pipeline that runs locally (auto-detects CUDA/MPS/CPU) or dispatches to a free Colab GPU
- 🩺 `/health` reports real model-loaded state; `/admin/reload-model` hot-reloads a new checkpoint (and shadow) without a restart; `/metrics` exposes per-request prediction confidence, per-class counts, and shadow agreement rate alongside HTTP-layer metrics
- 🛡️ Request-ID correlation, structured JSON logs, CORS, a global exception handler, an upload size limit, and per-client rate limiting (`slowapi`)
- 🐳 Multi-stage Docker build (CPU-only torch wheels on Linux) + Docker Compose
- 🔄 CI (lint, type-check, security scan, tests, Docker build) on every push/PR to `main`/`develop`
- 🚚 CD pipeline: builds, vulnerability-scans, smoke-tests, then publishes to GHCR — only on a green CI run
- 📦 Dependency management via **uv**, with Dependabot keeping `uv`/Docker/Actions deps current
- ✅ Pytest suite at ~99% coverage (enforced at 80% in CI)
- 🔍 Ruff lint/format + mypy `--strict` + Bandit security scan + pre-commit hooks
- 🔐 Layered configuration: env vars > `.env` > `configs/default.yaml` > defaults (pydantic-settings)

---

# 📂 Project Structure

```text
.
├── app/
│   ├── api/
│   │   ├── routes/        # predict.py, admin.py
│   │   ├── dependencies.py
│   │   └── middleware.py  # request-ID correlation
│   │
│   ├── core/
│   │   ├── config.py      # pydantic-settings, layered sources
│   │   └── logging.py     # structured JSON logging
│   │
│   ├── models/ schemas/ services/   # request/response schema, InferenceService
│   │
│   ├── ml/
│   │   ├── models/         # MNISTModel
│   │   ├── datasets/       # DataLoader factory
│   │   ├── inference/      # checkpoint loading
│   │   ├── train/          # train.py, engine.py, evaluate.py
│   │   └── utils/          # device selection (CUDA/MPS/CPU)
│   │
│   └── main.py             # FastAPI app, middleware, exception handler
│
├── scripts/                # Colab GPU bridge (colab_server.py, run_remote.py,
│                            # train_dispatch.py) + gpu_sanity_check.py
├── configs/default.yaml    # optional config file, layered under env vars
├── tests/
├── .github/workflows/      # ci.yml, cd.yml
├── .github/dependabot.yml, CODEOWNERS
│
├── pyproject.toml, uv.lock
├── Dockerfile, docker-compose.yml, .dockerignore
├── Makefile
├── TODO.md                 # improvement backlog
└── CLAUDE.md                # notes for AI coding agents working in this repo
```

---

# 🛠 Tech Stack

| Category | Technology |
|-----------|------------|
| Language | Python 3.11+ |
| API | FastAPI |
| ML Framework | PyTorch / torchvision |
| Experiment Tracking | MLflow |
| Validation | Pydantic v2 |
| ASGI Server | Uvicorn + Gunicorn |
| Package Manager | uv |
| Testing | Pytest + pytest-cov |
| Linting / Types | Ruff, mypy (strict), Bandit |
| Containerization | Docker (multi-stage) + Docker Compose |
| CI/CD | GitHub Actions → GHCR |
| Configuration | pydantic-settings (env / `.env` / YAML) |

---

# 🚀 Quick Start

## Clone

```bash
git clone https://github.com/diptu/fast-torch-serving-template.git
cd fast-torch-serving-template
```

## Install dependencies

```bash
uv sync
```

## Run the dev server

```bash
uv run uvicorn app.main:app --reload
```

| | |
|---|---|
| App | http://localhost:8000 |
| Swagger | http://localhost:8000/docs |
| ReDoc | http://localhost:8000/redoc |
| Health | http://localhost:8000/health |
| Version | http://localhost:8000/version |

No trained checkpoint is committed to the repo, so the API starts and serves
predictions from an **untrained** model until you train one (below) —
`/health`'s `model_loaded` field tells you which state you're in. `/version`
reports the package version and the git commit the running image was built
from (`"unknown"` outside Docker — see `APP_GIT_SHA`), handy for confirming
what's actually deployed.

Integrating with this API as a third party? Share
[`docs/external/`](docs/external/README.md) instead of this file — it's
written for API consumers (endpoints, auth, errors, versioning) without any
internal implementation/deployment detail.

---

# 🎓 Train a model

```bash
make train          # Colab GPU if a bridge is live, else local GPU (CUDA/MPS), else CPU
make evaluate        # evaluate checkpoints/model_latest.pth against the val set
make mlflow-ui       # browse tracked runs at http://127.0.0.1:5001
```

Each run saves `checkpoints/model_latest.pth` (served by the API) plus a
run-tagged `checkpoints/model_<run_id>.pth` so you can roll back, registers
a new version under `mnist-cnn` in the MLflow Model Registry
(`APP_MLFLOW_REGISTERED_MODEL_NAME`), and logs a confusion matrix,
per-class precision/recall/F1, and expected calibration error alongside
the usual loss/accuracy — `make evaluate` prints the same report for an
arbitrary checkpoint.

`make promote RUN_ID=<run-id>` points both `model_latest.pth` and the
registry's `"champion"` alias at that run, but refuses a candidate that
regresses vs. the current champion on aggregate `val_accuracy`
(`APP_PROMOTION_MIN_ACCURACY_IMPROVEMENT`), any single class's recall
(`APP_PROMOTION_MAX_RECALL_REGRESSION`), or calibration
(`APP_PROMOTION_MAX_CALIBRATION_REGRESSION`) — `FORCE=1` overrides. `make
promote-list` marks which run_id is currently the champion (and shadow, see
below).

Before committing to a full promotion, `make promote-shadow RUN_ID=<run-id>`
stages a candidate as a **shadow**: `POST /admin/reload-model` loads it
alongside the champion, and every real prediction also runs through the
shadow (best-effort, never affecting the response) with agreement/
disagreement tracked as `predict_shadow_agreement_total` on `GET /metrics`.
Once its live agreement rate looks acceptable, `make promote-commit` runs
it through the exact same gate as `make promote` and clears the shadow slot.

See `scripts/colab_server.py` if you want to dispatch training to a free
Colab GPU instead of your own machine.

---

# 🐳 Docker

```bash
make docker-build              # or: docker build -t fast-torch-serving .
make docker-run                # serves at http://localhost:8000
make docker-run WEB_CONCURRENCY=8
make docker-stop
make docker-logs

# or via Compose:
docker compose up
docker compose down
```

The image ships without a trained checkpoint; `make docker-run` mounts
`./checkpoints` so a locally trained model is picked up automatically.

---

# 🧪 Testing & Code Quality

```bash
make test     # pytest + coverage (fails under 80%)
make lint     # ruff check --fix && ruff format
make check    # mypy --strict + bandit
```

`pre-commit install` (done automatically by `make init`) runs the same lint
checks on every commit.

---

# ⚙️ Configuration

Every setting has a working default in `app/core/config.py`. Precedence,
highest first:

```
environment variables (APP_*) > .env > configs/default.yaml > field defaults
```

Static, non-secret defaults (deployment environment, log level, ML
hyperparameters, paths, MLflow/OTel identifiers) live in
`configs/default.yaml`, already committed — edit it directly to change one
without setting an env var. Copy `.env.example` to `.env` for everything
else: secrets (`APP_ADMIN_TOKEN`, `APP_PREDICT_API_KEY`,
`APP_METRICS_TOKEN`) and machine-/infra-specific settings
(`APP_MAX_CONCURRENT_PREDICTIONS`, `APP_OTEL_EXPORTER_OTLP_ENDPOINT`) that
shouldn't be committed or don't have one sensible default across machines.
Any setting can still be overridden with a real environment variable
regardless of where its default lives — that's how `k8s/configmap.yaml`
forces `APP_ENVIRONMENT=production` in the reference deployment even though
`configs/default.yaml` defaults it to `"development"`.

Key settings: `APP_ADMIN_TOKEN` (enables `POST /admin/reload-model`),
`APP_PREDICT_API_KEY`, `APP_METRICS_TOKEN`, `APP_CORS_ALLOW_ORIGINS`.

## 🔒 Before exposing this publicly

A few things are opt-in rather than on by default, since this is a template
with no assumed deployment target:

- Set `APP_ENVIRONMENT=production` to disable `/docs`, `/redoc`, and
  `/openapi.json` (on by default otherwise).
- Set `APP_PREDICT_API_KEY` / `APP_ADMIN_TOKEN` / `APP_METRICS_TOKEN` to
  require a shared-secret header (`X-API-Key` / `X-Admin-Token` /
  `X-Metrics-Token`) on `/v1/predict`, `/admin/*`, and `/metrics`
  respectively — all three are open by default.
- `/v1/predict` and `/v1/predict/batch` are rate-limited via `slowapi`
  (`APP_PREDICT_RATE_LIMIT`, default `60/minute`, keyed by `X-API-Key` if
  set, else by remote address) on top of the concurrency semaphore above —
  the semaphore bounds concurrent requests, this bounds total request rate.
  Set to `""` to disable, or lower it (or put the API behind an upstream
  gateway) if 60/minute is still too generous for untrusted clients.

---

# 🧠 Request Flow

```text
Client
   │
   ▼
RequestIDMiddleware ── tags the request, echoes X-Request-ID back
   │
   ▼
CORS check
   │
   ▼
POST /v1/predict ── size-capped upload, content-type check
   │
   ▼
InferenceService ── PIL preprocessing → MNISTModel → softmax
   │
   ▼
PredictionResponse (digit, confidence, full distribution)
```

Unhandled errors anywhere in this chain are caught by a global exception
handler that logs the full traceback and returns a structured
`{"detail": ..., "request_id": ...}` response instead of leaking a stack trace.

---

# 📈 Roadmap

- [ ] GPU-accelerated serving (training already auto-detects CUDA/MPS; the
      served Docker image is CPU-only by design — see Dockerfile)
- [ ] Redis cache / Celery workers for async batch inference
- [ ] Full JWT/OAuth auth (currently: shared-secret tokens gate
      `/admin/reload-model` and, optionally, `/v1/predict` and `/metrics`)

Already shipped, despite sometimes being requested as if missing: ONNX
export (`make export-onnx`), Prometheus metrics (`GET /metrics`),
OpenTelemetry tracing (opt-in via `APP_OTEL_EXPORTER_OTLP_ENDPOINT`),
Kubernetes manifests (`k8s/`), an MLflow Model Registry-backed `make
promote` gated on accuracy/recall/calibration, shadow deployment (`make
promote-shadow`/`promote-commit`), and per-client rate limiting on
`/v1/predict` (`slowapi`, `APP_PREDICT_RATE_LIMIT`).

See [`TODO.md`](TODO.md) for the fuller, prioritized improvement backlog.

---

# 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. `make init` to install dependencies and pre-commit hooks
4. Make your changes — `make lint && make check && make test` before pushing
5. Open a Pull Request against `develop` or `main` (CI runs on both)

---

# 📜 License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for details.

---

<div align="center">

Made with ❤️ for the Machine Learning & FastAPI community.

If this project helped you, consider giving it a ⭐

</div>
