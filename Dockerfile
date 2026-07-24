# syntax=docker/dockerfile:1

FROM python:3.14-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.9.5 /uv /uvx /usr/local/bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Install dependencies first, in their own layer, so editing app source
# doesn't invalidate the (slow, torch-heavy) dependency install cache.
# --no-dev drops the "dev" group; the "train" and "onnx" groups (mlflow +
# its heavy transitive deps, the Colab-bridge-only requests/websocket-client,
# onnx/onnxscript) are non-default and excluded automatically — the serving
# path never imports any of them.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY app ./app


FROM python:3.14-slim AS runtime

# Git commit this image was built from, surfaced at runtime via GET
# /version (see APP_GIT_SHA / Settings.git_sha) — passed by the CD workflow
# as --build-arg GIT_SHA=<sha>; "unknown" if built without it (e.g. a bare
# `docker build .` locally).
ARG GIT_SHA=unknown
ENV APP_GIT_SHA=$GIT_SHA

RUN groupadd --system app && useradd --system --gid app --home-dir /app --no-create-home app

WORKDIR /app
# WORKDIR creates /app as root even though a non-root user runs the
# container — without this, gunicorn's control-socket (a file it creates
# directly under /app) fails with a permission error at startup.
RUN chown app:app /app

COPY --from=builder --chown=app:app /app/.venv ./.venv
COPY --from=builder --chown=app:app /app/app ./app
COPY --chown=app:app configs ./configs

RUN mkdir -p /app/checkpoints /app/data && chown -R app:app /app/checkpoints /app/data

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    WEB_CONCURRENCY=4

# No trained checkpoint is baked into the image (checkpoints/ isn't
# committed — see .gitignore). Mount a real one at /app/checkpoints for
# actual predictions, e.g.:
#   docker run -v ./checkpoints:/app/checkpoints ...
# Without one, the service still starts and serves predictions from an
# untrained model (see app/services/inference_service.py).

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/health', timeout=2)" || exit 1

# -w intentionally omitted: gunicorn reads worker count from WEB_CONCURRENCY
# when it's not passed explicitly, so `docker run -e WEB_CONCURRENCY=8 ...`
# works without overriding this whole command.
#
# --timeout/--graceful-timeout spelled out explicitly at gunicorn's own
# defaults (30s) rather than left implicit: a worker's actual predict-request
# ceiling is already bounded well under this by
# APP_PREDICTION_QUEUE_TIMEOUT_SECONDS (5s default, see app/core/config.py),
# so these just document the fallback for anything slower (startup, /admin,
# non-predict routes) instead of leaving the reader to go look it up.
#
# --preload deliberately NOT added: it would let model weights be shared
# copy-on-write across WEB_CONCURRENCY workers instead of each loading its
# own copy — immaterial for this tiny MNIST CNN, but worth it for a bigger
# model. It wouldn't help as-is, though: checkpoint loading happens in
# app.main's lifespan hook, which Uvicorn runs per-worker *after* gunicorn
# forks, not at import time — --preload only shares what's loaded before the
# fork. Whoever revisits this would need to (a) move model loading to
# module/import time so it runs once in gunicorn's master, and (b) confirm
# nothing else set up at import time is fork-unsafe — notably
# app.core.tracing's BatchSpanProcessor, which spawns a background export
# thread in __init__ when APP_OTEL_EXPORTER_OTLP_ENDPOINT is set, and
# threads don't survive fork().
#
# --no-control-socket: gunicorn 26+ otherwise writes a control socket to
# ~/.gunicorn/gunicorn.ctl (i.e. /app/.gunicorn, $HOME for the `app` user)
# on startup. Nothing in this project uses that control interface, and it
# fails outright under a read-only root filesystem (verified: `docker run
# --read-only` logs "Control server error: Read-only file system" — the
# app still serves fine, but it's an unnecessary error on every start; see
# k8s/deployment.yaml's securityContext.readOnlyRootFilesystem). --worker-
# tmp-dir points gunicorn's worker-heartbeat files at tmpfs instead of disk,
# gunicorn's own recommendation for containers where /tmp may be slow or,
# as here, absent from the writable filesystem entirely.
CMD ["gunicorn", "app.main:app", "-k", "uvicorn.workers.UvicornWorker", \
     "-b", "0.0.0.0:8000", "--timeout", "30", "--graceful-timeout", "30", \
     "--no-control-socket", "--worker-tmp-dir", "/dev/shm"]
