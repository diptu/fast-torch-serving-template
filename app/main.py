from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app import __version__
from app.api.dependencies import InferenceServiceDep
from app.api.middleware import RequestIDMiddleware
from app.api.routes.admin import router as admin_router
from app.api.routes.predict import router as predict_router
from app.core.config import get_settings
from app.core.logging import get_logger, request_id_var, setup_logging
from app.core.rate_limit import limiter
from app.core.security import secrets_match
from app.core.tracing import setup_tracing
from app.services.inference_service import get_inference_service

settings = get_settings()

setup_logging(settings.log_level)
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:  # noqa: ARG001 — required by FastAPI's lifespan signature
    """Load the model checkpoint at startup, before serving any request.

    Parameters
    ----------
    app : FastAPI

    Notes
    -----
    Loaded now, not on the first request, so /health reflects reality
    immediately and cold-start cost doesn't land on whichever user happens
    to hit the API first.
    """
    get_inference_service()
    logger.info("Application startup complete")
    yield


# /docs, /redoc, and /openapi.json reveal internal schemas and endpoint
# structure, so they're switched off outside "development" (see
# APP_ENVIRONMENT in app/core/config.py).
_docs_enabled = settings.environment != "production"
app = FastAPI(
    title="Fast Torch Serving Template",
    lifespan=lifespan,
    docs_url="/docs" if _docs_enabled else None,
    redoc_url="/redoc" if _docs_enabled else None,
    openapi_url="/openapi.json" if _docs_enabled else None,
)
# Versioned so the prediction contract can evolve later without breaking
# existing clients (a v2 router could be added and included alongside this
# one). /health and /admin are operational endpoints, not part of the
# versioned public API, so they stay unprefixed.
app.include_router(predict_router, prefix="/v1")
app.include_router(admin_router)

# Backs the @limiter.limit(...) decorators in app/api/routes/predict.py —
# slowapi looks these up on app.state at request time.
app.state.limiter = limiter
# slowapi's own handler is typed for RateLimitExceeded specifically, narrower
# than Starlette's Exception-typed handler signature — slowapi's own
# internals register it the same way with the same type: ignore.
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]


def _verify_metrics_token(x_metrics_token: str | None = Header(default=None)) -> None:
    """Reject the request if ``APP_METRICS_TOKEN`` is set and mismatched.

    Parameters
    ----------
    x_metrics_token : str, optional
        The ``X-Metrics-Token`` header.

    Raises
    ------
    HTTPException
        401 if a token is configured and ``x_metrics_token`` doesn't match.
    """
    # Opt-in: unset (the default) means no token is required, unchanged from
    # before this setting existed — see APP_METRICS_TOKEN in config.py.
    configured = settings.metrics_token
    if configured and not secrets_match(x_metrics_token, configured):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid metrics token"
        )


# GET /metrics in Prometheus text format: request count/latency by path,
# status code, method — plus Python/process-level metrics, out of the box.
Instrumentator().instrument(app).expose(
    app, dependencies=[Depends(_verify_metrics_token)]
)

# No-op unless APP_OTEL_EXPORTER_OTLP_ENDPOINT is set — see app/core/tracing.py.
setup_tracing(app)

# Added last so it's the outermost middleware: the request ID needs to be
# set before anything else (including CORS) runs, so it's available to logs
# and error responses no matter where a request fails.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestIDMiddleware)


@app.exception_handler(Exception)
async def unhandled_exception_handler(
    request: Request,  # noqa: ARG001 — both required by FastAPI's exception_handler signature
    exc: Exception,  # noqa: ARG001
) -> JSONResponse:
    """Log the traceback and return a structured 500 with the request ID.

    Parameters
    ----------
    request : Request
    exc : Exception

    Returns
    -------
    JSONResponse
        500, body ``{"detail": ..., "request_id": ...}``.
    """
    logger.exception("Unhandled exception")
    request_id = request_id_var.get()
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error", "request_id": request_id},
        # RequestIDMiddleware can't attach this itself here: call_next()
        # raised before reaching its own header-setting line.
        headers={"X-Request-ID": request_id} if request_id else None,
    )


@app.get("/version")
def version() -> dict[str, str]:
    """Report package version and the git commit this image was built from.

    Returns
    -------
    dict of str to str
        ``{"version": ..., "git_sha": ...}`` (see ``APP_GIT_SHA`` / the
        Dockerfile's ``GIT_SHA`` build arg).
    """
    return {"version": __version__, "git_sha": settings.git_sha}


@app.get("/health")
def health(inference_service: InferenceServiceDep) -> dict[str, bool | str]:
    """Liveness: is the process up and responsive at all.

    Parameters
    ----------
    inference_service : InferenceService
        Injected via ``InferenceServiceDep``.

    Returns
    -------
    dict of str to (bool or str)
        Always 200 — an untrained model is still a functioning process,
        just not a useful one, which is what /ready is for.
    """
    return {"status": "ok", "model_loaded": inference_service.checkpoint_loaded}


@app.get("/ready")
def ready(inference_service: InferenceServiceDep) -> dict[str, bool]:
    """Readiness: is this instance actually able to serve useful predictions.

    Parameters
    ----------
    inference_service : InferenceService
        Injected via ``InferenceServiceDep``.

    Returns
    -------
    dict of str to bool

    Raises
    ------
    HTTPException
        503 if no checkpoint is loaded.

    Notes
    -----
    Orchestrators should gate traffic on this, not /health — a probe only
    sees the status code, not the JSON body, so ``model_loaded`` needs its
    own endpoint to matter for routing decisions.
    """
    if not inference_service.checkpoint_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No checkpoint loaded",
        )
    return {"ready": True}
