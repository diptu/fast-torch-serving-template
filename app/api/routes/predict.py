import asyncio
from collections.abc import Callable
from typing import Any, TypeVar

from fastapi import (
    APIRouter,
    Depends,
    Header,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from PIL import UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from app.api.dependencies import InferenceServiceDep
from app.api.docs.predict import (
    ALLOWED_CONTENT_TYPES,
    MAX_BATCH_SIZE,
    MAX_UPLOAD_BYTES,
    PREDICT_BATCH_DESCRIPTION,
    PREDICT_BATCH_RESPONSES,
    PREDICT_BATCH_SUMMARY,
    PREDICT_DESCRIPTION,
    PREDICT_RESPONSES,
    PREDICT_SUMMARY,
)
from app.core.config import get_settings
from app.core.rate_limit import limiter
from app.core.security import secrets_match
from app.schemas.prediction import BatchPredictionResponse, PredictionResponse

settings = get_settings()

T = TypeVar("T")

# "" disables rate limiting entirely (see APP_PREDICT_RATE_LIMIT in
# config.py) — a no-op decorator keeps the route definitions below
# unconditional either way.
_rate_limit: Callable[[Callable[..., Any]], Callable[..., Any]] = (
    limiter.limit(settings.predict_rate_limit)
    if settings.predict_rate_limit
    else (lambda f: f)
)


def _verify_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Reject the request if ``APP_PREDICT_API_KEY`` is set and mismatched.

    Parameters
    ----------
    x_api_key : str, optional
        The ``X-API-Key`` header.

    Raises
    ------
    HTTPException
        401 if a key is configured and ``x_api_key`` doesn't match.
    """
    # Opt-in: unset (the default) means no key is required, unchanged from
    # before this setting existed.
    configured = settings.predict_api_key
    if configured and not secrets_match(x_api_key, configured):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key"
        )


router = APIRouter(
    prefix="/predict", tags=["prediction"], dependencies=[Depends(_verify_api_key)]
)

_CHUNK_SIZE = 64 * 1024

# Caps how many inference calls run at once across this worker process.
# InferenceService's predict methods are synchronous/CPU-bound, so they
# always run via run_in_threadpool below — without this cap, a burst of
# uploads would still queue in the (much larger) default thread pool and
# thrash the CPU with unbounded concurrent forward passes. A batched request
# still only takes one slot, same as a single-image one.
_inference_semaphore = asyncio.Semaphore(settings.max_concurrent_predictions)


async def _read_limited(file: UploadFile, max_bytes: int) -> bytes:
    """Read an upload in bounded chunks, rejecting oversized bodies early.

    Parameters
    ----------
    file : UploadFile
    max_bytes : int

    Returns
    -------
    bytes

    Raises
    ------
    HTTPException
        413 if the body exceeds ``max_bytes`` before being fully read.

    Notes
    -----
    Content-Length alone can't be trusted: it's absent for chunked transfer
    encoding and isn't verified against the actual body by the ASGI server.
    """
    chunks: list[bytes] = []
    total = 0
    while chunk := await file.read(_CHUNK_SIZE):
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail=f"File too large (max {max_bytes // (1024 * 1024)}MB)",
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _check_content_type(file: UploadFile) -> None:
    """Reject the request if ``file``'s content type isn't allowed.

    Parameters
    ----------
    file : UploadFile

    Raises
    ------
    HTTPException
        415 if ``file.content_type`` isn't in ``ALLOWED_CONTENT_TYPES``.
    """
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported content type: {file.content_type}",
        )


async def _run_bounded(fn: Callable[..., T], *args: object) -> T:
    """Run ``fn(*args)`` off the event loop, bounded by capacity/timeout.

    Parameters
    ----------
    fn : Callable
        Sync/CPU-bound callable, e.g. an ``InferenceService`` predict method.
    *args : object
        Positional arguments passed to ``fn``.

    Returns
    -------
    T
        Whatever ``fn`` returns.

    Raises
    ------
    HTTPException
        503 if the semaphore/timeout is exceeded; 400 if the image can't be
        decoded.

    Notes
    -----
    Shared by the single and batch predict routes.
    """
    try:
        async with (
            asyncio.timeout(settings.prediction_queue_timeout_seconds),
            _inference_semaphore,
        ):
            result: T = await run_in_threadpool(fn, *args)
            return result
    except TimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server is at capacity, please retry shortly",
            headers={"Retry-After": "2"},
        ) from exc
    except (UnidentifiedImageError, OSError) as exc:
        # Known, expected failure mode: an upload isn't a decodable image.
        # Anything else is unexpected and propagates to the global handler.
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Could not process image"
        ) from exc


@router.post(
    "",
    response_model=PredictionResponse,
    summary=PREDICT_SUMMARY,
    description=PREDICT_DESCRIPTION,
    responses=PREDICT_RESPONSES,
)
@_rate_limit
async def predict_digit(
    request: Request,  # noqa: ARG001 — required by slowapi to read app.state.limiter
    file: UploadFile,
    inference_service: InferenceServiceDep,
) -> PredictionResponse:
    """Classify a single uploaded digit image.

    Parameters
    ----------
    request : Request
        Unused directly; slowapi's decorator inspects it for the limiter.
    file : UploadFile
    inference_service : InferenceService
        Injected via ``InferenceServiceDep``.

    Returns
    -------
    PredictionResponse
    """
    _check_content_type(file)
    image_bytes = await _read_limited(file, MAX_UPLOAD_BYTES)

    # predict_image() is sync/CPU-bound — run_in_threadpool (inside
    # _run_bounded) keeps it off the event loop so it can't stall other
    # requests (including /health) while a prediction is in flight.
    predicted_digit, probabilities = await _run_bounded(
        inference_service.predict_image, image_bytes
    )

    return PredictionResponse(
        predicted_digit=predicted_digit,
        confidence=probabilities[predicted_digit],
        probabilities=probabilities,
    )


@router.post(
    "/batch",
    response_model=BatchPredictionResponse,
    summary=PREDICT_BATCH_SUMMARY,
    description=PREDICT_BATCH_DESCRIPTION,
    responses=PREDICT_BATCH_RESPONSES,
)
@_rate_limit
async def predict_batch(
    request: Request,  # noqa: ARG001 — required by slowapi to read app.state.limiter
    files: list[UploadFile],
    inference_service: InferenceServiceDep,
) -> BatchPredictionResponse:
    """Classify a batch of uploaded digit images in one forward pass.

    Parameters
    ----------
    request : Request
        Unused directly; slowapi's decorator inspects it for the limiter.
    files : list of UploadFile
        Up to ``MAX_BATCH_SIZE`` images.
    inference_service : InferenceService
        Injected via ``InferenceServiceDep``.

    Returns
    -------
    BatchPredictionResponse
    """
    # An empty list never reaches here: FastAPI's own validation returns 422
    # first since `files` is a required field with no multipart parts sent.
    if len(files) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Too many files (max {MAX_BATCH_SIZE} per batch)",
        )
    for file in files:
        _check_content_type(file)

    images = [await _read_limited(file, MAX_UPLOAD_BYTES) for file in files]

    # One forward pass for the whole batch (see InferenceService.predict_batch)
    # — this is what makes it a real batch endpoint rather than a loop.
    results = await _run_bounded(inference_service.predict_batch, images)

    return BatchPredictionResponse(
        predictions=[
            PredictionResponse(
                predicted_digit=digit, confidence=probs[digit], probabilities=probs
            )
            for digit, probs in results
        ]
    )
