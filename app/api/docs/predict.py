"""OpenAPI documentation for the /predict routes — summaries, descriptions,
and per-status-code response docs — plus the upload-limit constants that
documentation describes. Kept out of app/api/routes/predict.py so that
module stays focused on request handling instead of decorator text; the
constants live here (rather than duplicated as prose) so the docs can't
drift out of sync with the values routes/predict.py actually enforces.
"""

from typing import Any

ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg", "image/jpg"}
MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # generous for a single digit image
MAX_BATCH_SIZE = 32

_MAX_UPLOAD_MB = MAX_UPLOAD_BYTES // (1024 * 1024)
_CONTENT_TYPES_TEXT = ", ".join(sorted(ALLOWED_CONTENT_TYPES))
_API_KEY_NOTE = (
    "If `APP_PREDICT_API_KEY` is set, requests must include a matching "
    "`X-API-Key` header."
)

PREDICT_SUMMARY = "Classify a single handwritten digit"
PREDICT_DESCRIPTION = (
    f"Upload a single image ({_CONTENT_TYPES_TEXT}, max {_MAX_UPLOAD_MB}MB) "
    "of a handwritten digit and get back the predicted digit (0-9), its "
    "confidence, and the full probability distribution across all 10 "
    f"classes.\n\n{_API_KEY_NOTE}"
)
PREDICT_RESPONSES: dict[int | str, dict[str, Any]] = {
    400: {"description": "Upload was not a decodable image."},
    401: {"description": "Missing or invalid X-API-Key."},
    413: {
        "description": f"Upload exceeded the maximum allowed size ({_MAX_UPLOAD_MB}MB)."
    },
    415: {"description": f"Content-Type is not one of: {_CONTENT_TYPES_TEXT}."},
    503: {"description": "Server is at capacity; retry after the given Retry-After."},
}

PREDICT_BATCH_SUMMARY = "Classify a batch of handwritten digits"
PREDICT_BATCH_DESCRIPTION = (
    f"Upload up to {MAX_BATCH_SIZE} images ({_CONTENT_TYPES_TEXT}, max "
    f"{_MAX_UPLOAD_MB}MB each) in a single request and get back one "
    "prediction per image, computed in a single batched forward pass rather "
    f"than one call per image.\n\n{_API_KEY_NOTE}"
)
PREDICT_BATCH_RESPONSES: dict[int | str, dict[str, Any]] = {
    400: {
        "description": (
            f"More than {MAX_BATCH_SIZE} files, or one of the uploads was "
            "not a decodable image."
        )
    },
    401: {"description": "Missing or invalid X-API-Key."},
    413: {
        "description": (
            f"One of the uploads exceeded the maximum allowed size "
            f"({_MAX_UPLOAD_MB}MB)."
        )
    },
    415: {
        "description": (
            f"Content-Type is not one of: {_CONTENT_TYPES_TEXT} for one of the uploads."
        )
    },
    503: {"description": "Server is at capacity; retry after the given Retry-After."},
}
