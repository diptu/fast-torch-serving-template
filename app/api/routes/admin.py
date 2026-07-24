from fastapi import APIRouter, Header, HTTPException, status

from app.api.dependencies import InferenceServiceDep
from app.api.docs.admin import (
    RELOAD_MODEL_DESCRIPTION,
    RELOAD_MODEL_RESPONSES,
    RELOAD_MODEL_SUMMARY,
)
from app.core.config import get_settings
from app.core.security import secrets_match

router = APIRouter(prefix="/admin", tags=["admin"])


def _check_admin_token(x_admin_token: str | None) -> None:
    """Reject the request unless a matching admin token is configured.

    Parameters
    ----------
    x_admin_token : str, optional
        The ``X-Admin-Token`` header.

    Raises
    ------
    HTTPException
        503 if ``APP_ADMIN_TOKEN`` isn't set; 401 if it's set but mismatched.
    """
    configured = get_settings().admin_token
    if not configured:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Admin endpoints are disabled (APP_ADMIN_TOKEN not set)",
        )
    if not secrets_match(x_admin_token, configured):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid admin token"
        )


@router.post(
    "/reload-model",
    summary=RELOAD_MODEL_SUMMARY,
    description=RELOAD_MODEL_DESCRIPTION,
    responses=RELOAD_MODEL_RESPONSES,
)
def reload_model(
    inference_service: InferenceServiceDep,
    x_admin_token: str | None = Header(default=None),
) -> dict[str, bool]:
    """Reload model weights from checkpoint, without restarting the process.

    Parameters
    ----------
    inference_service : InferenceService
        Injected via ``InferenceServiceDep``.
    x_admin_token : str, optional
        The ``X-Admin-Token`` header.

    Returns
    -------
    dict of str to bool
        ``{"reloaded": True, "checkpoint_loaded": ..., "shadow_loaded": ...}``.
    """
    _check_admin_token(x_admin_token)
    checkpoint_loaded = inference_service.reload()
    return {
        "reloaded": True,
        "checkpoint_loaded": checkpoint_loaded,
        "shadow_loaded": inference_service.shadow_loaded,
    }
