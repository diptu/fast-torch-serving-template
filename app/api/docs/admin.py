"""OpenAPI documentation for the /admin routes — summary, description, and
per-status-code response docs. Kept out of app/api/routes/admin.py for the
same reason as app/api/docs/predict.py: keeps the route module focused on
request handling instead of decorator text.
"""

from typing import Any

RELOAD_MODEL_SUMMARY = "Reload model weights from checkpoint"
RELOAD_MODEL_DESCRIPTION = (
    "Reloads model weights from `checkpoint_dir/model_latest.pth` without "
    "restarting the process — e.g. after `make train` produces a new "
    "checkpoint. Also (re)loads `checkpoint_dir/model_shadow.pth` if one is "
    "staged (`make promote-shadow`), so a candidate model starts/stops "
    "being scored against live traffic without ever affecting what's "
    "actually served — see `predict_shadow_agreement_total` on `GET "
    "/metrics`.\n\n"
    "Requires `APP_ADMIN_TOKEN` to be set; requests must include a matching "
    "`X-Admin-Token` header."
)
RELOAD_MODEL_RESPONSES: dict[int | str, dict[str, Any]] = {
    401: {"description": "Missing or invalid X-Admin-Token."},
    503: {"description": "Admin endpoints are disabled (APP_ADMIN_TOKEN not set)."},
}
