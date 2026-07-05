import json
import logging
import sys
from contextvars import ContextVar
from typing import Any

# Set by RequestIDMiddleware for the duration of a request so every log line
# emitted while handling it can be correlated back to that request.
request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)


class JsonFormatter(logging.Formatter):
    """Formats log records as single-line JSON for log aggregation systems."""

    def format(self, record: logging.LogRecord) -> str:
        """Render one log record as a JSON string.

        Parameters
        ----------
        record : logging.LogRecord

        Returns
        -------
        str
            JSON object with ``timestamp``/``level``/``message``/``module``/
            ``line``, plus ``request_id``, ``exception``, and any
            ``extra_data`` when present.
        """
        log_object: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno,
        }

        request_id = request_id_var.get()
        if request_id is not None:
            log_object["request_id"] = request_id

        if record.exc_info:
            log_object["exception"] = self.formatException(record.exc_info)

        if hasattr(record, "extra_data"):
            log_object.update(record.extra_data)

        return json.dumps(log_object)


def setup_logging(log_level: str = "INFO") -> None:
    """Configure the root logger with a JSON stdout handler.

    Parameters
    ----------
    log_level : str, default "INFO"
    """
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Avoid duplicate log lines if setup_logging() is called more than once
    # (e.g. once at import time, again in a test fixture).
    if logger.hasHandlers():
        logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)

    # uvicorn/httpx access logs are noisy at INFO and duplicate what's
    # already captured per-request elsewhere; keep only their warnings+.
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a module logger.

    Parameters
    ----------
    name : str
        Conventionally ``__name__``.

    Returns
    -------
    logging.Logger
    """
    return logging.getLogger(name)
