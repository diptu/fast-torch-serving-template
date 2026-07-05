import json
import logging
import sys

import pytest

from app.core.logging import JsonFormatter, get_logger, setup_logging


@pytest.fixture
def isolated_root_logger():
    """setup_logging() mutates the root logger globally; save/restore it so
    tests don't leak handlers into each other or into pytest's own logging."""
    root = logging.getLogger()
    original_handlers = list(root.handlers)
    original_level = root.level
    yield root
    root.handlers = original_handlers
    root.setLevel(original_level)


def _make_record(**overrides: object) -> logging.LogRecord:
    defaults = dict(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=42,
        msg="hello %s",
        args=("world",),
        exc_info=None,
    )
    defaults.update(overrides)
    return logging.LogRecord(**defaults)  # type: ignore[arg-type]


def test_json_formatter_produces_expected_fields() -> None:
    record = _make_record()
    payload = json.loads(JsonFormatter().format(record))

    assert payload["level"] == "INFO"
    assert payload["message"] == "hello world"
    assert payload["line"] == 42
    assert "timestamp" in payload


def test_json_formatter_includes_exception_info() -> None:
    try:
        raise ValueError("boom")
    except ValueError:
        record = _make_record(exc_info=sys.exc_info())

    payload = json.loads(JsonFormatter().format(record))
    assert "ValueError: boom" in payload["exception"]


def test_json_formatter_merges_extra_data() -> None:
    record = _make_record()
    record.extra_data = {"val_accuracy": 0.98}

    payload = json.loads(JsonFormatter().format(record))
    assert payload["val_accuracy"] == 0.98


def test_setup_logging_configures_root_logger_once(isolated_root_logger) -> None:
    isolated_root_logger.addHandler(logging.NullHandler())

    setup_logging(log_level="DEBUG")

    assert isolated_root_logger.level == logging.DEBUG
    assert len(isolated_root_logger.handlers) == 1
    assert isinstance(isolated_root_logger.handlers[0].formatter, JsonFormatter)
    assert logging.getLogger("uvicorn").level == logging.WARNING
    assert logging.getLogger("httpx").level == logging.WARNING


def test_get_logger_returns_named_logger() -> None:
    logger = get_logger("my.module")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "my.module"
