"""Structured, application-owned logging for VOIAGE."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
import json
import logging
import os
import pathlib  # noqa: TC003 - Pydantic resolves this annotation at runtime
import sys
from typing import TYPE_CHECKING, ClassVar, final, override
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

if TYPE_CHECKING:
    from collections.abc import Generator, Mapping

_CONTEXT: ContextVar[dict[str, str] | None] = ContextVar(
    "voiage_log_context", default=None
)
_OWNED_HANDLER = "_voiage_handler"
_SENSITIVE_FRAGMENTS = ("authorization", "password", "secret", "token", "api_key")


def _safe_context(values: Mapping[str, object]) -> dict[str, str]:
    """Stringify safe context and redact common credential-bearing fields."""
    return {
        key: "[REDACTED]"
        if any(fragment in key.casefold() for fragment in _SENSITIVE_FRAGMENTS)
        else str(value)
        for key, value in values.items()
    }


class LoggingSettings(BaseModel):
    """Pydantic-v2 validated logging settings."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid", frozen=True, defer_build=True
    )
    level: str = "ERROR"
    json_output: bool = False
    console: bool = True
    log_file: pathlib.Path | None = None
    service: str = "voiage"
    run_id: str = Field(default_factory=lambda: uuid4().hex)

    @field_validator("level")
    @classmethod
    def validate_level(cls, value: str) -> str:
        """Normalize and validate a standard-library logging level."""
        normalized = value.upper()
        if normalized not in logging.getLevelNamesMapping():
            raise ValueError(f"unknown logging level: {value}")
        return normalized

    @classmethod
    def from_environment(cls, **overrides: object) -> LoggingSettings:
        """Load the stable ``VOIAGE_LOG_*`` environment contract."""
        values: dict[str, object] = {
            "level": os.getenv("VOIAGE_LOG_LEVEL", "ERROR"),
            "json_output": os.getenv("VOIAGE_LOG_FORMAT", "human").lower() == "json",
            "run_id": os.getenv("VOIAGE_RUN_ID") or uuid4().hex,
        }
        values.update(overrides)
        return cls.model_validate(values)


@final
class _ContextFilter(logging.Filter):
    def __init__(self, settings: LoggingSettings) -> None:
        super().__init__()
        self.settings: LoggingSettings = settings

    @override
    def filter(self, record: logging.LogRecord) -> bool:
        record.service = self.settings.service
        record.run_id = self.settings.run_id
        record.context = dict(_CONTEXT.get() or {})
        return True


@final
class JsonFormatter(logging.Formatter):
    """Emit newline-delimited JSON for CI and observability systems."""

    @override
    def format(self, record: logging.LogRecord) -> str:
        """Serialize one log record without relying on global state."""
        payload: dict[str, object] = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "service": getattr(record, "service", "voiage"),
            "run_id": getattr(record, "run_id", None),
            "message": record.getMessage(),
        }
        payload.update(getattr(record, "context", {}))
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)


@contextmanager
def log_context(**values: object) -> Generator[None]:
    """Bind context for the current thread, task, or async execution path."""
    merged = {
        **(_CONTEXT.get() or {}),
        **_safe_context(values),
    }
    token = _CONTEXT.set(merged)
    try:
        yield
    finally:
        _CONTEXT.reset(token)


def configure_logging(settings: LoggingSettings | None = None) -> logging.Logger:
    """Configure VOIAGE-owned handlers without disturbing host applications."""
    settings = settings or LoggingSettings.from_environment()
    logger = logging.getLogger("voiage")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    for handler in tuple(logger.handlers):
        if getattr(handler, _OWNED_HANDLER, False):
            logger.removeHandler(handler)
            handler.close()

    context_filter = _ContextFilter(settings)
    human = logging.Formatter("%(levelname)s:%(name)s:%(message)s [run_id=%(run_id)s]")
    formatter: logging.Formatter = JsonFormatter() if settings.json_output else human
    if settings.console:
        console = logging.StreamHandler(sys.stderr)
        console.setLevel(settings.level)
        console.setFormatter(formatter)
        console.addFilter(context_filter)
        setattr(console, _OWNED_HANDLER, True)
        logger.addHandler(console)
    if settings.log_file is not None:
        settings.log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(
            settings.log_file, mode="a", encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(JsonFormatter())
        file_handler.addFilter(context_filter)
        setattr(file_handler, _OWNED_HANDLER, True)
        logger.addHandler(file_handler)
    logger.info("logging_configured")
    return logger
