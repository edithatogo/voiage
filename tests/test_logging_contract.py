from __future__ import annotations

import json
import logging

from pydantic import ValidationError
import pytest

from voiage.logging import LoggingSettings, configure_logging, log_context


def test_logging_settings_are_pydantic_v2_and_strict() -> None:
    assert LoggingSettings.model_fields
    with pytest.raises(ValidationError):
        LoggingSettings(level="verbose")


def test_json_logging_carries_run_and_bound_context(tmp_path) -> None:
    destination = tmp_path / "run.jsonl"
    logger = configure_logging(
        LoggingSettings(console=False, log_file=destination, run_id="test-run")
    )
    with log_context(
        track="C08",
        command="evpi",
        access_token="never-log-this",  # noqa: S106 - verifies redaction
    ):
        logger.info("analysis_started")
    for handler in logger.handlers:
        handler.flush()

    records = [json.loads(line) for line in destination.read_text().splitlines()]
    assert records[-1]["message"] == "analysis_started"
    assert records[-1]["run_id"] == "test-run"
    assert records[-1]["track"] == "C08"
    assert records[-1]["access_token"] == "[REDACTED]"  # noqa: S105


def test_configuration_preserves_root_handlers() -> None:
    root = logging.getLogger()
    sentinel = logging.NullHandler()
    root.addHandler(sentinel)
    try:
        configure_logging(LoggingSettings(console=False))
        assert sentinel in root.handlers
    finally:
        root.removeHandler(sentinel)
