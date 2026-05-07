"""Property-based and fuzzing tests for CLI input validation."""

from __future__ import annotations

import json

from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from typer.testing import CliRunner

from voiage import cli

runner = CliRunner()


def _is_valid_json(payload: str) -> bool:
    try:
        json.loads(payload)
    except json.JSONDecodeError:
        return False
    return True


@given(payload=st.text(min_size=1, max_size=128))
@settings(
    deadline=None,
    max_examples=40,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_structural_evpi_rejects_fuzzed_invalid_json(tmp_path, payload: str) -> None:
    """Reject malformed structural-VOI config JSON across a wide input range."""
    assume(not _is_valid_json(payload))
    config = tmp_path / "config.json"
    config.write_text(payload, encoding="utf-8")

    result = runner.invoke(cli.app, ["calculate-structural-evpi", str(config)])

    assert result.exit_code != 0
    assert "Invalid JSON" in result.stderr or "Error" in result.stderr


def test_structural_evpi_rejects_empty_config_file(tmp_path) -> None:
    """Reject an empty structural-VOI config file explicitly."""
    config = tmp_path / "empty.json"
    config.write_text("", encoding="utf-8")

    result = runner.invoke(cli.app, ["calculate-structural-evpi", str(config)])

    assert result.exit_code != 0
    assert "Invalid JSON" in result.stderr or "Error" in result.stderr
