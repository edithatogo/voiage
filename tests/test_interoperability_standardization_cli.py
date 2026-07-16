"""CLI contract tests for interoperability and standardization VOI."""

import json
from pathlib import Path

from typer.testing import CliRunner

from voiage.cli import _generate_config_template, app


def test_interoperability_standardization_cli_returns_json(tmp_path: Path) -> None:
    input_file = tmp_path / "interoperability.json"
    input_file.write_text(
        json.dumps(_generate_config_template("interoperability-standardization")),
        encoding="utf-8",
    )
    result = CliRunner().invoke(
        app,
        [
            "--format",
            "json",
            "calculate-interoperability-standardization",
            str(input_file),
        ],
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["method_maturity"] == "fixture-backed"
    assert payload["diagnostics"]["evidence_reuse"] is True
