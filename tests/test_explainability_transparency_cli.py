"""CLI contract tests for explainability and transparency VOI."""

import json
from pathlib import Path

from typer.testing import CliRunner

from voiage.cli import _generate_config_template, app


def test_explainability_transparency_cli_returns_json(tmp_path: Path) -> None:
    input_file = tmp_path / "explainability.json"
    input_file.write_text(
        json.dumps(_generate_config_template("explainability-transparency")),
        encoding="utf-8",
    )
    result = CliRunner().invoke(
        app,
        [
            "--format",
            "json",
            "calculate-explainability-transparency",
            str(input_file),
        ],
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["method_maturity"] == "fixture-backed"
    assert payload["diagnostics"]["transparency_evidence"] is True
