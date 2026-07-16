"""CLI contract tests for regulatory and market-access VOI."""

import json
from pathlib import Path

from typer.testing import CliRunner

from voiage.cli import _generate_config_template, app


def test_regulatory_market_access_cli_returns_json(tmp_path: Path) -> None:
    input_file = tmp_path / "market-access.json"
    input_file.write_text(
        json.dumps(_generate_config_template("regulatory-market-access")),
        encoding="utf-8",
    )
    result = CliRunner().invoke(
        app,
        ["--format", "json", "calculate-regulatory-market-access", str(input_file)],
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["method_maturity"] == "fixture-backed"
    assert payload["diagnostics"]["market_access_decision"] is True
