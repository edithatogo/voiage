"""CLI tests for equity-information VOI."""

import json
from pathlib import Path

from typer.testing import CliRunner

from voiage import cli


def test_equity_information_cli_emits_fixture_backed_result(tmp_path: Path) -> None:
    source = json.loads(
        Path(
            "specs/frontier/equity-information/v1/fixtures/normative/"
            "equity-information-input.json"
        ).read_text()
    )
    input_file = tmp_path / "equity-information.json"
    input_file.write_text(json.dumps(source))
    result = CliRunner().invoke(
        cli.app,
        ["--format", "json", "calculate-equity-information", str(input_file)],
    )
    assert result.exit_code == 0, result.output
    assert '"value": 0.75' in result.output
    assert '"method_maturity": "fixture-backed"' in result.output
