"""CLI tests for ambiguity and distribution-shift VOI."""

import json
from pathlib import Path

from typer.testing import CliRunner

from voiage import cli


def test_ambiguity_distribution_shift_cli_emits_fixture_backed_result(
    tmp_path: Path,
) -> None:
    source = json.loads(
        Path(
            "specs/frontier/ambiguity-distribution-shift/v1/fixtures/normative/"
            "ambiguity-distribution-shift-input.json"
        ).read_text()
    )
    input_file = tmp_path / "ambiguity-distribution-shift.json"
    input_file.write_text(json.dumps(source))
    result = CliRunner().invoke(
        cli.app,
        ["--format", "json", "calculate-ambiguity-distribution-shift", str(input_file)],
    )
    assert result.exit_code == 0, result.output
    assert '"value": 2.369999999999999' in result.output
    assert '"robust_strategy_name": "B"' in result.output
