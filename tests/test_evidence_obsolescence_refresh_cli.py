"""CLI tests for evidence obsolescence and refresh VOI."""

import json

from typer.testing import CliRunner

from voiage.cli import app


def test_obsolescence_refresh_cli_returns_json(tmp_path) -> None:
    payload = {
        "evidence_values": [10.0, 8.0, 12.0, 6.0, 4.0],
        "evidence_age_months": [2.0, 18.0, 24.0, 15.0, 3.0],
        "half_lives_months": [36.0] * 5,
        "obsolescence_risks": [0.1, 0.8, 0.9, 0.7, 0.2],
        "refresh_costs": [1.0] * 5,
        "living_review_values": [0.5, 3.0, 2.0, 1.0, 0.2],
        "model_refresh_values": [0.2, 2.0, 3.0, 1.0, 0.1],
        "drift_rates": [0.1, 0.7, 0.9, 0.6, 0.2],
    }
    input_file = tmp_path / "refresh.json"
    input_file.write_text(json.dumps(payload), encoding="utf-8")
    result = CliRunner().invoke(
        app,
        [
            "--format",
            "json",
            "calculate-evidence-obsolescence-refresh",
            str(input_file),
        ],
    )
    assert result.exit_code == 0, result.stdout
    output = json.loads(result.stdout)
    assert output["analysis_type"] == "value_of_evidence_obsolescence_refresh"
    assert output["selected_refresh_indices"] == [1, 2]
