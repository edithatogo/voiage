"""CLI tests for replication and reproducibility VOI."""

import json

from typer.testing import CliRunner

from voiage.cli import app


def test_replication_reproducibility_cli_returns_json(tmp_path) -> None:
    payload = {
        "evidence_values": [10.0, 8.0, 6.0, 4.0],
        "replication_probabilities": [0.9, 0.5, 0.7, 0.2],
        "reproducibility_failure_risks": [0.1, 0.4, 0.2, 0.8],
        "audit_costs": [1.0, 1.0, 1.0, 1.0],
        "reanalysis_values": [2.0, 1.0, 1.5, 0.5],
        "credibility_adjustments": [1.2, 1.0, 1.1, 0.8],
        "evidence_downgrades": [0.1, 0.3, 0.2, 0.6],
    }
    input_file = tmp_path / "replication.json"
    input_file.write_text(json.dumps(payload), encoding="utf-8")

    result = CliRunner().invoke(
        app,
        ["--format", "json", "calculate-replication-reproducibility", str(input_file)],
    )

    assert result.exit_code == 0, result.stdout
    output = json.loads(result.stdout)
    assert output["analysis_type"] == "value_of_replication_reproducibility"
    assert output["selected_replication_indices"] == [0, 2]
