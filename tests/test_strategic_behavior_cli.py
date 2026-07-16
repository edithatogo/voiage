"""CLI tests for strategic behavior VOI."""

import json

from typer.testing import CliRunner

from voiage.cli import app


def test_strategic_behavior_cli_returns_json(tmp_path) -> None:
    payload = {
        "scenario_values": [10.0, 8.0, 6.0, 4.0],
        "equilibrium_probabilities": [0.8, 0.6, 0.3, 0.2],
        "incentive_response_values": [2.0, 1.0, 0.5, 0.2],
        "disclosure_values": [1.0, 2.0, 0.5, 0.2],
        "bargaining_values": [1.5, 1.0, 0.2, 0.1],
        "adversarial_risks": [0.1, 0.2, 0.4, 0.8],
        "response_sensitivities": [0.2, 0.5, 0.8, 0.9],
        "strategic_regrets": [0.5, 0.4, 0.2, 0.1],
    }
    input_file = tmp_path / "strategic.json"
    input_file.write_text(json.dumps(payload), encoding="utf-8")
    result = CliRunner().invoke(
        app, ["--format", "json", "calculate-strategic-behavior", str(input_file)]
    )
    assert result.exit_code == 0, result.stdout
    output = json.loads(result.stdout)
    assert output["analysis_type"] == "value_of_strategic_behavior"
    assert output["selected_scenario_indices"] == [0, 1]
