"""CLI tests for adaptive learning and bandit VOI."""

import json
from pathlib import Path

from typer.testing import CliRunner

from voiage.cli import app


def test_adaptive_learning_bandit_cli(tmp_path: Path) -> None:
    input_file = tmp_path / "bandit.json"
    input_file.write_text(
        json.dumps(
            {
                "reward_samples": [[0.5, 0.6, 0.7, 0.8], [0.7, 0.8, 0.9, 1.0]],
                "arm_names": ["control", "adaptive"],
                "policy": "ucb",
                "horizon": 4,
                "exploration_cost": 0.01,
                "confidence": 2.0,
            }
        )
    )
    result = CliRunner().invoke(
        app, ["--format", "json", "calculate-adaptive-learning-bandit", str(input_file)]
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["analysis_type"] == "value_of_adaptive_learning_bandit"
    assert payload["selected_arms"] == [0, 1, 1, 0]
    assert payload["method_maturity"] == "fixture-backed"
