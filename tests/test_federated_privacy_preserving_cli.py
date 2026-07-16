"""CLI tests for federated and privacy-preserving VOI."""

import json
from pathlib import Path

from typer.testing import CliRunner

from voiage.cli import app


def test_federated_privacy_preserving_cli(tmp_path: Path) -> None:
    input_file = tmp_path / "federated.json"
    input_file.write_text(
        json.dumps(
            {
                "site_summaries": [[8.0, 7.0], [6.0, 9.0], [7.0, 8.0]],
                "site_weights": [0.2, 0.5, 0.3],
                "privacy_budgets": [1.0, 0.8, 1.2],
                "prior_strategy_values": [6.5, 7.0],
                "strategy_names": ["status_quo", "privacy_preserving"],
                "noise_scale": 0.0,
                "individual_data_access": "blocked",
                "seed": 0,
            }
        )
    )
    result = CliRunner().invoke(
        app,
        ["--format", "json", "calculate-federated-privacy-preserving", str(input_file)],
    )
    assert result.exit_code == 0, result.stdout
    assert (
        json.loads(result.stdout)["analysis_type"]
        == "value_of_federated_privacy_preserving"
    )
