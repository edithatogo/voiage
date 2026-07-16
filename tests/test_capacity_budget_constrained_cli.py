"""CLI contract tests for constrained VOI."""

import json
from pathlib import Path

from typer.testing import CliRunner

from voiage.cli import app


def test_capacity_budget_constrained_cli(tmp_path: Path) -> None:
    path = tmp_path / "constrained.json"
    path.write_text(json.dumps({
        "scenario_values": [[10, 8], [6, 11]],
        "strategy_costs": [2, 5],
        "strategy_capacity": [1, 2],
        "budget": 5,
        "capacity": 2,
        "strategy_names": ["small", "balanced"],
    }))
    result = CliRunner().invoke(app, ["--format", "json", "calculate-capacity-budget-constrained", str(path)])
    assert result.exit_code == 0, result.stdout
    assert json.loads(result.stdout)["analysis_type"] == "value_of_capacity_budget_constrained"
