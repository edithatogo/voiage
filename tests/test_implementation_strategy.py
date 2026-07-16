"""Runtime tests for implementation-strategy comparison VOI."""

from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from voiage import cli
from voiage.methods.implementation_strategy import (
    value_of_implementation_strategy_comparison,
)

runner = CliRunner()


def test_implementation_strategy_comparison_reports_adoption_diagnostics() -> None:
    result = value_of_implementation_strategy_comparison(
        np.array([[[10.0, 10.5], [12.0, 13.0]]]),
        ["status-quo", "supported"],
        np.array([[1.0, 0.8], [1.0, 0.9]]),
        np.array([[1.0, 0.8], [1.0, 0.9]]),
        np.array([[1.0, 0.7], [1.0, 0.8]]),
        np.zeros((2, 2)),
        np.zeros((2, 2)),
        np.array([[0.0, 0.5], [0.0, 0.7]]),
    )
    assert result.method_maturity == "fixture-backed"
    assert result.adoption_uncertainty_matrix.shape == (2, 2)
    assert set(result.optimal_strategy_by_period) == {"0", "1"}


def test_implementation_strategy_comparison_rejects_invalid_uptake() -> None:
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        value_of_implementation_strategy_comparison(
            np.ones((1, 1, 1)),
            ["strategy"],
            np.array([[1.1]]),
            np.ones((1, 1)),
            np.ones((1, 1)),
            np.zeros((1, 1)),
            np.zeros((1, 1)),
            np.zeros((1, 1)),
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"net_benefits": np.ones((1, 1))}, "3D array"),
        ({"uptake": np.zeros((1, 2))}, "period x strategy"),
        ({"net_benefits": np.array([[[np.nan]]])}, "finite"),
        ({"discount_rate": -1.0}, "non-negative"),
    ],
)
def test_implementation_strategy_comparison_rejects_invalid_inputs(
    kwargs: dict[str, object], message: str
) -> None:
    base: dict[str, object] = {
        "net_benefits": np.ones((1, 1, 1)),
        "strategy_names": ["strategy"],
        "uptake": np.ones((1, 1)),
        "adherence": np.ones((1, 1)),
        "coverage": np.ones((1, 1)),
        "implementation_delays": np.zeros((1, 1)),
        "scale_up_costs": np.zeros((1, 1)),
        "population_impacts": np.zeros((1, 1)),
    }
    base.update(kwargs)
    with pytest.raises(ValueError, match=message):
        value_of_implementation_strategy_comparison(**base)  # type: ignore[arg-type]


def test_implementation_strategy_cli_reports_input_errors(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.json"
    invalid.write_text("[]", encoding="utf-8")
    result = runner.invoke(cli.app, ["calculate-implementation-strategy", str(invalid)])
    assert result.exit_code == 1
    assert "must be a JSON object" in result.stderr

    missing = runner.invoke(
        cli.app, ["calculate-implementation-strategy", str(tmp_path / "missing.json")]
    )
    assert missing.exit_code != 0


def test_implementation_strategy_comparison_rejects_duplicate_strategies() -> None:
    with pytest.raises(ValueError, match="unique"):
        value_of_implementation_strategy_comparison(
            np.ones((1, 2, 1)),
            ["x", "x"],
            np.ones((1, 2)),
            np.ones((1, 2)),
            np.ones((1, 2)),
            np.zeros((1, 2)),
            np.zeros((1, 2)),
            np.zeros((1, 2)),
        )
