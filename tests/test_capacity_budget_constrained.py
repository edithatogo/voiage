"""Tests for capacity and budget-constrained VOI."""

import numpy as np
import pytest

from voiage.analysis import DecisionAnalysis
from voiage.methods.capacity_budget_constrained import (
    value_of_capacity_budget_constrained,
)


def _inputs() -> dict[str, object]:
    return {
        "scenario_values": [[10.0, 8.0, 4.0], [6.0, 11.0, 9.0], [12.0, 7.0, 10.0]],
        "strategy_costs": [2.0, 5.0, 8.0],
        "strategy_capacity": [1.0, 2.0, 4.0],
        "budget": 5.0,
        "capacity": 2.0,
        "strategy_names": ["small", "balanced", "large"],
    }


def test_constrained_result_reports_information_value() -> None:
    result = value_of_capacity_budget_constrained(**_inputs())
    assert result.method_maturity == "fixture-backed"
    assert result.selected_strategy == "small"
    assert result.value == pytest.approx(5 / 3)
    assert result.scenario_optimal_strategies == ["small", "balanced", "small"]
    assert result.diagnostics["parity_status"] == "deferred"


def test_constraints_and_names_are_validated() -> None:
    with pytest.raises(Exception):
        value_of_capacity_budget_constrained(
            np.ones((2, 2)),
            strategy_costs=[1.0],
            strategy_capacity=[1.0, 1.0],
            budget=1,
            capacity=1,
        )
    with pytest.raises(Exception):
        value_of_capacity_budget_constrained(
            np.ones((2, 2)),
            strategy_costs=[2.0, 2.0],
            strategy_capacity=[1.0, 1.0],
            budget=1,
            capacity=1,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "scenario_values": [[1.0, float("nan")]],
            "strategy_costs": [1, 1],
            "strategy_capacity": [1, 1],
            "budget": 1,
            "capacity": 1,
        },
        {
            "scenario_values": [[1.0, 2.0]],
            "strategy_costs": [-1, 1],
            "strategy_capacity": [1, 1],
            "budget": 1,
            "capacity": 1,
        },
        {
            "scenario_values": [[1.0, 2.0]],
            "strategy_costs": [1, 1],
            "strategy_capacity": [1, 1],
            "budget": -1,
            "capacity": 1,
        },
        {
            "scenario_values": [[1.0, 2.0]],
            "strategy_costs": [1, 1],
            "strategy_capacity": [1, 1],
            "budget": 1,
            "capacity": 1,
            "information_cost": -1,
        },
        {
            "scenario_values": [[1.0, 2.0]],
            "strategy_costs": [1, 1],
            "strategy_capacity": [1, 1],
            "budget": 1,
            "capacity": 1,
            "strategy_names": ["one"],
        },
    ],
)
def test_invalid_constrained_inputs_are_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(Exception):
        value_of_capacity_budget_constrained(**kwargs)


def test_decision_analysis_wrapper_exposes_method() -> None:
    result = DecisionAnalysis(np.ones((2, 2))).value_of_capacity_budget_constrained(
        **_inputs()
    )
    assert result.selected_strategy == "small"
