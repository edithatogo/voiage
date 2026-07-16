"""Runtime tests for computational refinement VOI."""

import numpy as np
import pytest

from voiage.methods.computational import value_of_computational_refinement


def test_computational_refinement_returns_budget_specific_decisions() -> None:
    result = value_of_computational_refinement(
        np.array([[[10.0, 9.7], [11.4, 12.1], [11.0, 11.8]]]),
        ["baseline", "enhanced"],
        ["status-quo", "refine", "fast"],
        np.zeros((2, 3)),
        np.zeros((2, 3)),
        np.ones((2, 3)),
    )
    assert result.method_maturity == "fixture-backed"
    assert result.expected_net_benefits.shape == (2, 3)
    assert result.optimal_strategy_by_compute_budget == {
        "baseline": "refine",
        "enhanced": "refine",
    }


def test_computational_refinement_rejects_invalid_weights() -> None:
    with pytest.raises(ValueError, match="refinement weights"):
        value_of_computational_refinement(
            np.ones((1, 1, 1)),
            ["budget"],
            ["strategy"],
            np.zeros((1, 1)),
            np.zeros((1, 1)),
            np.ones((1, 1)) * 2,
        )


@pytest.mark.parametrize("case", ["shape", "matrix", "duplicate", "finite", "cost"])
def test_computational_refinement_rejects_boundary_inputs(case: str) -> None:
    values = np.ones((1, 1, 1))
    budgets = ["budget"]
    strategies = ["strategy"]
    matrix = np.zeros((1, 1))
    if case == "shape":
        values = np.ones((1, 2, 1))
    elif case == "matrix":
        matrix = np.zeros((2, 1))
    elif case == "duplicate":
        budgets = ["x", "x"]
        values = np.ones((1, 1, 2))
        matrix = np.zeros((2, 1))
    elif case == "finite":
        values = np.array([[[np.nan]]])
    elif case == "cost":
        matrix = np.full((1, 1), -1.0)
    with pytest.raises(ValueError):
        value_of_computational_refinement(
            values, budgets, strategies, matrix, matrix, np.ones_like(matrix)
        )
