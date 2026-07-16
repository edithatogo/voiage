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
