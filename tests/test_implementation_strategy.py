"""Runtime tests for implementation-strategy comparison VOI."""

import numpy as np
import pytest

from voiage.methods.implementation_strategy import (
    value_of_implementation_strategy_comparison,
)


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
