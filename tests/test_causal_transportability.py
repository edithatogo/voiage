"""Runtime tests for causal transportability VOI."""

import numpy as np
import pytest

from voiage.methods.causal_transportability import value_of_causal_transportability


def test_causal_transportability_returns_target_specific_decisions() -> None:
    result = value_of_causal_transportability(
        np.array([[[10.0, 12.0], [9.0, 11.0]], [[11.0, 13.0], [8.0, 12.0]]]),
        ["trial", "registry"],
        ["urban", "rural"],
        ["usual-care", "adapted"],
        np.array([[1.0, 0.6], [0.8, 1.0]]),
        np.array([[0.0, 1.0], [0.5, 0.2]]),
    )
    assert result.method_maturity == "fixture-backed"
    assert result.expected_net_benefits.shape == (2, 2)
    assert result.optimal_strategy_by_target_population["urban"] == "adapted"
    assert result.robust_strategy in result.strategy_names
    assert result.value >= 0


def test_causal_transportability_rejects_missing_target_weight() -> None:
    with pytest.raises(ValueError, match="positive transport weight"):
        value_of_causal_transportability(
            np.ones((1, 1, 1)), ["source"], ["target"], ["strategy"],
            np.zeros((1, 1)), np.zeros((1, 1)),
        )
