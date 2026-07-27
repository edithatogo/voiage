import numpy as np

from voiage.methods.dynamic_real_options import value_of_flexibility


def test_value_of_flexibility_compares_fixed_and_adaptive_policy() -> None:
    result = value_of_flexibility(np.array([[10.0, 0.0], [0.0, 10.0]]))
    assert result.constrained_value == 5.0
    assert result.flexible_value == 10.0
    assert result.value == 5.0
    assert result.scenario_optimal_strategy_indices.tolist() == [0, 1]
