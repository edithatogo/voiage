import numpy as np

from voiage.methods.risk_sensitive import value_of_risk_sensitive_information


def test_risk_sensitive_voi_returns_tail_adjusted_value() -> None:
    result = value_of_risk_sensitive_information(np.array([[10.0, 0.0], [0.0, 10.0]]), risk_aversion=0.5)
    assert result.value >= 0.0
    assert result.scenario_optimal_strategy_indices.shape == (2,)
