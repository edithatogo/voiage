from voiage.methods.sensitivity import deterministic_sensitivity_analysis


def test_deterministic_sensitivity_orders_absolute_scenario_impact() -> None:
    result = deterministic_sensitivity_analysis(100.0, {"low": 90, "high": 115, "mid": 101})
    assert list(result.scenario_values) == ["high", "low", "mid"]
    assert result.deltas == {"high": 15.0, "low": -10.0, "mid": 1.0}


def test_deterministic_sensitivity_does_not_claim_probabilities() -> None:
    result = deterministic_sensitivity_analysis(0.0, {"a": -2, "b": 2})
    assert result.baseline == 0.0
    assert not hasattr(result, "probabilities")
