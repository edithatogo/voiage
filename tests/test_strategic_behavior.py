"""Tests for strategic behavior and game-theoretic VOI."""

from hypothesis import given, settings
from hypothesis import strategies as st
import numpy as np
import pytest

from voiage.analysis import DecisionAnalysis
from voiage.methods.strategic_behavior import value_of_strategic_behavior


def _inputs() -> dict[str, object]:
    return {
        "scenario_values": [10.0, 8.0, 6.0, 4.0],
        "equilibrium_probabilities": [0.8, 0.6, 0.3, 0.2],
        "incentive_response_values": [2.0, 1.0, 0.5, 0.2],
        "disclosure_values": [1.0, 2.0, 0.5, 0.2],
        "bargaining_values": [1.5, 1.0, 0.2, 0.1],
        "adversarial_risks": [0.1, 0.2, 0.4, 0.8],
        "response_sensitivities": [0.2, 0.5, 0.8, 0.9],
        "strategic_regrets": [0.5, 0.4, 0.2, 0.1],
        "equilibrium_threshold": 0.5,
    }


def test_strategic_result_reports_equilibrium_and_response_value() -> None:
    result = value_of_strategic_behavior(**_inputs())
    assert result.method_maturity == "fixture-backed"
    assert result.selected_scenario_indices.tolist() == [0, 1]
    assert result.equilibrium_probability == pytest.approx(0.7)
    assert result.response_sensitivity == pytest.approx(0.35)
    assert result.disclosure_value == pytest.approx(3.0)
    assert result.incentive_value == pytest.approx(3.0)
    assert result.diagnostics["parity_status"] == "deferred"
    assert result.reporting["analysis_type"] == "value_of_strategic_behavior"


def test_decision_analysis_wrapper_is_available() -> None:
    result = DecisionAnalysis(np.ones((2, 2))).value_of_strategic_behavior(**_inputs())
    assert result.method_maturity == "fixture-backed"


def test_strategic_inputs_are_validated() -> None:
    for overrides in (
        {"scenario_values": [1.0]},
        {"equilibrium_probabilities": [0.1, 0.2, 1.2, 0.4]},
        {"adversarial_risks": [0.1, 0.2, np.nan, 0.4]},
        {"equilibrium_threshold": -0.1},
        {"strategic_regrets": [0.5, -0.1, 0.2, 0.1]},
    ):
        payload = _inputs()
        payload.update(overrides)
        with pytest.raises(Exception):
            value_of_strategic_behavior(**payload)


@given(
    st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=8,
    )
)
@settings(max_examples=20, deadline=None)
def test_lower_equilibrium_threshold_cannot_select_fewer_scenarios(
    probability: list[float],
) -> None:
    common = {
        "scenario_values": [1.0] * len(probability),
        "equilibrium_probabilities": probability,
        "incentive_response_values": [0.0] * len(probability),
        "disclosure_values": [0.0] * len(probability),
        "bargaining_values": [0.0] * len(probability),
        "adversarial_risks": [0.0] * len(probability),
        "response_sensitivities": probability,
        "strategic_regrets": [0.0] * len(probability),
    }
    low = value_of_strategic_behavior(equilibrium_threshold=0.25, **common)
    high = value_of_strategic_behavior(equilibrium_threshold=0.75, **common)
    assert len(low.selected_scenario_indices) >= len(high.selected_scenario_indices)
