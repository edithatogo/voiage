"""Tests for regulatory and market-access VOI."""

from hypothesis import given, settings
from hypothesis import strategies as st
import numpy as np
import pytest

from voiage.analysis import DecisionAnalysis
from voiage.methods.regulatory_market_access import (
    value_of_regulatory_market_access,
)


def _inputs() -> dict[str, object]:
    return {
        "scenario_values": [10.0, 8.0, 6.0, 4.0],
        "approval_probabilities": [0.8, 0.5, 0.3, 0.2],
        "reimbursement_probabilities": [0.7, 0.5, 0.2, 0.1],
        "access_delays_months": [2.0, 4.0, 8.0, 12.0],
        "evidence_package_costs": [1.0, 1.0, 1.0, 1.0],
        "label_expansion_values": [3.0, 2.0, 1.0, 0.5],
        "price_threshold_values": [2.0, 1.0, 0.5, 0.25],
        "scenario_names": [
            "approved",
            "restricted_label",
            "coverage_evidence",
            "rejected",
        ],
        "approval_threshold": 0.5,
        "monthly_access_delay_cost": 0.1,
    }


def test_market_access_result_reports_regulatory_and_payer_value() -> None:
    result = value_of_regulatory_market_access(**_inputs())

    assert result.method_maturity == "fixture-backed"
    assert result.selected_scenario_indices.tolist() == [0]
    assert result.approval_probability == pytest.approx(0.8)
    assert result.reimbursement_probability == pytest.approx(0.7)
    assert result.regulatory_uncertainty > 0.0
    assert result.payer_uncertainty > 0.0
    assert result.expected_access_value > 0.0
    assert result.evidence_package_cost == pytest.approx(1.0)
    assert result.diagnostics["parity_status"] == "deferred"
    assert result.reporting["analysis_type"] == "value_of_regulatory_market_access"


def test_decision_analysis_wrapper_is_available() -> None:
    result = DecisionAnalysis(np.ones((2, 2))).value_of_regulatory_market_access(
        **_inputs()
    )
    assert result.method_maturity == "fixture-backed"


def test_market_access_inputs_are_validated() -> None:
    for overrides in (
        {"scenario_values": [1.0]},
        {"approval_probabilities": [0.1, 0.2, 0.3, 1.2]},
        {"reimbursement_probabilities": [0.1, 0.2, np.nan, 0.4]},
        {"approval_threshold": -0.1},
        {"access_delays_months": [1.0, 1.0, 1.0, -1.0]},
        {"evidence_package_costs": [1.0, 1.0, 1.0, -1.0]},
    ):
        payload = _inputs()
        payload.update(overrides)
        with pytest.raises(Exception):
            value_of_regulatory_market_access(**payload)


@given(
    st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=8,
    )
)
@settings(max_examples=20, deadline=None)
def test_lower_approval_threshold_cannot_select_fewer_scenarios(
    probabilities: list[float],
) -> None:
    common = {
        "scenario_values": [1.0] * len(probabilities),
        "approval_probabilities": probabilities,
        "reimbursement_probabilities": probabilities,
        "access_delays_months": [0.0] * len(probabilities),
        "evidence_package_costs": [0.0] * len(probabilities),
        "label_expansion_values": [0.0] * len(probabilities),
        "price_threshold_values": [0.0] * len(probabilities),
    }
    low = value_of_regulatory_market_access(approval_threshold=0.25, **common)
    high = value_of_regulatory_market_access(approval_threshold=0.75, **common)
    assert len(low.selected_scenario_indices) >= len(high.selected_scenario_indices)
