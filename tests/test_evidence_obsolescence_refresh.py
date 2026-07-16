"""Tests for evidence obsolescence and refresh VOI."""

from hypothesis import given, settings
from hypothesis import strategies as st
import numpy as np
import pytest

from voiage.analysis import DecisionAnalysis
from voiage.methods.evidence_obsolescence_refresh import (
    value_of_evidence_obsolescence_refresh,
)


def _inputs() -> dict[str, object]:
    return {
        "evidence_values": [10.0, 8.0, 12.0, 6.0, 4.0],
        "evidence_age_months": [2.0, 18.0, 24.0, 15.0, 3.0],
        "half_lives_months": [36.0] * 5,
        "obsolescence_risks": [0.1, 0.8, 0.9, 0.7, 0.2],
        "refresh_costs": [1.0] * 5,
        "living_review_values": [0.5, 3.0, 2.0, 1.0, 0.2],
        "model_refresh_values": [0.2, 2.0, 3.0, 1.0, 0.1],
        "drift_rates": [0.1, 0.7, 0.9, 0.6, 0.2],
        "refresh_threshold": 0.5,
        "refresh_cadence_months": 12.0,
    }


def test_refresh_result_reports_obsolescence_and_cadence_value() -> None:
    result = value_of_evidence_obsolescence_refresh(**_inputs())
    assert result.method_maturity == "fixture-backed"
    assert result.selected_refresh_indices.tolist() == [1, 2]
    assert result.evidence_age_months == pytest.approx(12.4)
    assert result.refresh_burden == pytest.approx(2.0)
    assert result.living_review_value == pytest.approx(5.0)
    assert result.model_refresh_value == pytest.approx(5.0)
    assert result.diagnostics["parity_status"] == "deferred"
    assert result.reporting["analysis_type"] == "value_of_evidence_obsolescence_refresh"


def test_decision_analysis_wrapper_is_available() -> None:
    result = DecisionAnalysis(np.ones((2, 2))).value_of_evidence_obsolescence_refresh(
        **_inputs()
    )
    assert result.method_maturity == "fixture-backed"


def test_refresh_inputs_are_validated() -> None:
    for overrides in (
        {"evidence_values": [1.0]},
        {"half_lives_months": [1.0, 0.0, 1.0, 1.0, 1.0]},
        {"obsolescence_risks": [0.1, 0.2, np.nan, 0.4, 0.2]},
        {"refresh_threshold": -0.1},
        {"refresh_costs": [1.0, -1.0, 1.0, 1.0, 1.0]},
        {"refresh_cadence_months": 0.0},
    ):
        payload = _inputs()
        payload.update(overrides)
        with pytest.raises(Exception):
            value_of_evidence_obsolescence_refresh(**payload)


@given(
    st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=8,
    )
)
@settings(max_examples=20, deadline=None)
def test_lower_refresh_threshold_cannot_select_fewer_records(
    risk: list[float],
) -> None:
    common = {
        "evidence_values": [1.0] * len(risk),
        "evidence_age_months": [1.0] * len(risk),
        "half_lives_months": [12.0] * len(risk),
        "obsolescence_risks": risk,
        "refresh_costs": [0.0] * len(risk),
        "living_review_values": [0.0] * len(risk),
        "model_refresh_values": [0.0] * len(risk),
        "drift_rates": [1.0] * len(risk),
    }
    low = value_of_evidence_obsolescence_refresh(refresh_threshold=0.25, **common)
    high = value_of_evidence_obsolescence_refresh(refresh_threshold=0.75, **common)
    assert len(low.selected_refresh_indices) >= len(high.selected_refresh_indices)
