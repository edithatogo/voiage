"""Tests for AI-assisted evidence-triage VOI."""

from hypothesis import given, settings
from hypothesis import strategies as st
import numpy as np
import pytest

from voiage.analysis import DecisionAnalysis
from voiage.methods.ai_assisted_evidence_triage import (
    value_of_ai_assisted_evidence_triage,
)


def _inputs() -> dict[str, object]:
    return {
        "relevance_labels": [1, 1, 0, 1, 0, 1],
        "triage_scores": [0.95, 0.4, 0.8, 0.7, 0.2, 0.6],
        "decision_impacts": [10.0, 8.0, 4.0, 6.0, 3.0, 5.0],
        "reviewer_time_minutes": [10.0, 8.0, 6.0, 7.0, 4.0, 5.0],
        "extraction_error_rates": [0.01, 0.10, 0.05, 0.02, 0.20, 0.03],
        "triage_threshold": 0.5,
        "audit_sample_rate": 0.5,
        "human_override_rate": 0.25,
        "model_drift": 0.1,
        "automation_cost": 2.0,
        "audit_cost_per_item": 0.5,
        "reviewer_cost_per_minute": 0.1,
    }


def test_triage_result_reports_quality_economics_and_audit_value() -> None:
    result = value_of_ai_assisted_evidence_triage(**_inputs())

    assert result.method_maturity == "fixture-backed"
    assert result.selected_item_indices.tolist() == [0, 2, 3, 5]
    assert result.sensitivity == pytest.approx(0.75)
    assert result.specificity == pytest.approx(0.5)
    assert 0.0 <= result.false_exclusion_risk <= 1.0
    assert result.false_inclusion_burden > 0.0
    assert result.reviewer_time_saved > 0.0
    assert result.audit_value > 0.0
    assert result.model_drift == pytest.approx(0.1)
    assert result.diagnostics["human_in_the_loop"] is True
    assert result.diagnostics["parity_status"] == "deferred"
    assert result.reporting["analysis_type"] == "value_of_ai_assisted_evidence_triage"


def test_zero_audit_and_override_preserve_raw_error_metrics() -> None:
    payload = _inputs()
    payload.update(
        audit_sample_rate=0.0,
        human_override_rate=0.0,
        model_drift=0.0,
    )
    result = value_of_ai_assisted_evidence_triage(**payload)

    assert result.audit_value == pytest.approx(0.0)
    assert result.effective_false_exclusion_impact == pytest.approx(8.0)
    assert result.effective_false_inclusion_impact == pytest.approx(4.0)
    assert result.diagnostics["audit_items"] == 0


def test_decision_analysis_wrapper_is_available() -> None:
    result = DecisionAnalysis(np.ones((2, 2))).value_of_ai_assisted_evidence_triage(
        **_inputs()
    )
    assert result.method_maturity == "fixture-backed"


def test_triage_inputs_are_validated() -> None:
    for overrides in (
        {"relevance_labels": [1, 0]},
        {"triage_scores": [0.2, np.nan, 0.4, 0.5, 0.6, 0.7]},
        {"triage_threshold": 1.5},
        {"audit_sample_rate": -0.1},
        {"human_override_rate": 1.1},
        {"model_drift": -0.1},
        {"reviewer_time_minutes": [1, 1, 1, 1, 1, -1]},
    ):
        payload = _inputs()
        payload.update(overrides)
        with pytest.raises(Exception):
            value_of_ai_assisted_evidence_triage(**payload)


@given(
    st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=8,
    )
)
@settings(max_examples=20, deadline=None)
def test_triage_threshold_property(scores: list[float]) -> None:
    """Lower thresholds cannot select fewer evidence items."""
    labels = [index % 2 for index in range(len(scores))]
    common = {
        "relevance_labels": labels,
        "decision_impacts": [1.0] * len(scores),
        "reviewer_time_minutes": [1.0] * len(scores),
        "extraction_error_rates": [0.0] * len(scores),
        "audit_sample_rate": 0.0,
    }
    low = value_of_ai_assisted_evidence_triage(
        triage_scores=scores, triage_threshold=0.25, **common
    )
    high = value_of_ai_assisted_evidence_triage(
        triage_scores=scores, triage_threshold=0.75, **common
    )
    assert len(low.selected_item_indices) >= len(high.selected_item_indices)
    assert 0.0 <= low.sensitivity <= 1.0
    assert 0.0 <= high.specificity <= 1.0
