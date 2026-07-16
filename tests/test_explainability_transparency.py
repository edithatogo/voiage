"""Tests for explainability and transparency VOI."""

from hypothesis import given, settings
from hypothesis import strategies as st
import numpy as np
import pytest

from voiage.analysis import DecisionAnalysis
from voiage.methods.explainability_transparency import (
    value_of_explainability_transparency,
)


def _inputs() -> dict[str, object]:
    return {
        "predictive_utilities": [8.0, 7.0, 6.0, 5.0],
        "explanation_quality": [0.9, 0.4, 0.8, 0.2],
        "transparency_evidence": [0.8, 0.3, 0.7, 0.1],
        "trust_scores": [0.9, 0.4, 0.8, 0.2],
        "governance_impacts": [5.0, 3.0, 4.0, 2.0],
        "audit_costs": [1.0, 1.0, 1.0, 1.0],
        "adoption_threshold": 0.5,
        "transparency_weight": 0.5,
    }


def test_explainability_result_reports_adoption_trust_and_governance() -> None:
    result = value_of_explainability_transparency(**_inputs())

    assert result.method_maturity == "fixture-backed"
    assert result.adopted_model_indices.tolist() == [0, 2]
    assert result.adoption_probability == pytest.approx(0.5)
    assert result.transparency_value > 0.0
    assert result.governance_value > 0.0
    assert result.audit_cost == pytest.approx(2.0)
    assert result.diagnostics["parity_status"] == "deferred"
    assert result.reporting["analysis_type"] == "value_of_explainability_transparency"


def test_decision_analysis_wrapper_is_available() -> None:
    result = DecisionAnalysis(np.ones((2, 2))).value_of_explainability_transparency(
        **_inputs()
    )
    assert result.method_maturity == "fixture-backed"


def test_explainability_inputs_are_validated() -> None:
    for overrides in (
        {"predictive_utilities": [1.0]},
        {"explanation_quality": [0.1, 0.2, 0.3, 1.2]},
        {"transparency_evidence": [0.1, 0.2, np.nan, 0.4]},
        {"adoption_threshold": -0.1},
        {"transparency_weight": 1.1},
        {"audit_costs": [1.0, 1.0, 1.0, -1.0]},
    ):
        payload = _inputs()
        payload.update(overrides)
        with pytest.raises(Exception):
            value_of_explainability_transparency(**payload)


@given(
    st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=8,
    )
)
@settings(max_examples=20, deadline=None)
def test_lower_adoption_threshold_cannot_adopt_fewer_models(
    evidence: list[float],
) -> None:
    common = {
        "predictive_utilities": [1.0] * len(evidence),
        "explanation_quality": evidence,
        "transparency_evidence": evidence,
        "trust_scores": evidence,
        "governance_impacts": [1.0] * len(evidence),
        "audit_costs": [0.0] * len(evidence),
    }
    low = value_of_explainability_transparency(adoption_threshold=0.25, **common)
    high = value_of_explainability_transparency(adoption_threshold=0.75, **common)
    assert len(low.adopted_model_indices) >= len(high.adopted_model_indices)
