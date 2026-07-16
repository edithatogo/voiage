"""Tests for replication and reproducibility VOI."""

from hypothesis import given, settings
from hypothesis import strategies as st
import numpy as np
import pytest

from voiage.analysis import DecisionAnalysis
from voiage.methods.replication_reproducibility import (
    value_of_replication_reproducibility,
)


def _inputs() -> dict[str, object]:
    return {
        "evidence_values": [10.0, 8.0, 6.0, 4.0],
        "replication_probabilities": [0.9, 0.5, 0.7, 0.2],
        "reproducibility_failure_risks": [0.1, 0.4, 0.2, 0.8],
        "audit_costs": [1.0, 1.0, 1.0, 1.0],
        "reanalysis_values": [2.0, 1.0, 1.5, 0.5],
        "credibility_adjustments": [1.2, 1.0, 1.1, 0.8],
        "evidence_downgrades": [0.1, 0.3, 0.2, 0.6],
        "replication_threshold": 0.5,
    }


def test_replication_result_reports_credibility_and_audit_value() -> None:
    result = value_of_replication_reproducibility(**_inputs())

    assert result.method_maturity == "fixture-backed"
    assert result.selected_replication_indices.tolist() == [0, 2]
    assert result.replication_probability == pytest.approx(0.5)
    assert result.replication_value > 0.0
    assert result.audit_burden == pytest.approx(2.0)
    assert result.credibility_impact > 0.0
    assert result.diagnostics["parity_status"] == "deferred"
    assert result.reporting["analysis_type"] == "value_of_replication_reproducibility"


def test_decision_analysis_wrapper_is_available() -> None:
    result = DecisionAnalysis(np.ones((2, 2))).value_of_replication_reproducibility(
        **_inputs()
    )
    assert result.method_maturity == "fixture-backed"


def test_replication_inputs_are_validated() -> None:
    for overrides in (
        {"evidence_values": [1.0]},
        {"replication_probabilities": [0.1, 0.2, 1.2, 0.4]},
        {"reproducibility_failure_risks": [0.1, 0.2, np.nan, 0.4]},
        {"replication_threshold": -0.1},
        {"audit_costs": [1.0, 1.0, 1.0, -1.0]},
        {"credibility_adjustments": [1.0, 1.0, -0.1, 1.0]},
    ):
        payload = _inputs()
        payload.update(overrides)
        with pytest.raises(Exception):
            value_of_replication_reproducibility(**payload)


@given(
    st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=8,
    )
)
@settings(max_examples=20, deadline=None)
def test_lower_replication_threshold_cannot_select_fewer_records(
    probability: list[float],
) -> None:
    common = {
        "evidence_values": [1.0] * len(probability),
        "replication_probabilities": probability,
        "reproducibility_failure_risks": [0.0] * len(probability),
        "audit_costs": [0.0] * len(probability),
        "reanalysis_values": [0.0] * len(probability),
        "credibility_adjustments": [1.0] * len(probability),
        "evidence_downgrades": [0.0] * len(probability),
    }
    low = value_of_replication_reproducibility(replication_threshold=0.25, **common)
    high = value_of_replication_reproducibility(replication_threshold=0.75, **common)
    assert len(low.selected_replication_indices) >= len(
        high.selected_replication_indices
    )
