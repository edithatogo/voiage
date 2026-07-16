"""Tests for interoperability and standardization VOI."""

from hypothesis import given, settings
from hypothesis import strategies as st
import numpy as np
import pytest

from voiage.analysis import DecisionAnalysis
from voiage.methods.interoperability_standardization import (
    value_of_interoperability_standardization,
)


def _inputs() -> dict[str, object]:
    return {
        "evidence_utilities": [10.0, 8.0, 6.0, 5.0],
        "schema_compatibility": [0.9, 0.4, 0.8, 0.2],
        "semantic_alignment": [0.8, 0.3, 0.7, 0.1],
        "data_usability": [0.9, 0.5, 0.8, 0.2],
        "transformation_error_rates": [0.05, 0.2, 0.1, 0.3],
        "standardization_costs": [1.0, 1.0, 1.0, 1.0],
        "reuse_probabilities": [0.8, 0.4, 0.7, 0.2],
        "harmonization_threshold": 0.5,
    }


def test_interoperability_result_reports_reuse_and_harmonization_value() -> None:
    result = value_of_interoperability_standardization(**_inputs())

    assert result.method_maturity == "fixture-backed"
    assert result.reusable_evidence_indices.tolist() == [0, 2]
    assert result.reuse_probability == pytest.approx(0.5)
    assert result.standardization_value > 0.0
    assert result.reduced_transformation_error > 0.0
    assert result.data_usability_value > 0.0
    assert result.standardization_cost == pytest.approx(2.0)
    assert result.diagnostics["parity_status"] == "deferred"
    assert (
        result.reporting["analysis_type"] == "value_of_interoperability_standardization"
    )


def test_decision_analysis_wrapper_is_available() -> None:
    result = DecisionAnalysis(
        np.ones((2, 2))
    ).value_of_interoperability_standardization(**_inputs())
    assert result.method_maturity == "fixture-backed"


def test_interoperability_inputs_are_validated() -> None:
    for overrides in (
        {"evidence_utilities": [1.0]},
        {"schema_compatibility": [0.1, 0.2, 0.3, 1.2]},
        {"transformation_error_rates": [0.1, 0.2, np.nan, 0.4]},
        {"harmonization_threshold": -0.1},
        {"standardization_costs": [1.0, 1.0, 1.0, -1.0]},
        {"reuse_probabilities": [0.1, 0.2, 0.3, 1.1]},
    ):
        payload = _inputs()
        payload.update(overrides)
        with pytest.raises(Exception):
            value_of_interoperability_standardization(**payload)


@given(
    st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=8,
    )
)
@settings(max_examples=20, deadline=None)
def test_lower_harmonization_threshold_cannot_reuse_fewer_records(
    compatibility: list[float],
) -> None:
    common = {
        "evidence_utilities": [1.0] * len(compatibility),
        "schema_compatibility": compatibility,
        "semantic_alignment": compatibility,
        "data_usability": compatibility,
        "transformation_error_rates": [0.0] * len(compatibility),
        "standardization_costs": [0.0] * len(compatibility),
        "reuse_probabilities": compatibility,
    }
    low = value_of_interoperability_standardization(
        harmonization_threshold=0.25, **common
    )
    high = value_of_interoperability_standardization(
        harmonization_threshold=0.75, **common
    )
    assert len(low.reusable_evidence_indices) >= len(high.reusable_evidence_indices)
