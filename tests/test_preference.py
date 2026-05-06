"""Tests for preference heterogeneity and individualized care analysis."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from voiage import (
    PreferenceHeterogeneityResult as TopLevelPreferenceHeterogeneityResult,
)
from voiage import PreferenceProfile as TopLevelPreferenceProfile
from voiage import PreferenceProfileSet as TopLevelPreferenceProfileSet
from voiage import (
    preference_optimal_strategies as top_level_preference_optimal_strategies,
)
from voiage import value_of_preference as top_level_value_of_preference
from voiage import (
    value_of_preference_information as top_level_value_of_preference_information,
)
from voiage.analysis import DecisionAnalysis
from voiage.exceptions import InputError
from voiage.methods.preference import (
    PreferenceHeterogeneityResult,
    PreferenceProfile,
    PreferenceProfileSet,
    preference_optimal_strategies,
    value_of_preference,
    value_of_preference_information,
)
from voiage.schema import ValueArray


def test_value_of_preference_compares_conflicting_preference_profiles() -> None:
    """Different preference profiles should expose regret and individualized care."""
    net_benefits = np.array(
        [
            [[10.0, 7.0], [8.0, 11.0], [5.0, 9.0]],
            [[10.0, 7.0], [8.0, 11.0], [5.0, 9.0]],
        ]
    )
    profiles = PreferenceProfileSet(
        [
            PreferenceProfile(id="access_first", label="Access first", weight=0.25),
            PreferenceProfile(id="outcomes_first", label="Outcomes first", weight=0.75),
        ]
    )

    result = value_of_preference(
        net_benefits,
        preference_profiles=profiles,
        strategy_names=["A", "B", "C"],
        preference_profile_weights={"access_first": 0.25, "outcomes_first": 0.75},
        reference_preference_profile="access_first",
        analysis_id="preference-screening-001",
        decision_problem_id="screening-program-001",
    )

    assert isinstance(result, PreferenceHeterogeneityResult)
    assert TopLevelPreferenceProfile is PreferenceProfile
    assert TopLevelPreferenceProfileSet is PreferenceProfileSet
    assert TopLevelPreferenceHeterogeneityResult is PreferenceHeterogeneityResult
    assert top_level_value_of_preference is value_of_preference
    assert top_level_value_of_preference_information is value_of_preference_information
    assert top_level_preference_optimal_strategies is preference_optimal_strategies
    assert result.preference_profile_ids == ["access_first", "outcomes_first"]
    assert result.preference_profile_labels == ["Access first", "Outcomes first"]
    assert result.strategy_names == ["A", "B", "C"]
    np.testing.assert_allclose(
        result.expected_net_benefits,
        np.array([[10.0, 8.0, 5.0], [7.0, 11.0, 9.0]]),
    )
    assert result.optimal_strategy_names == ["A", "B"]
    assert result.optimal_strategy_by_preference_profile == {
        "access_first": "A",
        "outcomes_first": "B",
    }
    np.testing.assert_allclose(
        result.regret_matrix,
        np.array([[0.0, 2.0], [4.0, 0.0]]),
    )
    np.testing.assert_allclose(result.switching_values, np.array([0.0, 4.0]))
    assert result.value == pytest.approx(3.0)
    assert result.consensus_strategy == "B"
    assert result.consensus_strategy_name == "B"
    assert result.consensus_weighted_expected_net_benefit == pytest.approx(10.25)
    assert result.individualized_care_value == pytest.approx(0.5)
    assert result.individualized_care_weighted_expected_net_benefit == pytest.approx(
        10.75
    )
    assert result.robust_strategy == "B"
    assert result.pareto_strategies == ["A", "B"]
    assert result.method_maturity == "fixture-backed"
    assert result.analysis_id == "preference-screening-001"
    assert result.decision_problem_id == "screening-program-001"
    assert result.reference_preference_profile == "access_first"
    assert result.reporting["reporting_standard"] == "CHEERS-VOI"
    assert result.reporting["analysis_type"] == "value_of_preference_information"
    assert result.reporting["preference_profile_ids"] == [
        "access_first",
        "outcomes_first",
    ]

    assert preference_optimal_strategies(result) == {
        "access_first": "A",
        "outcomes_first": "B",
    }


def test_value_of_preference_serializes_analysis_identifiers_to_dict() -> None:
    """Preference results should carry analysis identifiers through to_dict."""
    net_benefits = np.array(
        [
            [[10.0, 7.0], [8.0, 11.0], [5.0, 9.0]],
            [[10.0, 7.0], [8.0, 11.0], [5.0, 9.0]],
        ]
    )

    result = value_of_preference(
        net_benefits,
        preference_profile_names=["access_first", "outcomes_first"],
        strategy_names=["A", "B", "C"],
        analysis_id="preference-screening-001",
        decision_problem_id="screening-program-001",
    )

    payload = result.to_dict()

    assert result.analysis_id == "preference-screening-001"
    assert result.decision_problem_id == "screening-program-001"
    assert payload["analysis_id"] == "preference-screening-001"
    assert payload["decision_problem_id"] == "screening-program-001"
    assert payload["analysis_type"] == "value_of_preference_information"
    assert payload["method_maturity"] == "fixture-backed"
    assert payload["reporting"]["analysis_id"] == "preference-screening-001"
    assert payload["reporting"]["decision_problem_id"] == "screening-program-001"
    assert payload["reporting"]["reporting_standard"] == "CHEERS-VOI"
    assert payload["preference_profile_ids"] == ["access_first", "outcomes_first"]
    assert payload["switching_values"] == {
        "access_first": 0.0,
        "outcomes_first": 4.0,
    }
    assert payload["consensus_strategy"] == "B"
    assert payload["robust_strategy"] == "B"


def test_value_of_preference_accepts_value_array_with_perspective_dimension() -> None:
    """The preference surface should use ValueArray profile coordinates."""
    value_array = ValueArray.from_numpy_perspectives(
        np.array(
            [
                [[1.0, 6.0], [3.0, 2.0]],
                [[2.0, 8.0], [4.0, 3.0]],
            ]
        ),
        strategy_names=["usual", "new"],
        perspective_names=["payer", "patient"],
    )

    result = value_of_preference(value_array)

    assert result.strategy_names == ["usual", "new"]
    assert result.preference_profile_ids == ["payer", "patient"]
    assert result.preference_profile_labels == ["payer", "patient"]
    assert result.optimal_strategy_names == ["new", "usual"]
    assert result.reference_preference_profile == "payer"


def test_decision_analysis_wraps_value_of_preference() -> None:
    """DecisionAnalysis should expose the preference-comparison method."""
    analysis = DecisionAnalysis(
        np.array(
            [
                [[10.0, 7.0], [8.0, 11.0]],
                [[10.0, 7.0], [8.0, 11.0]],
            ]
        )
    )

    result = analysis.value_of_preference(
        preference_profile_names=["payer", "societal"],
        strategy_names=["A", "B"],
    )

    assert result.optimal_strategy_names == ["A", "B"]
    assert result.consensus_strategy_name == "B"


def test_value_of_preference_rejects_invalid_inputs() -> None:
    """Invalid dimensions, metadata, and weights should fail early."""
    with pytest.raises(InputError, match="3D"):
        value_of_preference(np.ones((2, 2)))

    with pytest.raises(InputError, match="preference profiles"):
        value_of_preference(np.ones((2, 2, 2)), preference_profile_names=["only-one"])

    with pytest.raises(InputError, match="unique"):
        PreferenceProfileSet(
            [PreferenceProfile(id="payer"), PreferenceProfile(id="payer")]
        )

    with pytest.raises(InputError, match="weight"):
        PreferenceProfile(id="payer", weight=np.inf)

    with pytest.raises(InputError, match="population"):
        PreferenceProfile(id="payer", population=0.0)

    with pytest.raises(InputError, match="weights"):
        value_of_preference(
            np.ones((2, 2, 2)),
            preference_profile_names=["payer", "societal"],
            preference_profile_weights=[1.0],
        )

    with pytest.raises(InputError, match="reference_preference_profile"):
        value_of_preference(
            np.ones((2, 2, 2)),
            preference_profile_names=["payer", "societal"],
            reference_preference_profile="missing",
        )

    with pytest.raises(InputError, match="ValueArray"):
        value_of_preference(cast("ValueArray", "not an array"))


def test_value_of_preference_information_alias_matches_primary_api() -> None:
    """The contract wording alias should return the same result object shape."""
    values = np.array(
        [
            [[10.0, 7.0], [8.0, 11.0]],
            [[10.0, 7.0], [8.0, 11.0]],
        ]
    )
    result = value_of_preference_information(
        values,
        preference_profile_names=["payer", "societal"],
        strategy_names=["A", "B"],
    )

    assert isinstance(result, PreferenceHeterogeneityResult)
    assert result.preference_profile_ids == ["payer", "societal"]
