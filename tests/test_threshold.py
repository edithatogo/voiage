"""Tests for threshold, tipping-point, and robust VOI analysis."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from voiage import ThresholdProfile, ThresholdProfileSet, value_of_threshold_information
from voiage.analysis import DecisionAnalysis
from voiage.exceptions import InputError
from voiage.methods.threshold import ThresholdResult, value_of_threshold


def _threshold_fixture_root() -> Path:
    return Path(__file__).resolve().parents[1] / "specs" / "frontier" / "threshold" / "v1" / "fixtures"


def test_value_of_threshold_information_compares_threshold_profiles() -> None:
    """Threshold VOI should expose crossing and reversal summaries."""
    values = np.array(
        [
            [[10.0, 7.0], [8.0, 11.0], [5.0, 9.0]],
            [[10.0, 7.0], [8.0, 11.0], [5.0, 9.0]],
        ]
    )
    profiles = ThresholdProfileSet(
        [
            ThresholdProfile(id="wtp_reversal", label="WTP reversal", weight=0.5),
            ThresholdProfile(
                id="policy_constraint", label="Policy constraint", weight=0.5
            ),
        ]
    )

    result = value_of_threshold_information(
        values,
        threshold_profiles=profiles,
        strategy_names=["A", "B", "C"],
        reference_threshold_profile="wtp_reversal",
    )

    assert isinstance(result, ThresholdResult)
    assert result.analysis_id == "threshold-screening-001"
    assert result.decision_problem_id == "screening-program-001"
    assert result.threshold_profile_ids == ["wtp_reversal", "policy_constraint"]
    assert result.threshold_profile_labels == ["WTP reversal", "Policy constraint"]
    assert result.strategy_names == ["A", "B", "C"]
    np.testing.assert_allclose(
        result.expected_net_benefits,
        np.array([[10.0, 8.0, 5.0], [7.0, 11.0, 9.0]]),
    )
    assert result.optimal_strategy_by_threshold_profile == {
        "wtp_reversal": "A",
        "policy_constraint": "B",
    }
    np.testing.assert_allclose(
        result.threshold_crossing_probability_matrix, np.array([[0.0, 1.0], [1.0, 0.0]])
    )
    np.testing.assert_allclose(
        result.decision_reversal_matrix, np.array([[0.0, 2.0], [4.0, 0.0]])
    )
    assert result.robust_strategy == "B"
    assert result.tipping_point_strategy == "B"
    assert result.pareto_strategies == ["A", "B"]
    assert result.reference_threshold_profile == "wtp_reversal"
    assert result.value == pytest.approx(2.0)
    assert result.method_maturity == "fixture-backed"
    assert result.reporting["analysis_type"] == "value_of_threshold_information"
    assert result.reporting["threshold_profile_ids"] == [
        "wtp_reversal",
        "policy_constraint",
    ]


def test_decision_analysis_wraps_value_of_threshold_information() -> None:
    """DecisionAnalysis should expose the threshold comparison method."""
    analysis = DecisionAnalysis(
        np.array(
            [
                [[10.0, 7.0], [8.0, 11.0]],
                [[10.0, 7.0], [8.0, 11.0]],
            ]
        )
    )

    result = analysis.value_of_threshold_information(
        threshold_profile_names=["wtp_reversal", "policy_constraint"],
        reference_threshold_profile="wtp_reversal",
    )

    assert result.threshold_profile_ids == ["wtp_reversal", "policy_constraint"]
    assert result.tipping_point_strategy == "Strategy 1"


def test_value_of_threshold_alias_matches_primary_api() -> None:
    """The shorter alias should match the contract-wording entry point."""
    values = np.array(
        [
            [[10.0, 7.0], [8.0, 11.0]],
            [[10.0, 7.0], [8.0, 11.0]],
        ]
    )

    result = value_of_threshold(
        values,
        threshold_profile_names=["wtp_reversal", "policy_constraint"],
        reference_threshold_profile="wtp_reversal",
    )

    assert result.threshold_profile_ids == ["wtp_reversal", "policy_constraint"]
    assert result.analysis_id == "threshold-screening-001"


def test_value_of_threshold_information_rejects_invalid_inputs() -> None:
    """Threshold VOI should reject malformed inputs."""
    with pytest.raises(InputError, match="3D net benefits"):
        value_of_threshold_information(np.ones((2, 2)))

    with pytest.raises(InputError, match="threshold_profile_names"):
        value_of_threshold_information(
            np.ones((2, 2, 2)),
            threshold_profile_names=["only-one"],
        )


def test_threshold_fixture_payload_matches_runtime_output() -> None:
    """The committed threshold fixture should match the live runtime output."""
    fixture_root = _threshold_fixture_root()
    with open(fixture_root / "normative" / "threshold-surface.json", encoding="utf-8") as handle:
        surface = json.load(handle)
    with open(
        fixture_root / "normative" / "value-of-threshold.json",
        encoding="utf-8",
    ) as handle:
        expected = json.load(handle)

    values = np.asarray(surface["net_benefit"])
    profiles = ThresholdProfileSet(
        [ThresholdProfile(**profile) for profile in surface["threshold_profiles"]]
    )
    result = value_of_threshold_information(
        values,
        threshold_profiles=profiles,
        strategy_names=surface["strategy_names"],
        reference_threshold_profile=surface["reference_threshold_profile"],
    )

    assert result.to_dict() == expected
