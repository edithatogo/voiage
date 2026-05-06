"""Tests for model-validation VOI analysis."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from voiage import ValidationProfile, ValidationProfileSet, value_of_model_validation
from voiage.analysis import DecisionAnalysis
from voiage.exceptions import InputError
from voiage.methods.validation import ModelValidationResult, value_of_validation


def _validation_fixture_root() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "specs"
        / "frontier"
        / "validation"
        / "v1"
        / "fixtures"
    )


def test_value_of_model_validation_compares_validation_profiles() -> None:
    """Validation VOI should expose discrepancy reduction and robust summaries."""
    values = np.array(
        [
            [[10.0, 7.0], [8.0, 11.0], [5.0, 9.0]],
            [[10.0, 7.0], [8.0, 11.0], [5.0, 9.0]],
        ]
    )
    profiles = ValidationProfileSet(
        [
            ValidationProfile(
                id="external_validation", label="External validation", weight=0.6
            ),
            ValidationProfile(
                id="discrepancy_reduction",
                label="Discrepancy reduction",
                weight=0.4,
            ),
        ]
    )

    result = value_of_model_validation(
        values,
        validation_profiles=profiles,
        strategy_names=["A", "B", "C"],
        reference_validation_profile="external_validation",
    )

    assert isinstance(result, ModelValidationResult)
    assert result.analysis_id == "validation-screening-001"
    assert result.decision_problem_id == "screening-program-001"
    assert result.validation_profile_ids == [
        "external_validation",
        "discrepancy_reduction",
    ]
    assert result.validation_profile_labels == [
        "External validation",
        "Discrepancy reduction",
    ]
    assert result.strategy_names == ["A", "B", "C"]
    np.testing.assert_allclose(
        result.expected_net_benefits,
        np.array([[10.0, 8.0, 5.0], [7.0, 11.0, 9.0]]),
    )
    assert result.optimal_strategy_by_validation_profile == {
        "external_validation": "A",
        "discrepancy_reduction": "B",
    }
    np.testing.assert_allclose(
        result.discrepancy_matrix, np.array([[0.0, 2.0], [4.0, 0.0]])
    )
    assert result.consensus_strategy == "B"
    assert result.robust_strategy == "B"
    assert result.pareto_strategies == ["A", "B"]
    assert result.reference_validation_profile == "external_validation"
    assert result.value == pytest.approx(1.6)
    assert result.discrepancy_reduction_value == pytest.approx(1.2)
    assert result.method_maturity == "fixture-backed"
    assert result.reporting["analysis_type"] == "value_of_model_validation"
    assert result.reporting["validation_profile_ids"] == [
        "external_validation",
        "discrepancy_reduction",
    ]


def test_decision_analysis_wraps_value_of_model_validation() -> None:
    """DecisionAnalysis should expose the validation-comparison method."""
    analysis = DecisionAnalysis(
        np.array(
            [
                [[10.0, 7.0], [8.0, 11.0]],
                [[10.0, 7.0], [8.0, 11.0]],
            ]
        )
    )

    result = analysis.value_of_model_validation(
        validation_profile_names=["external_validation", "discrepancy_reduction"],
        reference_validation_profile="external_validation",
    )

    assert result.validation_profile_ids == [
        "external_validation",
        "discrepancy_reduction",
    ]
    assert result.consensus_strategy == "Strategy 1"


def test_value_of_validation_alias_matches_primary_api() -> None:
    """The shorter alias should match the contract-wording entry point."""
    values = np.array(
        [
            [[10.0, 7.0], [8.0, 11.0]],
            [[10.0, 7.0], [8.0, 11.0]],
        ]
    )

    result = value_of_validation(
        values,
        validation_profile_names=["external_validation", "discrepancy_reduction"],
        reference_validation_profile="external_validation",
    )

    assert result.validation_profile_ids == [
        "external_validation",
        "discrepancy_reduction",
    ]
    assert result.analysis_id == "validation-screening-001"


def test_value_of_model_validation_rejects_invalid_inputs() -> None:
    """Validation VOI should reject malformed inputs."""
    with pytest.raises(InputError, match="3D net benefits"):
        value_of_model_validation(np.ones((2, 2)))

    with pytest.raises(InputError, match="validation_profile_names"):
        value_of_model_validation(
            np.ones((2, 2, 2)),
            validation_profile_names=["only-one"],
        )


def test_validation_fixture_payload_matches_runtime_output() -> None:
    """The committed validation fixture should match the live runtime output."""
    fixture_root = _validation_fixture_root()
    with open(
        fixture_root / "normative" / "validation-surface.json", encoding="utf-8"
    ) as handle:
        surface = json.load(handle)
    with open(
        fixture_root / "normative" / "value-of-model-validation.json",
        encoding="utf-8",
    ) as handle:
        expected = json.load(handle)

    values = np.asarray(surface["net_benefit"])
    profiles = ValidationProfileSet(
        [ValidationProfile(**profile) for profile in surface["validation_profiles"]]
    )
    result = value_of_model_validation(
        values,
        validation_profiles=profiles,
        strategy_names=surface["strategy_names"],
        reference_validation_profile=surface["reference_validation_profile"],
    )

    assert result.to_dict() == expected
