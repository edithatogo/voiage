"""Analytical and boundary tests for issue #557 distribution-family information."""

from __future__ import annotations

import numpy as np
import pytest

from voiage.methods.distributional_information import (
    _validate_information_values,
    value_of_distributional_information,
)


def _exact_result(**overrides: object):
    arguments: dict[str, object] = {
        "conditional_values": np.asarray([[10.0, 6.0], [4.0, 12.0]]),
        "model_ids": ["family-a", "family-b"],
        "alternative_names": ["A", "B"],
        "model_probabilities": [0.5, 0.5],
        "value_unit": "net-benefit-point",
        "information_cost": 0.5,
        "provenance": {
            "fixture_id": "vdi-exact-v1",
            "probability_source": "synthetic-reference",
            "value_source": "conditional-expectation-table",
            "family_definition_source": "synthetic-candidate-families",
        },
        "comparability": {
            "population_id": "same-population",
            "horizon_id": "same-horizon",
            "discounting_id": "same-discounting",
            "value_semantics_id": "conditional-expected-value",
            "cost_location_id": "same-cost-location",
            "verified": True,
            "verification_reference": "test:comparability",
        },
        "conditional_value_assurance": {
            "input_status": "exact_enumerated_conditional_expectations",
            "source_values_exact": True,
            "source_uncertainty": "none_by_construction",
            "enumeration_method": "test analytical enumeration",
            "evidence_reference": "test:exact-values",
        },
    }
    arguments.update(overrides)
    arguments.setdefault(
        "model_labels", {name: name for name in arguments["model_ids"]}
    )
    arguments.setdefault(
        "model_definitions",
        [
            {
                "model_id": name,
                "family_or_assumption": f"synthetic {name}",
                "parameterization": "finite exact table",
                "within_family_integration": "analytical expectation",
                "definition_source": "test definition",
                "parameter_source": "test exact parameters",
                "data_reference": f"test:{name}",
                "value_transformation": "identity",
            }
            for name in arguments["model_ids"]
        ],
    )
    return value_of_distributional_information(**arguments)


def test_exact_two_family_reference() -> None:
    result = _exact_result()
    assert result.current_value == pytest.approx(9.0)
    assert result.current_optimal_alternatives == ["B"]
    assert result.expected_resolved_value == pytest.approx(11.0)
    assert result.gross_vdi == pytest.approx(2.0)
    assert result.net_vdi == pytest.approx(1.5)
    assert [item.optimal_alternatives for item in result.resolved_models] == [
        ["A"],
        ["B"],
    ]
    assert [item.model_id for item in result.resolved_models] == result.model_ids
    assert [item.probability for item in result.resolved_models] == (
        result.model_probabilities
    )
    assert sum(item.weighted_contribution for item in result.resolved_models) == (
        pytest.approx(result.expected_resolved_value)
    )
    assert (
        result.estimator["input_value_status"]
        == (result.conditional_value_assurance["input_status"])
    )
    assert (
        result.estimator["evidence_reference"]
        == (result.conditional_value_assurance["evidence_reference"])
    )


def test_model_and_alternative_permutations_preserve_value() -> None:
    original = _exact_result()
    model_permuted = _exact_result(
        conditional_values=np.asarray([[4.0, 12.0], [10.0, 6.0]]),
        model_ids=["family-b", "family-a"],
    )
    alternative_permuted = _exact_result(
        conditional_values=np.asarray([[6.0, 10.0], [12.0, 4.0]]),
        alternative_names=["B", "A"],
    )
    assert model_permuted.gross_vdi == pytest.approx(original.gross_vdi)
    assert alternative_permuted.gross_vdi == pytest.approx(original.gross_vdi)


def test_minimization_is_direction_equivalent() -> None:
    utility = _exact_result(information_cost=0.0)
    loss = _exact_result(
        conditional_values=-np.asarray([[10.0, 6.0], [4.0, 12.0]]),
        direction="minimize",
        information_cost=0.0,
    )
    assert loss.gross_vdi == pytest.approx(utility.gross_vdi)


def test_complete_ties_and_canonical_representative() -> None:
    result = _exact_result(
        conditional_values=np.asarray([[5.0, 5.0], [5.0, 5.0]]),
        alternative_names=["z", "a"],
        information_cost=0.0,
    )
    assert result.current_optimal_alternatives == ["a", "z"]
    assert result.current_representative == "a"
    assert result.gross_vdi == 0.0


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"model_probabilities": [0.4, 0.4]}, "sum to 1"),
        ({"model_probabilities": [-0.1, 1.1]}, "non-negative"),
        ({"model_ids": ["duplicate", "duplicate"]}, "unique"),
        ({"alternative_names": ["A", "A"]}, "unique"),
        ({"value_unit": ""}, "value_unit"),
        ({"conditional_values": [[1.0, float("nan")], [2.0, 3.0]]}, "finite"),
    ],
)
def test_fail_closed_pathologies(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _exact_result(**overrides)


def test_information_cost_is_signed_and_not_clipped() -> None:
    result = _exact_result(information_cost=3.0)
    assert result.gross_vdi == pytest.approx(2.0)
    assert result.net_vdi == pytest.approx(-1.0)


def test_single_family_common_optimum_and_zero_probability_have_zero_value() -> None:
    single = _exact_result(
        conditional_values=[[5.0, 2.0]],
        model_ids=["only"],
        model_probabilities=[1.0],
        information_cost=0.0,
    )
    common = _exact_result(
        conditional_values=[[5.0, 2.0], [8.0, 1.0]], information_cost=0.0
    )
    zero_probability = _exact_result(
        conditional_values=[[10.0, 6.0], [-1000.0, 1000.0]],
        model_probabilities=[1.0, 0.0],
        information_cost=0.0,
    )
    assert single.gross_vdi == common.gross_vdi == zero_probability.gross_vdi == 0.0


def test_splitting_identical_family_preserves_value() -> None:
    original = _exact_result(information_cost=0.0)
    split = _exact_result(
        conditional_values=[[10.0, 6.0], [10.0, 6.0], [4.0, 12.0]],
        model_ids=["family-a-1", "family-a-2", "family-b"],
        model_probabilities=[0.2, 0.3, 0.5],
        information_cost=0.0,
    )
    assert split.gross_vdi == pytest.approx(original.gross_vdi)


def test_positive_scaling_and_common_translation() -> None:
    original = _exact_result(information_cost=0.0)
    scaled = _exact_result(
        conditional_values=3.0 * np.asarray([[10.0, 6.0], [4.0, 12.0]]),
        information_cost=0.0,
    )
    shifted = _exact_result(
        conditional_values=np.asarray([[10.0, 6.0], [4.0, 12.0]]) + 17.0,
        information_cost=0.0,
    )
    assert scaled.gross_vdi == pytest.approx(3.0 * original.gross_vdi)
    assert shifted.gross_vdi == pytest.approx(original.gross_vdi)


def test_model_family_value_is_bounded_by_matched_full_information() -> None:
    result = _exact_result(information_cost=0.0)
    within_family_draws = np.asarray(
        [
            [[12.0, 4.0], [8.0, 8.0]],
            [[7.0, 9.0], [1.0, 15.0]],
        ]
    )
    full_information_value = float(
        np.mean(np.max(within_family_draws, axis=2))
        - np.max(np.mean(within_family_draws, axis=(0, 1)))
    )
    assert result.gross_vdi <= full_information_value + 1e-12


def test_provenance_is_required_and_exact() -> None:
    with pytest.raises(ValueError, match="provenance"):
        _exact_result(provenance={})


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"model_ids": []}, "model_ids"),
        ({"model_ids": ["family-a", " "]}, "model_ids"),
        ({"model_ids": ["family-a", 2]}, "model_ids"),
        ({"alternative_names": []}, "alternative_names"),
        ({"alternative_names": ["A", " "]}, "alternative_names"),
        ({"model_labels": {"family-a": "A", "family-b": " "}}, "model_labels"),
        ({"analysis_id": " "}, "analysis_id"),
        ({"value_unit": 7}, "value_unit"),
        ({"probability_sum_tolerance": 0.0}, "probability_sum_tolerance"),
        (
            {
                "model_probabilities": [0.0, 0.0],
                "probability_sum_tolerance": 1.0,
            },
            "at most 1e-6",
        ),
        (
            {
                "provenance": {
                    "fixture_id": "vdi-exact-v1",
                    "probability_source": " ",
                    "value_source": "conditional-expectation-table",
                    "family_definition_source": "synthetic-candidate-families",
                }
            },
            "provenance",
        ),
    ],
)
def test_direct_runtime_rejects_non_string_or_blank_metadata(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        _exact_result(**overrides)


def test_defensive_arithmetic_contract_rejects_impossible_results() -> None:
    with pytest.raises(ArithmeticError, match="negative"):
        _validate_information_values(
            current_value=2.0,
            expected_resolved=1.0,
            gross=-1.0,
            cost=0.0,
            net=-1.0,
        )
    with pytest.raises(ArithmeticError, match="non-finite"):
        _validate_information_values(
            current_value=1.0,
            expected_resolved=1.0,
            gross=0.0,
            cost=0.0,
            net=float("nan"),
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"direction": "sideways"}, "direction"),
        ({"information_cost": float("inf")}, "information_cost"),
        ({"information_cost": -1.0}, "information_cost"),
        ({"absolute_tolerance": -1.0}, "tolerances"),
        ({"model_labels": {"family-a": "A"}}, "exactly match"),
        ({"comparability": {}}, "comparability"),
    ],
)
def test_runtime_metadata_and_numerical_controls_fail_closed(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises((ValueError, TypeError), match=message):
        _exact_result(**overrides)


def test_model_definitions_are_complete_and_ordered() -> None:
    with pytest.raises(ValueError, match="model_definitions"):
        _exact_result(model_definitions=[])
    with pytest.raises(ValueError, match="model_ids order"):
        _exact_result(
            model_definitions=[
                {
                    "model_id": "family-b",
                    "family_or_assumption": "B",
                    "parameterization": "exact",
                    "within_family_integration": "analytical",
                    "definition_source": "test",
                    "parameter_source": "test",
                    "data_reference": "test:B",
                    "value_transformation": "identity",
                },
                {
                    "model_id": "family-a",
                    "family_or_assumption": "A",
                    "parameterization": "exact",
                    "within_family_integration": "analytical",
                    "definition_source": "test",
                    "parameter_source": "test",
                    "data_reference": "test:A",
                    "value_transformation": "identity",
                },
            ]
        )


def test_estimated_or_unverified_inputs_cannot_be_certified_exact() -> None:
    with pytest.raises(ValueError, match="exact enumerated input values"):
        _exact_result(
            conditional_value_assurance={
                "input_status": "monte_carlo_estimate",
                "source_values_exact": False,
                "source_uncertainty": "unknown",
                "enumeration_method": "simulation",
                "evidence_reference": "test:simulation",
            }
        )
    comparability = {
        "population_id": "same",
        "horizon_id": "same",
        "discounting_id": "same",
        "value_semantics_id": "same",
        "cost_location_id": "same",
        "verified": False,
        "verification_reference": "test:not-verified",
    }
    with pytest.raises(ValueError, match="explicitly verified"):
        _exact_result(comparability=comparability)
