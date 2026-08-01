"""Analytical and boundary tests for issue #557 distribution-family information."""

from __future__ import annotations

import numpy as np
import pytest

from voiage.methods.distributional_information import (
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
            "population": "same",
            "horizon": "same",
            "discounting": "same",
            "value_semantics": "conditional expected value",
            "cost_location": "same scale",
        },
    }
    arguments.update(overrides)
    arguments.setdefault(
        "model_labels", {name: name for name in arguments["model_ids"]}
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
