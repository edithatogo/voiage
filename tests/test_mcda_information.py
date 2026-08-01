"""Focused executable assurance for finite additive-MCDA information value."""

# pyright: reportAny=false, reportExplicitAny=false, reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownLambdaType=false, reportUnusedCallResult=false

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

from jsonschema import Draft202012Validator
import pytest

from voiage.contracts.mcda_information import MCDA_INFORMATION_RESULT_SCHEMA_V1
from voiage.exceptions import InputError
import voiage.methods.mcda_information as mcda_information_module
from voiage.methods.mcda_information import mcda_information_value

if TYPE_CHECKING:
    from collections.abc import Callable

ROOT = Path(__file__).parents[1]
FIXTURE = ROOT / "specs/frontier/mcda-information/v1/fixtures/normative/input.json"
EXPECTED = ROOT / "specs/frontier/mcda-information/v1/fixtures/normative/expected.json"


def _input() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _evaluate(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    return mcda_information_value(payload or _input()).to_contract_dict()


def _actions(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {action["action_type"]: action for action in result["conditional_actions"]}


def _rounded(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 14)
    if isinstance(value, dict):
        return {key: _rounded(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_rounded(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_rounded(item) for item in value)
    return value


def _numeric_signature(result: dict[str, Any]) -> dict[str, Any]:
    """Retain order-insensitive decision quantities for invariance tests."""
    actions = _actions(result)
    return _rounded(
        {
            "expected_scores": result["baseline"]["expected_scores"],
            "baseline_choice": sorted(result["baseline"]["choice_tie"]),
            "baseline_value": result["baseline"]["value"],
            "action_values": {
                action_type: {
                    "resolved": action["resolved_value"],
                    "gross": action["gross_voi"],
                    "net": action["net_voi"],
                    "regret": action["expected_regret"],
                    "partitions": sorted(
                        (
                            tuple(sorted(partition["key_values"].items())),
                            partition["probability"],
                            partition["conditional_scores"],
                            sorted(partition["choice_tie"]),
                            partition["conditional_value"],
                        )
                        for partition in action["partitions"]
                    ),
                }
                for action_type, action in actions.items()
            },
            "decomposition": result["decomposition"],
            "baseline_regret": result["regret"]["baseline_expected"],
            "rank_acceptability": result["rank_acceptability"]["by_alternative"],
            "expected_vectors": result["pareto"]["expected_value_vectors"],
            "expected_dominance": result["pareto"]["expected_dominance"],
            "expected_non_dominated": result["pareto"]["expected_non_dominated"],
        }
    )


def _assert_contract_match(actual: Any, expected: Any, path: str = "$") -> None:
    """Require exact structure/text and tolerance-bounded numeric fixture parity."""
    if isinstance(expected, float | int) and not isinstance(expected, bool):
        assert actual == pytest.approx(expected, abs=2e-12), path
    elif isinstance(expected, dict):
        assert isinstance(actual, dict), path
        assert set(actual) == set(expected), path
        for key, value in expected.items():
            _assert_contract_match(actual[key], value, f"{path}.{key}")
    elif isinstance(expected, list):
        assert isinstance(actual, list), path
        assert len(actual) == len(expected), path
        for index, value in enumerate(expected):
            _assert_contract_match(actual[index], value, f"{path}[{index}]")
    else:
        assert actual == expected, path


def test_normative_analysis_matches_independent_analytic_results_and_schema() -> None:
    result = _evaluate()
    Draft202012Validator(MCDA_INFORMATION_RESULT_SCHEMA_V1).validate(result)

    assert result["baseline"]["expected_scores"] == pytest.approx(
        {"service-a": 0.485, "service-b": 0.63375}
    )
    assert result["baseline"]["choice_tie"] == ["service-b"]
    assert result["baseline"]["value"] == pytest.approx(0.63375)

    actions = _actions(result)
    assert actions["criterion"]["resolved_value"] == pytest.approx(0.63375)
    assert actions["criterion"]["gross_voi"] == pytest.approx(0.0, abs=2e-16)
    assert actions["criterion"]["net_voi"] == pytest.approx(-0.005)
    assert actions["preference"]["resolved_value"] == pytest.approx(0.63375)
    assert actions["preference"]["gross_voi"] == pytest.approx(0.0, abs=2e-16)
    assert actions["preference"]["net_voi"] == pytest.approx(-0.004)
    assert actions["joint"]["resolved_value"] == pytest.approx(0.66175)
    assert actions["joint"]["gross_voi"] == pytest.approx(0.028)
    assert actions["joint"]["net_voi"] == pytest.approx(0.018)

    assert result["regret"]["baseline_expected"] == pytest.approx(0.028)
    ranks = result["rank_acceptability"]["by_alternative"]
    assert ranks["service-a"] == pytest.approx([0.35, 0.65])
    assert ranks["service-b"] == pytest.approx([0.65, 0.35])
    assert result["pareto"]["expected_non_dominated"] == [
        "service-a",
        "service-b",
    ]
    assert result["assurance"] == {
        "estimator": "exact_finite_enumeration",
        "arithmetic": "binary64_with_declared_tolerances",
        "joint_dependence_preserved": True,
        "normalization_frozen_ex_ante": True,
        "gross_voi_clipped": False,
        "probabilities_reconciled": True,
        "weights_reconciled": True,
        "fixture_status": "analytically_reviewed_contract_fixture",
    }


def test_complete_runtime_contract_matches_independent_normative_fixture() -> None:
    expected = json.loads(EXPECTED.read_text(encoding="utf-8"))
    _assert_contract_match(_evaluate(), expected)


def test_joint_law_drives_conditionals_and_nonadditive_information_interaction() -> (
    None
):
    correlated = _evaluate()
    correlated_actions = _actions(correlated)
    outcome_partitions = {
        partition["key_values"]["outcome_regime"]: partition
        for partition in correlated_actions["criterion"]["partitions"]
    }
    assert outcome_partitions["favourable"]["conditional_scores"] == pytest.approx(
        {"service-a": 0.625, "service-b": 0.635}
    )
    assert correlated["decomposition"] == {
        "criterion_action_id": "learn-outcome",
        "preference_action_id": "learn-preference",
        "joint_action_id": "learn-joint",
        "criterion_gross_voi": pytest.approx(0.0, abs=2e-16),
        "preference_gross_voi": pytest.approx(0.0, abs=2e-16),
        "joint_gross_voi": pytest.approx(0.028),
        "interaction": pytest.approx(0.028),
        "joint_increment_over_criterion": pytest.approx(0.028),
        "joint_increment_over_preference": pytest.approx(0.028),
        "no_double_counting_identity_residual": 0.0,
    }

    independent = _input()
    for state in independent["joint_states"]:
        state["probability"] = 0.25
    independent_result = _evaluate(independent)
    independent_actions = _actions(independent_result)
    independent_outcome = {
        partition["key_values"]["outcome_regime"]: partition
        for partition in independent_actions["criterion"]["partitions"]
    }
    assert independent_outcome["favourable"]["conditional_scores"] == pytest.approx(
        {"service-a": 0.575, "service-b": 0.645}
    )
    assert independent_result["decomposition"]["interaction"] == pytest.approx(0.02)


def test_complete_ties_are_retained_in_choices_and_fractional_rank_acceptability() -> (
    None
):
    payload = _input()
    for state in payload["joint_states"]:
        state["performances"]["service-b"] = deepcopy(
            state["performances"]["service-a"]
        )

    result = _evaluate(payload)
    assert result["baseline"]["choice_tie"] == ["service-a", "service-b"]
    assert len(result["baseline"]["ranking"]) == 1
    for action in result["conditional_actions"]:
        assert action["gross_voi"] == pytest.approx(0.0, abs=2e-16)
        assert action["expected_regret"] == pytest.approx(0.0)
        assert all(
            partition["choice_tie"] == ["service-a", "service-b"]
            for partition in action["partitions"]
        )
    assert result["regret"]["baseline_expected"] == pytest.approx(0.0)
    assert result["rank_acceptability"]["by_alternative"] == pytest.approx(
        {"service-a": [0.5, 0.5], "service-b": [0.5, 0.5]}
    )
    for groups in result["rank_acceptability"]["state_tie_groups"].values():
        assert groups[0]["alternative_ids"] == ["service-a", "service-b"]


def test_alternative_criterion_and_state_permutations_preserve_decision_quantities() -> (
    None
):
    expected = _numeric_signature(_evaluate())
    payload = _input()
    payload["alternatives"].reverse()
    payload["criteria"].reverse()
    payload["joint_states"].reverse()

    assert _numeric_signature(_evaluate(payload)) == expected


def test_equivalent_raw_unit_and_anchor_conversion_preserves_all_results() -> None:
    expected = _evaluate()
    payload = _input()
    quality = next(
        criterion
        for criterion in payload["criteria"]
        if criterion["criterion_id"] == "quality"
    )
    quality["raw_unit"] = "tenths of a quality point"
    for anchor in quality["value_function"]["anchors"]:
        anchor["raw"] *= 10
    quality["value_function"]["valid_domain"] = [0.0, 1000.0]
    for state in payload["joint_states"]:
        for performances in state["performances"].values():
            performances["quality"] *= 10

    assert _evaluate(payload) == expected


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload["joint_states"][0]["performances"]["service-a"].update(
            quality=101.0
        ),
        lambda payload: payload.update(aggregation_family="electre_outranking"),
        lambda payload: payload["information_actions"][0].update(
            outcome_partition_keys=["undeclared-regime"]
        ),
        lambda payload: payload["joint_states"][0].update(probability=float("nan")),
    ],
)
def test_invalid_domains_and_contract_boundaries_raise_public_input_error(
    mutation: Callable[[dict[str, Any]], object],
) -> None:
    payload = _input()
    mutation(payload)
    with pytest.raises(InputError):
        _ = mcda_information_value(payload)


def test_internal_evaluation_rejects_raw_performance_outside_fixed_domain() -> None:
    """Keep the defensive evaluator guard covered independently of schema checks."""
    criterion = next(
        item for item in _input()["criteria"] if item["criterion_id"] == "quality"
    )

    with pytest.raises(ValueError, match="falls outside the fixed valid domain"):
        _ = mcda_information_module._linear_value(101.0, criterion)


def test_internal_evaluation_rejects_nonfinite_scores_and_negative_information() -> (
    None
):
    """Malformed internal payloads fail closed even when public validation is bypassed."""
    nonfinite = _input()
    criterion = next(
        item for item in nonfinite["criteria"] if item["criterion_id"] == "quality"
    )
    criterion["value_function"]["extrapolation_policy"] = "allow"
    criterion["value_function"]["anchors"][1]["value"] = 1e308
    for state in nonfinite["joint_states"]:
        for performances in state["performances"].values():
            performances["quality"] = 1e308
    with pytest.raises(ValueError, match="aggregate MCDA score must be finite"):
        _ = mcda_information_module._evaluate(nonfinite)

    signed_probability = _input()
    signed_probability["joint_states"][0]["probability"] = -0.5
    signed_probability["joint_states"][1]["probability"] = 1.5
    with pytest.raises(
        ValueError,
        match="partition refinement produced negative gross information value",
    ):
        _ = mcda_information_module._evaluate(signed_probability)


def test_result_copy_is_independent_and_json_compatible() -> None:
    result = mcda_information_value(_input())
    first = result.to_contract_dict()
    first["baseline"]["expected_scores"]["service-a"] = -999.0
    second = result.to_contract_dict()
    assert second["baseline"]["expected_scores"]["service-a"] == pytest.approx(0.485)
    assert json.loads(json.dumps(second)) == second


def test_exact_gross_voi_is_nonnegative_by_construction_and_never_hidden_by_clipping() -> (
    None
):
    result = _evaluate()
    assert result["assurance"]["gross_voi_clipped"] is False
    for action in result["conditional_actions"]:
        recomputed_resolved = math.fsum(
            partition["probability"] * partition["conditional_value"]
            for partition in action["partitions"]
        )
        raw_difference = recomputed_resolved - result["baseline"]["value"]
        assert raw_difference >= -1e-15
        assert action["gross_voi"] == raw_difference
