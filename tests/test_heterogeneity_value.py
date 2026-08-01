"""Tests for static/dynamic heterogeneity-value decomposition."""

# pyright: reportAny=false, reportExplicitAny=false, reportUnknownArgumentType=false, reportUnknownLambdaType=false, reportUnknownMemberType=false

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import random
from typing import Any

import pytest
from typer.testing import CliRunner

import voiage
from voiage.cli import app
from voiage.contracts.heterogeneity_value import (
    HETEROGENEITY_VALUE_INPUT_SCHEMA_V1,
    HETEROGENEITY_VALUE_RESULT_SCHEMA_V1,
    validate_heterogeneity_value_result,
)
from voiage.exceptions import InputError
from voiage.methods.heterogeneity_value import heterogeneity_value_decomposition

ROOT = Path(__file__).parents[1]
FIXTURE = ROOT / "specs/frontier/heterogeneity-value/v1/fixtures/normative"


def _input() -> dict[str, Any]:
    return json.loads((FIXTURE / "input.json").read_text(encoding="utf-8"))


def _result(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    return heterogeneity_value_decomposition(payload or _input()).to_contract_dict()


def test_normative_four_value_and_evsi_decompositions() -> None:
    result = _result()
    assert result == json.loads((FIXTURE / "expected.json").read_text(encoding="utf-8"))
    assert result["four_value_decomposition"] == {
        "c0": pytest.approx(8.0),
        "cf": pytest.approx(9.0),
        "p0": pytest.approx(9.0),
        "pf": pytest.approx(9.5),
        "static_value": pytest.approx(1.0),
        "dynamic_value": pytest.approx(0.5),
        "identity_residual": 0.0,
    }
    sample = result["sample_information"]
    assert sample["population_common_evsi"] == pytest.approx(0.4)
    assert sample["subgroup_policy_evsi"] == pytest.approx(0.0)
    assert sample["sample_informed_segmentation_value"] == pytest.approx(0.6)
    assert sample["identity_residual"] == 0.0
    assert sample["population_common_net_evsi"] == pytest.approx(0.15)
    assert [row["evsi"] for row in sample["subgroup_evsi"]] == pytest.approx([0.0, 0.0])


def test_minimization_is_sign_equivalent_and_sample_is_optional() -> None:
    payload = _input()
    payload["objective"]["direction"] = "minimize"
    payload["sample_information"] = None
    for state in payload["states"]:
        for values in state["subgroup_action_values"].values():
            for action_id in values:
                values[action_id] *= -1
    result = _result(payload)
    assert result["four_value_decomposition"]["static_value"] == pytest.approx(1.0)
    assert result["four_value_decomposition"]["dynamic_value"] == pytest.approx(0.5)
    assert result["sample_information"] is None


def test_zero_value_and_complete_ties_are_not_clipped_into_a_policy() -> None:
    payload = _input()
    payload["sample_information"] = None
    for state in payload["states"]:
        for values in state["subgroup_action_values"].values():
            values["a"] = values["b"] = 4.0
    result = _result(payload)
    assert result["four_value_decomposition"]["static_value"] == 0.0
    assert result["four_value_decomposition"]["dynamic_value"] == 0.0
    assert result["policy_audit"]["current_population_common"]["action_tie"] == [
        "a",
        "b",
    ]
    assert all(
        item["current_action_tie"] == ["a", "b"] for item in result["subgroup_results"]
    )


def test_randomized_independent_oracle_and_permutation_invariance() -> None:
    generator = random.Random(599)  # noqa: S311 - deterministic numerical oracle
    for case in range(100):
        payload = _input()
        payload["analysis_id"] = f"random-{case}"
        payload["sample_information"] = None
        payload["actions"].append({"action_id": "c", "label": "Action C"})
        for subgroup in payload["subgroups"]:
            subgroup["eligible_action_ids"].append("c")
        payload["states"].append(
            {
                "state_id": "middle",
                "probability": 0.2,
                "subgroup_action_values": {},
            }
        )
        probabilities = [0.2, 0.3, 0.5]
        for state, probability in zip(payload["states"], probabilities, strict=True):
            state["probability"] = probability
            state["subgroup_action_values"] = {
                group_id: {
                    action_id: generator.uniform(-20.0, 20.0)
                    for action_id in ("a", "b", "c")
                }
                for group_id in ("g1", "g2")
            }
        result = _result(payload)
        states = payload["states"]
        groups = ("g1", "g2")
        actions = ("a", "b", "c")
        current_group = {
            group: {
                action: sum(
                    state["probability"]
                    * state["subgroup_action_values"][group][action]
                    for state in states
                )
                for action in actions
            }
            for group in groups
        }
        c0 = max(
            sum(0.5 * current_group[group][action] for group in groups)
            for action in actions
        )
        cf = sum(0.5 * max(current_group[group].values()) for group in groups)
        p0 = sum(
            state["probability"]
            * max(
                sum(
                    0.5 * state["subgroup_action_values"][group][action]
                    for group in groups
                )
                for action in actions
            )
            for state in states
        )
        pf = sum(
            state["probability"]
            * sum(
                0.5 * max(state["subgroup_action_values"][group].values())
                for group in groups
            )
            for state in states
        )
        assert result["four_value_decomposition"] == {
            "c0": pytest.approx(c0),
            "cf": pytest.approx(cf),
            "p0": pytest.approx(p0),
            "pf": pytest.approx(pf),
            "static_value": pytest.approx(cf - c0),
            "dynamic_value": pytest.approx(pf - p0),
            "identity_residual": pytest.approx(0.0),
        }
        permuted = deepcopy(payload)
        permuted["actions"].reverse()
        permuted["subgroups"].reverse()
        permuted["states"].reverse()
        assert _result(permuted)["four_value_decomposition"] == pytest.approx(
            result["four_value_decomposition"]
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda data: data["subgroups"][1].update({"subgroup_id": "g1"}),
            "subgroup identifiers",
        ),
        (lambda data: data["subgroups"][0].update({"weight": 0.4}), "weights must sum"),
        (
            lambda data: data["states"][0].update({"probability": 0.4}),
            "probabilities must sum",
        ),
        (
            lambda data: data["states"][1].update({"state_id": "low"}),
            "state identifiers",
        ),
        (
            lambda data: data["actions"][1].update({"action_id": "a"}),
            "action identifiers",
        ),
        (
            lambda data: data["subgroups"][0].update(
                {"eligible_action_ids": ["missing"]}
            ),
            "unknown action",
        ),
        (
            lambda data: (
                data["subgroups"][0].update({"eligible_action_ids": ["a"]})
                or data["subgroups"][1].update({"eligible_action_ids": ["b"]})
            ),
            "eligible for every",
        ),
        (
            lambda data: data["states"][0]["subgroup_action_values"].pop("g2"),
            "enough properties|every subgroup",
        ),
        (
            lambda data: data["states"][0]["subgroup_action_values"].update(
                {"extra": {"a": 1.0}}
            ),
            "every subgroup exactly",
        ),
        (
            lambda data: data["states"][0]["subgroup_action_values"]["g1"].pop("a"),
            "eligible actions",
        ),
        (
            lambda data: data["states"][0]["subgroup_action_values"]["g1"].update(
                {"a": float("nan")}
            ),
            "finite",
        ),
        (
            lambda data: data["sample_information"]["cost"].update({"unit": "wrong"}),
            "cost unit",
        ),
        (
            lambda data: data["sample_information"]["signals"][1].update(
                {"signal_id": "low-signal"}
            ),
            "signal identifiers",
        ),
        (
            lambda data: data["sample_information"]["signals"][0][
                "likelihood_by_state"
            ].pop("low"),
            "every state",
        ),
        (
            lambda data: data["sample_information"]["signals"][0][
                "likelihood_by_state"
            ].update({"low": 0.7}),
            "must sum",
        ),
    ],
)
def test_semantic_pathologies_fail_closed(mutation: object, message: str) -> None:
    payload = _input()
    mutation(payload)  # type: ignore[operator]
    with pytest.raises(InputError, match=message):
        _ = _result(payload)


def test_schema_and_result_identity_validation_fail_closed() -> None:
    assert HETEROGENEITY_VALUE_INPUT_SCHEMA_V1["additionalProperties"] is False
    assert HETEROGENEITY_VALUE_RESULT_SCHEMA_V1["additionalProperties"] is False
    payload = _input()
    payload["unexpected"] = True
    with pytest.raises(InputError, match="unexpected"):
        _ = _result(payload)
    result = _result()
    result["four_value_decomposition"]["dynamic_value"] += 1.0
    with pytest.raises(ValueError, match="identity"):
        validate_heterogeneity_value_result(result)


def test_installed_schema_artifacts_match_runtime_contracts() -> None:
    schemas = FIXTURE.parents[1] / "schemas"
    assert json.loads((schemas / "input.schema.json").read_text(encoding="utf-8")) == (
        HETEROGENEITY_VALUE_INPUT_SCHEMA_V1
    )
    assert (
        json.loads((schemas / "result.schema.json").read_text(encoding="utf-8"))
        == HETEROGENEITY_VALUE_RESULT_SCHEMA_V1
    )


def test_public_api_cli_and_deterministic_copy(tmp_path: Path) -> None:
    assert voiage.heterogeneity_value_decomposition is heterogeneity_value_decomposition
    first = _result()
    second = _result()
    assert first == second
    first["analysis_id"] = "mutated"
    assert second["analysis_id"] != "mutated"
    output = tmp_path / "result.json"
    run = CliRunner().invoke(
        app,
        [
            "--format",
            "json",
            "calculate-heterogeneity-value",
            str(FIXTURE / "input.json"),
            "--output",
            str(output),
        ],
    )
    assert run.exit_code == 0, run.stdout
    contract = json.loads(output.read_text(encoding="utf-8"))
    assert contract["four_value_decomposition"]["static_value"] == pytest.approx(1.0)
    assert json.loads(run.stdout) == contract


def test_result_validator_rejects_negative_optimized_value() -> None:
    result = _result()
    result["four_value_decomposition"].update(
        {
            "c0": 8.0,
            "cf": 7.0,
            "p0": 9.0,
            "pf": 8.0,
            "static_value": -1.0,
            "dynamic_value": -1.0,
        }
    )
    result["perfect_information"].update(
        {"population_common_evpi": 1.0, "subgroup_policy_evpi": 1.0}
    )
    with pytest.raises(ValueError, match="nonnegative"):
        validate_heterogeneity_value_result(result)


def test_ineligible_state_value_is_rejected_not_silently_ignored() -> None:
    payload = deepcopy(_input())
    payload["subgroups"][0]["eligible_action_ids"] = ["a"]
    with pytest.raises(InputError, match="eligible actions exactly"):
        _ = _result(payload)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda result: result["policy_audit"]["current_population_common"][
                "action_values"
            ].update({"a": 99.0}),
            "reconstruct C0",
        ),
        (
            lambda result: result["subgroup_results"][0].update({"evpi": 99.0}),
            "subgroup result|subgroup EVPI",
        ),
        (
            lambda result: result["subgroup_results"][0].update(
                {"current_value": 99.0}
            ),
            "subgroup result",
        ),
        (
            lambda result: result["subgroup_results"][0].update(
                {"perfect_information_value": 99.0}
            ),
            "subgroup result",
        ),
        (
            lambda result: result["sample_information"].update(
                {"population_common_net_evsi": 99.0}
            ),
            "sample-information decomposition",
        ),
        (
            lambda result: result["sample_information"]["signals"][0].update(
                {"probability": 0.6}
            ),
            "signal probabilities",
        ),
        (
            lambda result: result["sample_information"]["signals"][0][
                "subgroup_policies"
            ]["g1"]["joint_weighted_action_values"].update({"a": 99.0}),
            "reconstruct S0 and Sf",
        ),
        (
            lambda result: result["sample_information"]["subgroup_evsi"][0].update(
                {"weighted_evsi_contribution": 99.0}
            ),
            "subgroup sample|subgroup EVSI",
        ),
        (
            lambda result: result["sample_information"]["subgroup_evsi"][0].update(
                {"evsi": 99.0}
            ),
            "subgroup sample",
        ),
        (
            lambda result: result["sample_information"]["subgroup_evsi"][0].update(
                {"sample_value": 99.0}
            ),
            "subgroup sample",
        ),
        (
            lambda result: result["language_dispositions"].pop("rust"),
            "language_dispositions|language dispositions",
        ),
    ],
)
def test_result_audit_mutations_fail_closed(mutation: object, message: str) -> None:
    result = _result()
    mutation(result)  # type: ignore[operator]
    with pytest.raises(ValueError, match=message):
        validate_heterogeneity_value_result(result)
