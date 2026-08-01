"""Exact contract assurance for finite uncertainty-modelling value."""

# pyright: reportAny=false, reportExplicitAny=false, reportUnknownArgumentType=false, reportUnknownLambdaType=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnusedCallResult=false

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
import pytest
from typer.testing import CliRunner

from voiage import (
    UncertaintyModellingValueResult as PublicUncertaintyModellingValueResult,
)
from voiage import uncertainty_modelling_value as public_uncertainty_modelling_value
from voiage.cli import app
from voiage.contracts.uncertainty_modelling_value import (
    validate_uncertainty_modelling_value_result,
    validate_uncertainty_modelling_value_semantics,
)
from voiage.exceptions import InputError
from voiage.methods.uncertainty_modelling_value import value_of_uncertainty_modelling

ROOT = Path(__file__).parents[1]
CONTRACT = ROOT / "specs/frontier/uncertainty-modelling-value/v1"
TWO_STAGE = CONTRACT / "fixtures/normative/two-stage-nonlinear-input.json"
MULTISTAGE = CONTRACT / "fixtures/normative/multistage-input.json"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _input() -> dict[str, Any]:
    return _json(TWO_STAGE)


def _result(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    return value_of_uncertainty_modelling(payload or _input()).to_contract_dict()


def test_nonlinear_point_estimate_reference_decomposition() -> None:
    result = _result()
    expected = _json(CONTRACT / "fixtures/normative/two-stage-nonlinear-expected.json")
    assert result == expected
    assert result["expected_value_problem"]["selected_candidate_id"] == "risky-at-mean"
    assert result["expected_result_of_ev_solution"]["value"] == pytest.approx(9.0)
    assert result["recourse_problem"]["value"] == pytest.approx(8.0)
    assert result["wait_and_see"]["value"] == pytest.approx(5.0)
    assert result["decomposition"] == {
        "eviu": 1.0,
        "eviu_comparator": "declared_point_estimate_ev_solution",
        "eviu_equals_vss_under_v1_contract": True,
        "evpi": 3.0,
        "identity_status": "verified",
        "vss": 1.0,
    }


def test_multistage_maximization_and_nonanticipativity_reference() -> None:
    result = _result(_json(MULTISTAGE))
    assert result == _json(CONTRACT / "fixtures/normative/multistage-expected.json")
    assert result["recourse_problem"]["policy_tie"] == [
        "adaptive-policy",
        "commit-policy",
    ]
    assert result["decomposition"]["vss"] == 0.0
    assert result["decomposition"]["evpi"] == pytest.approx(1.5)
    assert result["assurance"]["nonanticipativity_representation"] == (
        "one_decision_per_shared_history"
    )
    assert [stage["stage_id"] for stage in result["scenario_structure"]["stages"]] == [
        "launch",
        "review",
        "adapt",
    ]
    assert len(result["policy_audit"][0]["state_outcomes"]) == 3


def test_multistage_histories_form_a_filtration() -> None:
    payload = _json(MULTISTAGE)
    payload["histories"] = [
        *[
            history
            for history in payload["histories"]
            if history["stage_id"] == "review"
        ],
        {
            "history_id": "adapt-cross",
            "stage_id": "adapt",
            "reachable_states": ["down", "up"],
        },
        {
            "history_id": "adapt-flat",
            "stage_id": "adapt",
            "reachable_states": ["flat"],
        },
    ]
    with pytest.raises(InputError, match="refine the prior-stage"):
        _result(payload)

    payload = _json(MULTISTAGE)
    payload["stages"][2]["information_available"] = ["late-band"]
    with pytest.raises(InputError, match="information available must be cumulative"):
        _result(payload)


def test_infeasible_recourse_for_ev_solution_is_explicit() -> None:
    payload = _json(CONTRACT / "fixtures/normative/infeasible-recourse-input.json")
    result = _result(payload)
    assert result == _json(
        CONTRACT / "fixtures/normative/infeasible-recourse-expected.json"
    )
    assert result["expected_result_of_ev_solution"] == {
        "status": "infeasible_recourse",
        "value": None,
        "infeasible_states": ["high"],
    }
    assert result["decomposition"]["vss"] is None
    assert result["decomposition"]["eviu"] is None
    assert result["decomposition"]["identity_status"] == "not_estimable_infeasible_eev"


def test_no_policy_with_relatively_complete_recourse_fails_closed() -> None:
    payload = _input()
    for policy in payload["policies"]:
        outcome = policy["state_outcomes"][0]
        outcome.update(
            {
                "feasible": False,
                "objective_value": None,
                "recourse_status": "infeasible",
            }
        )
    with pytest.raises(InputError, match="no policy feasible"):
        _result(payload)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda data: data["states"][1].update({"state_id": "high"}),
            "state identifiers",
        ),
        (lambda data: data["states"][0].update({"probability": 0.4}), "sum to one"),
        (lambda data: data["stages"][1].update({"order": 3}), "contiguous"),
        (lambda data: data["stages"][1].update({"stage_id": "commit"}), "unique"),
        (
            lambda data: (
                data["histories"][0].update({"history_id": "bad"})
                or data["histories"].append(deepcopy(data["histories"][0]))
            ),
            "history identifiers",
        ),
        (
            lambda data: data["histories"][0].update({"stage_id": "commit"}),
            "recourse stage",
        ),
        (
            lambda data: data["histories"][0].update({"reachable_states": ["unknown"]}),
            "unknown state",
        ),
        (
            lambda data: data["histories"][0].update({"reachable_states": ["high"]}),
            "partition states",
        ),
        (
            lambda data: data["policies"][1].update({"policy_id": "balanced-policy"}),
            "policy identifiers",
        ),
        (
            lambda data: data["policies"][0].update({"history_decisions": []}),
            "one decision",
        ),
        (lambda data: data["policies"][0]["state_outcomes"].pop(), "one outcome"),
        (
            lambda data: data["policies"][0]["state_outcomes"][0].update(
                {"recourse_status": "infeasible"}
            ),
            "disagree",
        ),
        (
            lambda data: data["policies"][0]["state_outcomes"][0].update(
                {"objective_value": float("nan")}
            ),
            "finite objective",
        ),
        (
            lambda data: data["policies"][0]["state_outcomes"][0].update(
                {
                    "feasible": False,
                    "objective_value": 1.0,
                    "recourse_status": "infeasible",
                }
            ),
            "null objective",
        ),
        (
            lambda data: data["deterministic_candidates"][1].update(
                {"candidate_id": "balanced-at-mean"}
            ),
            "candidate identifiers",
        ),
        (
            lambda data: data["deterministic_candidates"][0].update(
                {"induced_policy_id": "missing"}
            ),
            "unknown policy",
        ),
        (
            lambda data: data["deterministic_candidates"][0].update(
                {"first_stage_decision": "mismatch"}
            ),
            "first-stage decisions disagree",
        ),
        (
            lambda data: data["deterministic_candidates"][0].update(
                {"point_objective_value": float("inf")}
            ),
            "deterministic objectives",
        ),
        (
            lambda data: data["point_estimate"].update({"value": float("nan")}),
            "numeric contract",
        ),
    ],
)
def test_semantic_pathologies_fail_closed(mutation: Any, message: str) -> None:
    payload = _input()
    mutation(payload)
    with pytest.raises(InputError, match=message):
        _result(payload)


def test_strict_schemas_and_result_validator() -> None:
    input_schema = _json(CONTRACT / "schemas/input.schema.json")
    result_schema = _json(CONTRACT / "schemas/result.schema.json")
    Draft202012Validator(input_schema).validate(_input())
    result = _result()
    Draft202012Validator(result_schema).validate(result)
    validate_uncertainty_modelling_value_result(result)
    invalid = _input()
    invalid["unexpected"] = True
    assert list(Draft202012Validator(input_schema).iter_errors(invalid))
    invalid_result = deepcopy(result)
    invalid_result["unexpected"] = True
    assert list(Draft202012Validator(result_schema).iter_errors(invalid_result))
    with pytest.raises(ValueError, match="<root>"):
        validate_uncertainty_modelling_value_semantics({})
    nested = _input()
    nested["objective"]["unexpected"] = True
    with pytest.raises(ValueError, match="objective"):
        validate_uncertainty_modelling_value_semantics(nested)
    nonfinite_result = deepcopy(result)
    nonfinite_result["decomposition"]["evpi"] = float("nan")
    with pytest.raises(ValueError, match="non-finite number"):
        validate_uncertainty_modelling_value_result(nonfinite_result)


def test_input_order_and_result_copy_are_deterministic() -> None:
    baseline = value_of_uncertainty_modelling(_input())
    payload = _input()
    payload["states"].reverse()
    payload["policies"].reverse()
    payload["deterministic_candidates"].reverse()
    assert (
        value_of_uncertainty_modelling(payload).to_contract_dict()
        == baseline.to_contract_dict()
    )
    copy = baseline.to_contract_dict()
    copy["recourse_problem"]["policy_tie"].append("tampered")
    assert (
        "tampered" not in baseline.to_contract_dict()["recourse_problem"]["policy_tie"]
    )


def test_tolerance_ties_do_not_replace_exact_optimum_values() -> None:
    payload = _input()
    balanced_candidate = next(
        item
        for item in payload["deterministic_candidates"]
        if item["candidate_id"] == "balanced-at-mean"
    )
    balanced_candidate["point_objective_value"] = 4.0 + 5e-13
    balanced_policy = next(
        item for item in payload["policies"] if item["policy_id"] == "balanced-policy"
    )
    for outcome in balanced_policy["state_outcomes"]:
        outcome["objective_value"] = 8.0 + 5e-13

    result = _result(payload)
    assert result["expected_value_problem"]["candidate_tie"] == [
        "balanced-at-mean",
        "risky-at-mean",
    ]
    assert result["expected_value_problem"]["selected_candidate_id"] == "risky-at-mean"
    assert result["expected_value_problem"]["point_objective_value"] == 4.0
    assert result["recourse_problem"]["policy_tie"] == [
        "balanced-policy",
        "safe-policy",
    ]
    assert result["recourse_problem"]["selected_policy_id"] == "safe-policy"
    assert result["recourse_problem"]["value"] == 8.0
    assert result["assurance"]["objective_bound"] == 8.0
    assert result["assurance"]["optimality_gap"] == 0.0


def test_cli_and_public_experimental_discovery(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    invoked = CliRunner().invoke(
        app,
        [
            "--format",
            "json",
            "calculate-uncertainty-modelling-value",
            str(TWO_STAGE),
            "--output",
            str(output),
        ],
    )
    assert invoked.exit_code == 0, invoked.output
    payload = json.loads(invoked.stdout)
    assert payload["analysis_type"] == "uncertainty_modelling_value_result"
    assert json.loads(output.read_text(encoding="utf-8")) == payload
    assert public_uncertainty_modelling_value is value_of_uncertainty_modelling
    assert PublicUncertaintyModellingValueResult is not None


def test_cli_text_and_non_object_failure(tmp_path: Path) -> None:
    output = tmp_path / "result.txt"
    invoked = CliRunner().invoke(
        app,
        [
            "calculate-uncertainty-modelling-value",
            str(TWO_STAGE),
            "--output",
            str(output),
        ],
    )
    assert invoked.exit_code == 0
    assert "EVIU/VSS 1.0" in invoked.stdout
    invalid = tmp_path / "invalid.json"
    invalid.write_text("[]", encoding="utf-8")
    failed = CliRunner().invoke(
        app, ["calculate-uncertainty-modelling-value", str(invalid)]
    )
    assert failed.exit_code == 1
    assert "must be a JSON object" in failed.stderr


def test_methods_lazy_export_and_explicit_deferrals() -> None:
    from voiage import methods

    assert methods.value_of_uncertainty_modelling is value_of_uncertainty_modelling
    result = _result()
    assert result["assurance"]["information_acquisition_modelled"] is False
    assert result["unsupported_dispositions"]["dvss"].startswith("deferred")
    assert result["language_dispositions"]["rust"] == "not_implemented"
