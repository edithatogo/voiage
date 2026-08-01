"""Contract and numerical assurance for issue #597 belief-state VOI."""

# pyright: reportAny=false, reportExplicitAny=false, reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportMissingModuleSource=false, reportUnknownLambdaType=false
# pyright: reportUnusedCallResult=false

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
import pytest
from typer.testing import CliRunner

import voiage
from voiage import methods
from voiage.cli import app
from voiage.exceptions import InputError
import voiage.methods.belief_state_information as belief_module
from voiage.methods.belief_state_information import belief_state_information_value

ROOT = Path(__file__).parents[1]
CONTRACT = ROOT / "specs/frontier/belief-state-information/v1"
INPUT = CONTRACT / "fixtures/normative/input.json"
EXPECTED = CONTRACT / "fixtures/normative/expected.json"


def _input() -> dict[str, Any]:
    return json.loads(INPUT.read_text(encoding="utf-8"))


def _result(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    return belief_state_information_value(payload or _input()).to_contract_dict()


def test_portable_schemas_and_normative_fixture() -> None:
    input_schema = json.loads(
        (CONTRACT / "schemas/belief-state-information-input.schema.json").read_text()
    )
    result_schema = json.loads(
        (CONTRACT / "schemas/belief-state-information-result.schema.json").read_text()
    )
    Draft202012Validator(input_schema).validate(_input())
    result = _result()
    Draft202012Validator(result_schema).validate(result)
    assert result == json.loads(EXPECTED.read_text(encoding="utf-8"))


def test_nonmyopic_intervention_value_and_matched_comparator() -> None:
    result = _result()
    values = result["values"]
    assert values["closed_loop_gross"] == pytest.approx(7.0)
    assert values["expected_sensing_cost"] == pytest.approx(0.5)
    assert values["closed_loop_net"] == pytest.approx(6.5)
    assert values["no_information"] == pytest.approx(0.0)
    assert values["gross_information_value"] == pytest.approx(7.0)
    assert values["net_information_value"] == pytest.approx(6.5)
    assert values["myopic_information_value"] == pytest.approx(0.0)
    assert values["nonmyopic_information_value"] == pytest.approx(6.5)
    assert values["nonmyopic_minus_myopic"] == pytest.approx(6.5)
    assert values["fully_observed_value"] == pytest.approx(20.0)
    assert values["partial_observability_regret"] == pytest.approx(13.0)

    root = result["policy_tree"]
    assert root["control_choice_tie"] == ["probe"]
    assert root["selected_control"] == "probe"
    assert root["sensor_choice_tie"] == ["diagnostic"]
    assert root["selected_sensor"] == "diagnostic"
    assert [branch["observation_id"] for branch in root["branches"]] == [
        "negative",
        "positive",
    ]
    assert root["chronology"] == [
        "control",
        "transition",
        "observe",
        "update",
    ]


def test_conditional_sensing_null_sensor_and_martingale_assurance() -> None:
    result = _result()
    conditional = result["conditional_sensing_values"]
    probe = next(item for item in conditional if item["control_action_id"] == "probe")
    diagnostic = next(
        item for item in probe["sensors"] if item["sensor_id"] == "diagnostic"
    )
    null = next(item for item in probe["sensors"] if item["sensor_id"] == "none")
    assert diagnostic["gross_value"] == pytest.approx(8.0)
    assert diagnostic["net_value"] == pytest.approx(7.5)
    assert diagnostic["net_increment_vs_null"] == pytest.approx(7.5)
    assert null["gross_value"] == pytest.approx(0.0)
    assert null["net_value"] == pytest.approx(0.0)

    assurance = result["assurance"]
    assert assurance["posterior_martingale_verified"] is True
    assert assurance["null_sensor_reduction_verified"] is True
    assert assurance["no_information_reduction_verified"] is True
    assert assurance["exact_enumeration"] is True
    assert assurance["approximation_used"] is False
    assert assurance["complete_ties_reported"] is True
    assert assurance["action_dependent_learning"] is True
    assert assurance["dual_control_diagnostic"] is True
    assert assurance["unique_additive_dual_control_value_claimed"] is False
    assert result["language_dispositions"] == {
        "python": "executable-experimental",
        "rust": "unsupported",
        "r": "unsupported",
        "julia": "unsupported",
        "mojo": "external-boundary",
    }


def test_complete_ties_and_deterministic_permutation() -> None:
    tied = _input()
    tied["horizon"] = 1
    tied["constraints"]["allowed_control_action_ids_by_stage"].pop("1")
    result = _result(tied)
    assert result["policy_tree"]["control_choice_tie"] == [
        "choose_bad",
        "choose_good",
        "wait",
    ]
    assert result["policy_tree"]["selected_control"] == "choose_bad"

    permuted = _input()
    permuted["latent_states"].reverse()
    permuted["control_actions"].reverse()
    permuted["sensors"].reverse()
    assert _result(permuted)["values"] == _result()["values"]


def test_horizon_curve_stopping_and_exact_bounds() -> None:
    result = _result()
    assert result["value_by_horizon"] == [
        {
            "horizon": 1,
            "closed_loop_net": pytest.approx(0.0),
            "no_information": pytest.approx(0.0),
            "net_information_value": pytest.approx(0.0),
        },
        {
            "horizon": 2,
            "closed_loop_net": pytest.approx(6.5),
            "no_information": pytest.approx(0.0),
            "net_information_value": pytest.approx(6.5),
        },
    ]
    assert result["stopping"] == {
        "kind": "fixed_horizon",
        "reason": "horizon_reached",
        "stage": 2,
    }
    assert result["approximation_bounds"] == {
        "lower": pytest.approx(6.5),
        "upper": pytest.approx(6.5),
        "gap": pytest.approx(0.0),
    }


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda value: value["latent_states"][0].update(initial_probability=0.7),
            "sum to one",
        ),
        (lambda value: value.update(horizon=0), "horizon"),
        (
            lambda value: value["transition_model"]["probe"]["good"].pop("bad"),
            "all latent states",
        ),
        (lambda value: value["sensors"][0].update(cost=-1), "nonnegative"),
        (
            lambda value: value["sensors"][1].update(null_sensor=True),
            "exactly one null",
        ),
        (
            lambda value: value["sensors"][0]["likelihood_by_control"]["probe"][
                "good"
            ].update(positive=0.8),
            "sum to one",
        ),
    ],
)
def test_invalid_models_fail_closed(mutation: object, match: str) -> None:
    payload = _input()
    mutation(payload)  # type: ignore[operator]
    with pytest.raises(InputError, match=match):
        belief_state_information_value(payload)


def test_unknown_fields_nonfinite_values_and_chronology_fail_closed() -> None:
    unknown = _input()
    unknown["surprise"] = True
    with pytest.raises(InputError, match="unknown fields"):
        belief_state_information_value(unknown)

    nonfinite = _input()
    nonfinite["rewards"]["probe"]["good"] = float("inf")
    with pytest.raises(InputError, match="finite"):
        belief_state_information_value(nonfinite)

    chronology = _input()
    chronology["chronology"] = ["observe", "control", "transition", "update"]
    with pytest.raises(InputError, match="chronology"):
        belief_state_information_value(chronology)


def test_schema_rejects_open_envelopes() -> None:
    input_schema = json.loads(
        (CONTRACT / "schemas/belief-state-information-input.schema.json").read_text()
    )
    result_schema = json.loads(
        (CONTRACT / "schemas/belief-state-information-result.schema.json").read_text()
    )
    invalid_input = _input()
    invalid_input["surprise"] = True
    assert list(Draft202012Validator(input_schema).iter_errors(invalid_input))
    invalid_result = _result()
    invalid_result["values"] = {}
    assert list(Draft202012Validator(result_schema).iter_errors(invalid_result))


def test_public_api_cli_and_maturity_boundary(tmp_path: Path) -> None:
    assert voiage.belief_state_information_value is belief_state_information_value
    assert methods.belief_state_information_value is belief_state_information_value

    run = CliRunner().invoke(
        app,
        ["--format", "json", "calculate-belief-state-information", str(INPUT)],
    )
    assert run.exit_code == 0, run.output
    payload = json.loads(run.stdout)
    assert payload["analysis_type"] == "belief_state_information_result"
    assert payload["method_maturity"] == "experimental"

    output = tmp_path / "result.json"
    saved = CliRunner().invoke(
        app,
        [
            "--format",
            "json",
            "calculate-belief-state-information",
            str(INPUT),
            "--output",
            str(output),
        ],
    )
    assert saved.exit_code == 0, saved.output
    assert json.loads(output.read_text()) == json.loads(saved.stdout)

    bad = tmp_path / "bad.json"
    bad.write_text("[]", encoding="utf-8")
    failed = CliRunner().invoke(app, ["calculate-belief-state-information", str(bad)])
    assert failed.exit_code == 1
    assert "must be an object" in failed.output


def test_result_copy_is_independent() -> None:
    result = belief_state_information_value(_input())
    first = result.to_contract_dict()
    first["values"]["closed_loop_net"] = -999
    assert result.to_contract_dict()["values"]["closed_loop_net"] == pytest.approx(6.5)


def test_no_information_and_null_sensor_reductions() -> None:
    payload = _input()
    payload["constraints"]["allowed_sensor_ids_by_control"] = {
        action["action_id"]: ["none"] for action in payload["control_actions"]
    }
    result = _result(payload)
    assert result["values"]["closed_loop_net"] == pytest.approx(
        result["values"]["no_information"]
    )
    assert result["values"]["net_information_value"] == pytest.approx(0.0)


def test_minimization_direction_uses_declared_objective_consistently() -> None:
    payload = deepcopy(_input())
    payload["objective_direction"] = "minimize"
    for action in payload["rewards"].values():
        for state_id, value in action.items():
            action[state_id] = -value
    result = _result(payload)
    assert result["values"]["net_information_value"] == pytest.approx(6.5)
    assert result["values"]["closed_loop_net"] == pytest.approx(-6.5)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda value: value.pop("analysis_id"), "missing fields"),
        (lambda value: value.update(schema_version="2"), "schema_version"),
        (lambda value: value.update(analysis_type="other"), "analysis_type"),
        (lambda value: value.update(method_maturity="stable"), "method_maturity"),
        (lambda value: value.update(analysis_id=""), "analysis_id"),
        (
            lambda value: value.update(objective_direction="sideways"),
            "objective_direction",
        ),
        (lambda value: value.update(policy_class="open_loop"), "policy_class"),
        (lambda value: value.update(stopping={}), "fixed_horizon"),
        (lambda value: value.update(horizon=True), "integer"),
        (lambda value: value.update(horizon=13), "between 1 and 12"),
        (lambda value: value.update(discount_factor="unknown"), "must be numeric"),
        (lambda value: value.update(discount_factor=0), "discount_factor"),
        (lambda value: value.update(tolerances={}), "tolerances must declare"),
        (lambda value: value["tolerances"].update(absolute_tie=-1), "nonnegative"),
        (lambda value: value.update(latent_states=[]), "non-empty array"),
        (
            lambda value: value["latent_states"][0].update(state_id=""),
            "non-empty string",
        ),
        (
            lambda value: value["latent_states"].append(
                deepcopy(value["latent_states"][0])
            ),
            "unique",
        ),
        (
            lambda value: (
                value["latent_states"][0].update(initial_probability=-0.1),
                value["latent_states"][1].update(initial_probability=1.1),
            ),
            "between zero and one",
        ),
        (lambda value: value["transition_model"].pop("wait"), "all control actions"),
        (lambda value: value["transition_model"].update(wait=[]), "must be an object"),
        (
            lambda value: value["transition_model"]["wait"].pop("good"),
            "all latent states",
        ),
        (
            lambda value: value["transition_model"]["wait"]["bad"].update(
                bad=1.1, good=-0.1
            ),
            "between zero and one",
        ),
        (lambda value: value["rewards"].pop("wait"), "all control actions"),
        (lambda value: value["rewards"]["wait"].pop("good"), "all latent states"),
        (lambda value: value["sensors"][0].update(extra=True), "sensor fields"),
        (lambda value: value["sensors"][0].update(cost=1), "null sensor cost"),
        (
            lambda value: value["sensors"][1]["likelihood_by_control"].pop("wait"),
            "all control actions",
        ),
        (
            lambda value: value["sensors"][1]["likelihood_by_control"]["wait"].pop(
                "good"
            ),
            "all latent states",
        ),
        (
            lambda value: value["sensors"][0]["likelihood_by_control"]["wait"][
                "good"
            ].update(negative=0.6, positive=0.4),
            "state independent",
        ),
        (lambda value: value["constraints"].update(extra=True), "constraints fields"),
        (
            lambda value: value["constraints"][
                "allowed_control_action_ids_by_stage"
            ].pop("1"),
            "cover every stage",
        ),
        (
            lambda value: value["constraints"][
                "allowed_control_action_ids_by_stage"
            ].update({"0": []}),
            "non-empty known unique",
        ),
        (
            lambda value: value["constraints"]["allowed_sensor_ids_by_control"].pop(
                "wait"
            ),
            "cover all control actions",
        ),
        (
            lambda value: value["constraints"]["allowed_sensor_ids_by_control"].update(
                wait=["diagnostic"]
            ),
            "include null",
        ),
    ],
)
def test_all_strict_contract_boundaries_fail_closed(
    mutation: object, match: str
) -> None:
    payload = _input()
    mutation(payload)  # type: ignore[operator]
    with pytest.raises(InputError, match=match):
        belief_state_information_value(payload)


def test_zero_probability_observation_and_action_independence_are_exact() -> None:
    payload = _input()
    for sensor in payload["sensors"]:
        for by_state in sensor["likelihood_by_control"].values():
            for probabilities in by_state.values():
                probabilities.update(negative=1.0, positive=0.0)
    result = _result(payload)
    assert result["assurance"]["action_dependent_learning"] is False
    assert result["policy_tree"]["branches"][0]["observation_id"] == "negative"


def test_action_dependent_transition_is_reported() -> None:
    payload = _input()
    payload["transition_model"]["probe"] = {
        "bad": {"bad": 0.0, "good": 1.0},
        "good": {"bad": 1.0, "good": 0.0},
    }
    assert _result(payload)["assurance"]["action_dependent_transition"] is True


def test_defensive_observation_and_value_invariants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = belief_module._validate_and_build(_input())
    model.likelihood["none"]["wait"]["bad"] = {"negative": 0.4, "positive": 0.4}
    model.likelihood["none"]["wait"]["good"] = {"negative": 0.4, "positive": 0.4}
    with pytest.raises(ArithmeticError, match="branches must sum"):
        belief_module._observe(model, model.initial_belief, "wait", "none")

    valid_model = belief_module._validate_and_build(_input())
    monkeypatch.setattr(
        belief_module,
        "_adaptive",
        lambda *_args: belief_module._Evaluation(-1.0, -1.0, 0.0, {}),
    )
    with pytest.raises(ArithmeticError, match="information value is negative"):
        belief_module._evaluate(valid_model)
