"""Exact contract assurance for dependent information-source portfolios."""

# pyright: reportAny=false, reportExplicitAny=false, reportUnknownArgumentType=false, reportUnknownLambdaType=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnusedCallResult=false
# pyright: reportPrivateUsage=false

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
import pytest
from typer.testing import CliRunner

import voiage
from voiage.cli import app
from voiage.contracts.information_source_portfolio import (
    validate_information_source_portfolio_result,
)
from voiage.exceptions import InputError
from voiage.methods.information_source_portfolio import (
    _sequence_is_feasible,
    information_source_portfolio_value,
)

ROOT = Path(__file__).parents[1]
CONTRACT = ROOT / "specs/frontier/information-source-portfolio/v1"
INPUT = CONTRACT / "fixtures/normative/input.json"
EXPECTED = CONTRACT / "fixtures/normative/expected.json"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _input() -> dict[str, Any]:
    return _json(INPUT)


def test_normative_joint_portfolio_matches_independent_reference() -> None:
    result = information_source_portfolio_value(_input()).to_contract_dict()
    expected = _json(EXPECTED)
    assert result["baseline"]["value"] == pytest.approx(expected["baseline_value"])
    assert result["optimum"]["source_sequence"] == expected["selected_sequence"]
    assert result["optimum"]["gross_value"] == pytest.approx(expected["gross_value"])
    assert result["optimum"]["net_value"] == pytest.approx(expected["net_value"])
    assert [
        item["gross_marginal_value"] for item in result["conditional_marginals"]
    ] == pytest.approx(expected["conditional_marginals"])
    assert [
        item["gross_attribution"] for item in result["attribution"]
    ] == pytest.approx(expected["gross_attribution"])
    assert result["assurance"]["feasible_sequences"] == expected["feasible_sequences"]
    assert result["assurance"]["pruned_sequences"] == expected["pruned_sequences"]


def test_redundant_duplicate_has_zero_conditional_value() -> None:
    payload = _input()
    payload["constraints"]["required_coverage"] = ["clinical"]
    payload["constraints"]["max_sources"] = 2
    result = information_source_portfolio_value(payload).to_contract_dict()
    redundant = next(
        item
        for item in result["evaluated_sequences"]
        if item["source_sequence"] == ["registry", "survey"]
    )
    assert redundant["gross_value"] == pytest.approx(10.0)
    assert redundant["conditional_marginals"][1][
        "gross_marginal_value"
    ] == pytest.approx(0.0)


def test_no_procurement_beats_every_negative_net_sequence() -> None:
    payload = _input()
    for source in payload["sources"]:
        source["cost"] = 100.0
    payload["constraints"]["max_cost"] = 1000.0
    result = information_source_portfolio_value(payload).to_contract_dict()
    assert result["optimum"]["source_sequence"] == []
    assert result["optimum"]["net_value"] == pytest.approx(0.0)
    assert result["assurance"]["no_procurement_comparator_included"] is True
    assert result["assurance"]["no_procurement_subject_to_source_constraints"] is False
    assert result["assurance"]["feasible_non_empty_sequences"] == (
        result["assurance"]["feasible_sequences"] - 1
    )


def test_impossible_constraints_do_not_misreport_no_procurement_as_feasible() -> None:
    payload = _input()
    payload["constraints"]["required_coverage"] = ["unavailable-coverage"]
    with pytest.raises(InputError, match="no feasible non-empty"):
        information_source_portfolio_value(payload)


def test_complementary_sources_are_not_additive_evsi_scores() -> None:
    result = information_source_portfolio_value(_input()).to_contract_dict()
    by_sequence = {
        tuple(item["source_sequence"]): item for item in result["evaluated_sequences"]
    }
    registry = by_sequence[("registry",)]["gross_value"]
    sensor = by_sequence[("sensor",)]["gross_value"]
    joint = by_sequence[("registry", "sensor")]["gross_value"]
    assert joint > registry + sensor
    assert result["assurance"]["independent_additive_evsi_used"] is False


def test_complete_sequence_and_policy_ties_are_preserved() -> None:
    payload = _input()
    payload["sources"] = [payload["sources"][0]]
    payload["constraints"]["required_coverage"] = ["clinical"]
    payload["constraints"]["max_sources"] = 1
    payload["sources"][0]["cost"] = 0.0
    for state in payload["states"]:
        state["source_observations"] = {
            "survey": state["source_observations"]["survey"]
        }
    result = information_source_portfolio_value(payload).to_contract_dict()
    assert result["baseline"]["action_tie"] == ["act00", "act01", "act10", "act11"]
    assert result["optimum"]["source_sequence"] == ["survey"]
    assert all(len(item["action_tie"]) == 2 for item in result["optimum"]["partitions"])


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda data: data["sources"][0]["rights"].update({"status": "uncleared"}),
            "rights",
        ),
        (
            lambda data: data["states"][0]["source_observations"].pop("sensor"),
            "observation",
        ),
        (lambda data: data["states"][0].update({"probability": 0.4}), "probabilities"),
        (lambda data: data["sources"][0].update({"cost_unit": "USD"}), "unit"),
        (
            lambda data: data["value_context"].update({"cost_unit": "USD"}),
            "commensurate",
        ),
        (
            lambda data: data["states"][0]["action_values"].update(
                {"act00": float("nan")}
            ),
            "finite",
        ),
        (lambda data: None, "cycle"),
    ],
)
def test_pathologies_fail_closed(mutation: Any, message: str) -> None:
    payload = _input()
    if message == "cycle":
        payload["sources"][1]["must_precede"] = ["sensor"]
        payload["sources"][2]["must_precede"] = ["registry"]
    mutation(payload)
    with pytest.raises(InputError, match=message):
        _ = information_source_portfolio_value(payload)


def test_strict_schemas_validate_normative_contracts() -> None:
    input_schema = _json(CONTRACT / "schemas/input.schema.json")
    result_schema = _json(CONTRACT / "schemas/result.schema.json")
    Draft202012Validator(input_schema).validate(_input())
    Draft202012Validator(result_schema).validate(
        information_source_portfolio_value(_input()).to_contract_dict()
    )
    invalid = deepcopy(_input())
    invalid["unexpected"] = True
    assert list(Draft202012Validator(input_schema).iter_errors(invalid))


def test_source_order_and_input_order_do_not_change_the_optimum() -> None:
    payload = _input()
    forward = information_source_portfolio_value(payload).to_contract_dict()
    payload["sources"].reverse()
    payload["states"].reverse()
    reverse = information_source_portfolio_value(payload).to_contract_dict()
    assert forward["optimum"] == reverse["optimum"]
    assert forward["evaluated_sequences"] == reverse["evaluated_sequences"]


def test_result_copy_is_independent() -> None:
    result = information_source_portfolio_value(_input())
    first = result.to_contract_dict()
    first["optimum"]["source_sequence"].append("tampered")
    assert "tampered" not in result.to_contract_dict()["optimum"]["source_sequence"]


def test_cli_and_public_experimental_discovery(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    invoked = CliRunner().invoke(
        app,
        [
            "--format",
            "json",
            "calculate-information-source-portfolio",
            str(INPUT),
            "--output",
            str(output),
        ],
    )
    assert invoked.exit_code == 0, invoked.output
    payload = json.loads(invoked.stdout)
    assert payload["analysis_type"] == "information_source_portfolio_result"
    assert payload["method_maturity"] == "experimental"
    assert payload["optimum"]["source_sequence"] == ["registry", "sensor"]
    assert json.loads(output.read_text(encoding="utf-8")) == payload
    assert (
        voiage.information_source_portfolio_value is information_source_portfolio_value
    )


def test_cli_text_output_status_and_non_object_failure(tmp_path: Path) -> None:
    output = tmp_path / "result.txt"
    invoked = CliRunner().invoke(
        app,
        ["calculate-information-source-portfolio", str(INPUT), "--output", str(output)],
    )
    assert invoked.exit_code == 0, invoked.output
    assert f"Result saved to {output}" in invoked.stdout
    assert output.read_text(encoding="utf-8").startswith(
        "Information-source portfolio:"
    )

    invalid = tmp_path / "array.json"
    invalid.write_text("[]\n", encoding="utf-8")
    rejected = CliRunner().invoke(
        app, ["calculate-information-source-portfolio", str(invalid)]
    )
    assert rejected.exit_code == 1
    assert "must be a JSON object" in rejected.output


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("cost", 11.0, "cost"),
        ("latency", 11.0, "latency"),
        ("privacy_cost", 11.0, "privacy"),
        ("sla_probability", 0.1, "sla"),
        ("freshness_age", 11.0, "freshness"),
    ],
)
def test_each_source_feasibility_constraint_reports_its_reason(
    field: str, value: float, reason: str
) -> None:
    source = {
        "cost": 0.0,
        "latency": 0.0,
        "privacy_cost": 0.0,
        "sla_probability": 1.0,
        "freshness_age": 0.0,
        "coverage": ["required"],
        "excludes": [],
        "must_precede": [],
    }
    source[field] = value
    feasible, observed = _sequence_is_feasible(
        ["source"],
        {"source": source},
        {
            "max_cost": 10.0,
            "max_latency": 10.0,
            "max_privacy_cost": 10.0,
            "min_source_sla": 0.9,
            "max_freshness_age": 10.0,
            "required_coverage": ["required"],
        },
    )
    assert feasible is False
    assert observed == reason


def test_source_exclusivity_reports_its_reason() -> None:
    source = {
        "cost": 0.0,
        "latency": 0.0,
        "privacy_cost": 0.0,
        "sla_probability": 1.0,
        "freshness_age": 0.0,
        "coverage": ["required"],
        "excludes": ["other"],
        "must_precede": [],
    }
    other = {**source, "excludes": []}
    feasible, reason = _sequence_is_feasible(
        ["source", "other"],
        {"source": source, "other": other},
        {
            "max_cost": 10.0,
            "max_latency": 10.0,
            "max_privacy_cost": 10.0,
            "min_source_sla": 0.9,
            "max_freshness_age": 10.0,
            "required_coverage": ["required"],
        },
    )
    assert feasible is False
    assert reason == "exclusivity"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda data: data["sources"][1].update(
                {"source_id": data["sources"][0]["source_id"]}
            ),
            "source IDs",
        ),
        (
            lambda data: data["states"][1].update(
                {"state_id": data["states"][0]["state_id"]}
            ),
            "state IDs",
        ),
        (
            lambda data: data["states"][0]["action_values"].pop("act00"),
            "every action",
        ),
        (
            lambda data: data["sources"][0].update({"excludes": ["unknown"]}),
            "other declared sources",
        ),
    ],
)
def test_cross_field_semantic_pathologies_fail_closed(
    mutation: Any, message: str
) -> None:
    payload = _input()
    mutation(payload)
    with pytest.raises(InputError, match=message):
        information_source_portfolio_value(payload)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda result: result["evaluated_sequences"][0].update(
                {"resolved_value": result["baseline"]["value"] + 1.0}
            ),
            "resolved and gross",
        ),
        (
            lambda result: result["evaluated_sequences"][0].update(
                {"willingness_to_pay": 1.0}
            ),
            "willingness-to-pay",
        ),
        (
            lambda result: result["evaluated_sequences"][0].update({"net_value": 1.0}),
            "net decision-value",
        ),
        (
            lambda result: result["evaluated_sequences"][1]["conditional_marginals"][
                0
            ].update({"gross_marginal_value": 999.0}),
            "recover gross value",
        ),
        (
            lambda result: result["optimum"].update({"net_value": -1.0}),
            "not maximal",
        ),
        (
            lambda result: result["attribution"][0].update(
                {"gross_attribution": 999.0}
            ),
            "recover selected gross value",
        ),
    ],
)
def test_result_identity_validation_fails_closed(mutation: Any, message: str) -> None:
    result = information_source_portfolio_value(_input()).to_contract_dict()
    mutation(result)
    with pytest.raises(ValueError, match=message):
        validate_information_source_portfolio_result(result)
