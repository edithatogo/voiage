"""Tests for signed, social and selective-sharing information value."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import voiage
from voiage.cli import app
from voiage.contracts.signed_social_information import (
    validate_signed_social_information_input_or_raise,
    validate_signed_social_information_result,
    validate_signed_social_information_semantics,
)
from voiage.exceptions import InputError
from voiage.methods.signed_social_information import (
    SignedSocialInformationResult,
    signed_social_information_value,
)

ROOT = Path(__file__).parents[1]
FIXTURE = (
    ROOT
    / "tests/fixtures/signed_social_information"
    / "li_pozzi_harmful_private_positive_social.json"
)
CONTRACT_ROOT = ROOT / "specs/frontier/signed-social-information/v1"


def _specification() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_harmful_private_information_can_have_positive_social_value() -> None:
    result = signed_social_information_value(_specification())

    assert isinstance(result, SignedSocialInformationResult)
    payload = result.to_contract_dict()
    design = payload["designs"][1]
    assert design["selected_policy_id"] == "signal_contingent_social_policy"
    assert design["signed_values"]["by_agent"] == {
        "alice": -2.0,
        "board": 0.0,
        "public": 4.0,
    }
    assert design["signed_values"]["social"] == 2.0
    assert design["signed_values"]["by_role"]["recipient"] == -2.0
    assert design["signed_values"]["by_role"]["controller"] == 0.0
    assert design["signed_values"]["by_role"]["stakeholder"] == 4.0
    assert payload["diagnostics"]["winners"] == ["public"]
    assert payload["diagnostics"]["losers"] == ["alice"]
    assert payload["diagnostics"]["harmful_private_designs"] == [
        "selective_social_sharing"
    ]
    assert payload["diagnostics"]["information_avoidance"] == [
        {"agent_id": "alice", "design_id": "selective_social_sharing"}
    ]
    assert design["blackwell_nonnegativity"] == {
        "applicable": True,
        "checked_value": 2.0,
        "passed": True,
        "reasons_not_applicable": [],
    }
    validate_signed_social_information_result(payload)
    assert result.to_contract_dict() == payload


def test_exported_normative_fixture_is_exact() -> None:
    specification = json.loads(
        (CONTRACT_ROOT / "fixtures/normative/input.json").read_text(encoding="utf-8")
    )
    expected = json.loads(
        (CONTRACT_ROOT / "fixtures/normative/expected.json").read_text(encoding="utf-8")
    )
    assert signed_social_information_value(specification).to_contract_dict() == expected


def test_transfer_cost_and_declared_response_ledgers_are_signed() -> None:
    specification = _specification()
    design = deepcopy(specification["designs"][1])
    design.update(
        {
            "design_id": "controller_response",
            "selection_mode": "declared_response",
            "selector": "agent:board",
            "selected_policy_id": "signal_contingent_social_policy",
            "transfers": [
                {
                    "payer_agent_id": "public",
                    "recipient_agent_id": "alice",
                    "amount": 1.0,
                }
            ],
            "costs": [
                {"agent_id": "alice", "category": "privacy", "amount": 0.5},
                {"agent_id": "board", "category": "information", "amount": 0.25},
            ],
            "blackwell_assurance": None,
        }
    )
    specification["designs"].append(design)

    payload = signed_social_information_value(specification).to_contract_dict()
    evaluated = next(
        item
        for item in payload["designs"]
        if item["design_id"] == "controller_response"
    )
    assert evaluated["ledgers"]["pre_transfer"] == {
        "alice": -2.0,
        "board": 0.0,
        "public": 4.0,
    }
    assert evaluated["ledgers"]["transfer"] == {
        "alice": 1.0,
        "board": 0.0,
        "public": -1.0,
    }
    assert evaluated["ledgers"]["cost"] == {
        "alice": 0.5,
        "board": 0.25,
        "public": 0.0,
    }
    assert evaluated["ledgers"]["post_transfer"] == {
        "alice": -1.5,
        "board": -0.25,
        "public": 3.0,
    }
    assert evaluated["signed_values"]["social"] == 1.5
    assert evaluated["blackwell_nonnegativity"]["applicable"] is False
    assert (
        "selection_mode_not_centralized"
        in evaluated["blackwell_nonnegativity"]["reasons_not_applicable"]
    )


def test_negative_social_value_is_retained_without_clipping() -> None:
    specification = _specification()
    specification["designs"][1]["costs"] = [
        {"agent_id": "alice", "category": "information", "amount": 10.0}
    ]
    result = signed_social_information_value(specification).to_contract_dict()
    informed = result["designs"][1]
    assert informed["signed_values"]["social"] == -8.0
    assert informed["signed_values"]["clipped_at_zero"] is False
    assert result["assurance"]["negative_values_clipped"] is False
    assert result["optimum"]["selected_design_id"] == "no_sharing"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda data: data["welfare"].update(cardinal_comparability="undeclared"),
            "cardinal",
        ),
        (lambda data: data["worlds"][0].update(probability=0.7), "sum to one"),
        (lambda data: data["policies"][1]["decisions"].pop(), "observations"),
        (
            lambda data: data["designs"][1].update(rights_receipt_ids=[]),
            "rights receipt",
        ),
        (
            lambda data: data["designs"][1]["blackwell_assurance"].update(
                baseline_catalog_embedded=False
            ),
            "Blackwell",
        ),
    ],
)
def test_strict_contract_rejects_incomplete_or_false_assurance(
    mutation: Any, message: str
) -> None:
    specification = _specification()
    mutation(specification)
    with pytest.raises((InputError, ValueError), match=message):
        signed_social_information_value(specification)


def test_denied_consent_marks_design_infeasible_without_hiding_it() -> None:
    specification = _specification()
    specification["receipts"][0]["consent_status"] = "denied"
    specification["designs"][1]["blackwell_assurance"] = None

    payload = signed_social_information_value(specification).to_contract_dict()
    design = payload["designs"][1]
    assert design["feasible"] is False
    assert design["infeasibility_reasons"] == ["consent_denied:alice"]
    assert payload["optimum"]["selected_design_id"] == "no_sharing"


def test_blackwell_check_excludes_infeasible_and_transfer_cost_designs() -> None:
    denied = _specification()
    denied["receipts"][0]["consent_status"] = "denied"
    denied_check = signed_social_information_value(denied).to_contract_dict()[
        "designs"
    ][1]["blackwell_nonnegativity"]
    assert denied_check["applicable"] is False
    assert "design_infeasible" in denied_check["reasons_not_applicable"]

    comparator_denied = _specification()
    comparator_denied["receipts"].append(
        {
            "receipt_id": "baseline-denied",
            "subject_agent_id": "public",
            "consent_status": "denied",
            "purpose": comparator_denied["purpose"],
            "legal_basis": "synthetic_denial",
            "data_scope": "clinical_signal",
        }
    )
    comparator_denied["designs"][0]["rights_receipt_ids"] = ["baseline-denied"]
    comparator_denied_check = signed_social_information_value(
        comparator_denied
    ).to_contract_dict()["designs"][1]["blackwell_nonnegativity"]
    assert comparator_denied_check["applicable"] is False
    assert "comparator_infeasible" in comparator_denied_check["reasons_not_applicable"]

    costed = _specification()
    costed["designs"][1]["costs"] = [
        {"agent_id": "alice", "category": "information", "amount": 0.5}
    ]
    costed_check = signed_social_information_value(costed).to_contract_dict()[
        "designs"
    ][1]["blackwell_nonnegativity"]
    assert costed_check["applicable"] is False
    assert "design_has_transfers_or_costs" in costed_check["reasons_not_applicable"]

    comparator_costed = _specification()
    comparator_costed["designs"][0]["costs"] = [
        {"agent_id": "board", "category": "information", "amount": 0.25}
    ]
    comparator_check = signed_social_information_value(
        comparator_costed
    ).to_contract_dict()["designs"][1]["blackwell_nonnegativity"]
    assert comparator_check["applicable"] is False
    assert (
        "comparator_has_transfers_or_costs"
        in comparator_check["reasons_not_applicable"]
    )


def test_verified_equilibrium_is_a_catalog_not_a_general_solver() -> None:
    specification = _specification()
    specification["designs"][1].update(
        selection_mode="verified_finite_equilibrium",
        selector="agent:board",
        selected_policy_id="signal_contingent_social_policy",
        equilibrium_receipt={
            "solution_concept": "declared_finite_nash",
            "verification_method": "complete_catalog_best_response_check",
            "verified_policy_ids": ["signal_contingent_social_policy"],
        },
        blackwell_assurance=None,
    )
    payload = signed_social_information_value(specification).to_contract_dict()
    assert payload["designs"][1]["selection_mode"] == "verified_finite_equilibrium"
    assert payload["assurance"]["general_game_solver_used"] is False


def test_public_api_cli_and_input_semantics(tmp_path: Path) -> None:
    specification = _specification()
    validate_signed_social_information_semantics(specification)
    result = voiage.signed_social_information_value(specification)
    assert result.to_contract_dict()["method_maturity"] == "experimental"

    output = tmp_path / "result.json"
    invocation = CliRunner().invoke(
        app,
        [
            "--format",
            "json",
            "calculate-signed-social-information",
            str(FIXTURE),
            "--output",
            str(output),
        ],
    )
    assert invocation.exit_code == 0, invocation.output
    assert json.loads(invocation.output)["optimum"]["social_value"] == 2.0
    assert json.loads(output.read_text(encoding="utf-8"))["analysis_id"].startswith(
        "li-pozzi"
    )


def test_cli_rejects_non_object_specification(tmp_path: Path) -> None:
    specification = tmp_path / "signed-social-list.json"
    specification.write_text("[]\n", encoding="utf-8")

    invocation = CliRunner().invoke(
        app,
        ["calculate-signed-social-information", str(specification)],
    )

    assert invocation.exit_code == 1
    assert (
        "Signed-social information specification must be a JSON object."
        in invocation.output
    )


def test_cli_reports_saved_text_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("voiage.cli._should_echo_status_messages", lambda: True)
    output = tmp_path / "signed-social-result.txt"

    invocation = CliRunner().invoke(
        app,
        [
            "calculate-signed-social-information",
            str(FIXTURE),
            "--output",
            str(output),
        ],
    )

    assert invocation.exit_code == 0, invocation.output
    assert f"Result saved to {output}" in invocation.output
    assert output.read_text(encoding="utf-8").startswith(
        "Signed-social information value:"
    )


def _apply_semantic_pathology(name: str, data: dict[str, Any]) -> None:
    """Apply one schema-valid cross-record pathology."""
    if name == "nonfinite":
        data["worlds"][0]["action_utilities"]["safe"]["alice"] = float("nan")
    elif name == "duplicate_agent":
        data["agents"][1]["agent_id"] = "alice"
    elif name == "weights_mismatch":
        data["welfare"]["weights"].pop("board")
    elif name == "zero_weights":
        data["welfare"]["weights"] = dict.fromkeys(data["welfare"]["weights"], 0.0)
    elif name == "unknown_source":
        data["topology"]["source_agent_id"] = "unknown"
    elif name == "bad_topology_recipients":
        data["topology"]["eligible_recipients"] = ["unknown"]
    elif name == "duplicate_world":
        data["worlds"][1]["world_id"] = data["worlds"][0]["world_id"]
    elif name == "missing_action":
        data["worlds"][0]["action_utilities"].pop("restrict")
    elif name == "missing_agent_utility":
        data["worlds"][0]["action_utilities"]["safe"].pop("board")
    elif name == "duplicate_policy":
        data["policies"][1]["policy_id"] = data["policies"][0]["policy_id"]
    elif name == "unknown_decision_agent":
        data["policies"][0]["decision_agent_id"] = "unknown"
    elif name == "duplicate_observation":
        data["policies"][1]["decisions"][1]["observation"] = "negative"
    elif name == "unknown_action":
        data["policies"][0]["decisions"][0]["action_id"] = "unknown"
    elif name == "duplicate_receipt":
        data["receipts"][1]["receipt_id"] = data["receipts"][0]["receipt_id"]
    elif name == "unknown_receipt_subject":
        data["receipts"][0]["subject_agent_id"] = "unknown"
    elif name == "purpose_mismatch":
        data["receipts"][0]["purpose"] = "different_purpose"
    elif name == "duplicate_design":
        data["designs"][1]["design_id"] = data["designs"][0]["design_id"]
    elif name == "unknown_baseline":
        data["baseline_design_id"] = "unknown"
    elif name == "baseline_comparator":
        data["designs"][0]["comparator_design_id"] = "selective_social_sharing"
    elif name == "baseline_recipient_mismatch":
        data["designs"][0]["recipients"] = ["alice"]
    elif name == "unknown_comparator":
        data["designs"][1]["comparator_design_id"] = "unknown"
        data["designs"][1]["blackwell_assurance"] = None
    elif name == "self_comparator":
        data["designs"][1]["comparator_design_id"] = "selective_social_sharing"
        data["designs"][1]["blackwell_assurance"] = None
    elif name == "ineligible_recipient":
        data["designs"][1]["recipients"] = ["board"]
    elif name == "unknown_policy":
        data["designs"][1]["policy_ids"] = ["unknown"]
    elif name == "unshared_informed":
        data["designs"][0]["policy_ids"] = ["signal_contingent_social_policy"]
    elif name == "bad_selector":
        data["designs"][1]["selector"] = "agent:unknown"
    elif name == "centralized_selected":
        data["designs"][1]["selected_policy_id"] = "safe_without_information"
    elif name == "fixed_without_selected":
        data["designs"][1].update(selection_mode="fixed", selected_policy_id=None)
    elif name == "equilibrium_without_receipt":
        data["designs"][1].update(
            selection_mode="verified_finite_equilibrium",
            selected_policy_id="signal_contingent_social_policy",
            blackwell_assurance=None,
        )
    elif name == "equilibrium_outside_catalog":
        data["designs"][1].update(
            selection_mode="verified_finite_equilibrium",
            selected_policy_id="signal_contingent_social_policy",
            blackwell_assurance=None,
            equilibrium_receipt={
                "solution_concept": "finite",
                "verification_method": "complete_catalog_best_response_check",
                "verified_policy_ids": ["signal_contingent_social_policy", "unknown"],
            },
        )
    elif name == "receipt_on_centralized":
        data["designs"][1]["equilibrium_receipt"] = {
            "solution_concept": "finite",
            "verification_method": "complete_catalog_best_response_check",
            "verified_policy_ids": ["signal_contingent_social_policy"],
        }
    elif name == "unknown_transfer_agent":
        data["designs"][1]["transfers"] = [
            {"payer_agent_id": "unknown", "recipient_agent_id": "alice", "amount": 1.0}
        ]
    elif name == "self_transfer":
        data["designs"][1]["transfers"] = [
            {"payer_agent_id": "alice", "recipient_agent_id": "alice", "amount": 1.0}
        ]
    elif name == "unknown_cost_agent":
        data["designs"][1]["costs"] = [
            {"agent_id": "unknown", "category": "other_declared", "amount": 1.0}
        ]
    elif name == "unknown_rights_receipt":
        data["designs"][1]["rights_receipt_ids"] = ["unknown"]
    elif name == "missing_controller_receipt":
        data["designs"][1]["rights_receipt_ids"] = ["alice-purpose-consent"]
    elif name == "blackwell_comparator_mismatch":
        data["designs"][1]["blackwell_assurance"]["refines_design_id"] = (
            "selective_social_sharing"
        )
    elif name == "blackwell_catalog_not_embedded":
        data["designs"][1]["policy_ids"] = ["signal_contingent_social_policy"]
    else:  # pragma: no cover - test table owns the names
        raise AssertionError(name)


@pytest.mark.parametrize(
    "pathology",
    [
        "nonfinite",
        "duplicate_agent",
        "weights_mismatch",
        "zero_weights",
        "unknown_source",
        "bad_topology_recipients",
        "duplicate_world",
        "missing_action",
        "missing_agent_utility",
        "duplicate_policy",
        "unknown_decision_agent",
        "duplicate_observation",
        "unknown_action",
        "duplicate_receipt",
        "unknown_receipt_subject",
        "purpose_mismatch",
        "duplicate_design",
        "unknown_baseline",
        "baseline_comparator",
        "baseline_recipient_mismatch",
        "unknown_comparator",
        "self_comparator",
        "ineligible_recipient",
        "unknown_policy",
        "unshared_informed",
        "bad_selector",
        "centralized_selected",
        "fixed_without_selected",
        "equilibrium_without_receipt",
        "equilibrium_outside_catalog",
        "receipt_on_centralized",
        "unknown_transfer_agent",
        "self_transfer",
        "unknown_cost_agent",
        "unknown_rights_receipt",
        "missing_controller_receipt",
        "blackwell_comparator_mismatch",
        "blackwell_catalog_not_embedded",
    ],
)
def test_cross_record_semantic_pathologies_fail_closed(pathology: str) -> None:
    specification = _specification()
    _apply_semantic_pathology(pathology, specification)
    with pytest.raises((InputError, ValueError)):
        signed_social_information_value(specification)


def test_agent_selector_blackwell_and_inapplicability_reasons() -> None:
    specification = _specification()
    for design in specification["designs"]:
        design["selector"] = "agent:alice"
    payload = signed_social_information_value(specification).to_contract_dict()
    informed = payload["designs"][1]
    assert informed["blackwell_nonnegativity"]["applicable"] is True
    assert informed["blackwell_nonnegativity"]["checked_value"] == 0.0

    specification = _specification()
    specification["designs"][0].update(
        selection_mode="fixed", selected_policy_id="safe_without_information"
    )
    specification["designs"][1]["blackwell_assurance"] = None
    payload = signed_social_information_value(specification).to_contract_dict()
    assert (
        "comparator_not_centralized"
        in payload["designs"][1]["blackwell_nonnegativity"]["reasons_not_applicable"]
    )


def test_result_semantics_and_public_error_adapter_fail_closed() -> None:
    specification = _specification()
    result = signed_social_information_value(specification).to_contract_dict()
    broken_ledger = deepcopy(result)
    broken_ledger["designs"][1]["ledgers"]["post_transfer"]["alice"] += 1.0
    with pytest.raises(ValueError, match="ledgers disagree"):
        validate_signed_social_information_result(broken_ledger)

    broken_value = deepcopy(result)
    broken_value["designs"][1]["signed_values"]["by_agent"]["alice"] += 1.0
    with pytest.raises(ValueError, match="comparator ledger"):
        validate_signed_social_information_result(broken_value)

    chained_specification = _specification()
    chained = deepcopy(chained_specification["designs"][1])
    chained.update(
        design_id="chained_sharing",
        comparator_design_id="selective_social_sharing",
        blackwell_assurance=None,
    )
    chained_specification["designs"].append(chained)
    chained_result = signed_social_information_value(
        chained_specification
    ).to_contract_dict()
    chained_result["designs"][2]["signed_values"]["by_agent"]["alice"] = 1.0
    with pytest.raises(ValueError, match="comparator ledger"):
        validate_signed_social_information_result(chained_result)

    broken_social = deepcopy(result)
    broken_social["designs"][1]["social_post_transfer"] += 1.0
    with pytest.raises(ValueError, match="welfare ledger"):
        validate_signed_social_information_result(broken_social)

    broken_social_pre = deepcopy(result)
    broken_social_pre["designs"][1]["social_pre_transfer"] += 1.0
    with pytest.raises(ValueError, match="welfare ledger"):
        validate_signed_social_information_result(broken_social_pre)

    broken_signed_social = deepcopy(result)
    broken_signed_social["designs"][1]["signed_values"]["social"] += 1.0
    with pytest.raises(ValueError, match="signed social value"):
        validate_signed_social_information_result(broken_signed_social)

    duplicate_design = deepcopy(result)
    duplicate_design["designs"][1]["design_id"] = duplicate_design["baseline"][
        "design_id"
    ]
    with pytest.raises(ValueError, match="identifiers must be unique"):
        validate_signed_social_information_result(duplicate_design)

    unknown_baseline = deepcopy(result)
    unknown_baseline["baseline"]["design_id"] = "unknown"
    with pytest.raises(ValueError, match="baseline must match"):
        validate_signed_social_information_result(unknown_baseline)

    mismatched_baseline = deepcopy(result)
    mismatched_baseline["baseline"]["selected_policy_id"] = (
        "signal_contingent_social_policy"
    )
    with pytest.raises(ValueError, match="baseline must match"):
        validate_signed_social_information_result(mismatched_baseline)

    baseline_comparator = deepcopy(result)
    baseline_comparator["baseline"]["comparator_design_id"] = "selective_social_sharing"
    baseline_comparator["designs"][0]["comparator_design_id"] = (
        "selective_social_sharing"
    )
    with pytest.raises(ValueError, match="must not declare a comparator"):
        validate_signed_social_information_result(baseline_comparator)

    incomplete_ledger = deepcopy(result)
    incomplete_ledger["designs"][1]["ledgers"]["cost"].pop("board")
    with pytest.raises(ValueError, match="exactly the welfare agents"):
        validate_signed_social_information_result(incomplete_ledger)

    unknown_result_comparator = deepcopy(result)
    unknown_result_comparator["designs"][1]["comparator_design_id"] = "unknown"
    unknown_result_comparator["designs"][1]["signed_values"]["comparator_design_id"] = (
        "unknown"
    )
    with pytest.raises(ValueError, match="unknown comparator"):
        validate_signed_social_information_result(unknown_result_comparator)

    incomplete_signed_values = deepcopy(result)
    incomplete_signed_values["designs"][1]["signed_values"]["by_agent"].pop("board")
    with pytest.raises(ValueError, match="exactly the welfare agents"):
        validate_signed_social_information_result(incomplete_signed_values)

    extra_nested = deepcopy(result)
    extra_nested["designs"][1]["ledgers"]["unexpected"] = {}
    with pytest.raises(ValueError, match="Additional properties"):
        validate_signed_social_information_result(extra_nested)

    invalid = _specification()
    invalid["unexpected"] = True
    with pytest.raises(InputError):
        validate_signed_social_information_input_or_raise(invalid)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["designs"][1]["signed_values"]["by_role"].update(
            recipient=999.0
        ),
        lambda value: value["designs"][1].update(
            selected_policy_id="safe_without_information"
        ),
        lambda value: value["designs"][1].update(
            policy_tie=["safe_without_information"]
        ),
        lambda value: value["optimum"].update(selected_design_id="no_sharing"),
        lambda value: value["optimum"].update(social_value=999.0),
        lambda value: value["diagnostics"].update(winners=["alice"]),
        lambda value: value["designs"][1]["blackwell_nonnegativity"].update(
            checked_value=999.0
        ),
        lambda value: value["assurance"].update(worlds_evaluated=999),
        lambda value: value["agent_roles"].pop("public"),
        lambda value: value["optimum"]["tie_policy"].update(absolute_tolerance=-1.0),
        lambda value: value["designs"][1]["policy_selector_values"].pop(
            "safe_without_information"
        ),
        lambda value: (
            value["designs"][1].update(policies_evaluated=["safe_without_information"]),
            value["designs"][1].update(
                policy_selector_values={"safe_without_information": 0.0}
            ),
        ),
        lambda value: value["designs"][1].update(
            feasible=False, infeasibility_reasons=[]
        ),
        lambda value: value["designs"][1]["signed_values"].update(
            comparator_design_id=None
        ),
        lambda value: value["designs"][1].update(policy_switch=False),
        lambda value: value["designs"][0]["blackwell_nonnegativity"].update(
            checked_value=0.0
        ),
        lambda value: value["designs"][1]["blackwell_nonnegativity"].update(
            applicable=False
        ),
        lambda value: value["optimum"]["feasible_design_values"].update(
            selective_social_sharing=999.0
        ),
    ],
)
def test_result_validator_rejects_derived_surface_drift(mutation: Any) -> None:
    result = signed_social_information_value(_specification()).to_contract_dict()
    mutation(result)
    with pytest.raises(ValueError):
        validate_signed_social_information_result(result)
