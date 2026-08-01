"""Contract and runtime assurance for issue #570."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
import pytest
from typer.testing import CliRunner

import voiage
from voiage.cli import app
from voiage.contracts.risk_sensitive_voi import (
    RISK_SENSITIVE_VOI_INPUT_SCHEMA_V1,
    RISK_SENSITIVE_VOI_RESULT_SCHEMA_V1,
    validate_risk_sensitive_voi_semantics,
)
from voiage.exceptions import InputError
from voiage.methods.risk_sensitive_voi import (
    RiskSensitiveVoiResult,
    risk_sensitive_constrained_voi,
)

ROOT = Path(__file__).parents[1]
CONTRACT = ROOT / "specs/frontier/risk-sensitive-constrained-voi/v1"
INPUT = CONTRACT / "fixtures/normative/input.json"
EXPECTED = CONTRACT / "fixtures/normative/expected.json"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _input() -> dict[str, Any]:
    return _json(INPUT)


def test_portable_schemas_match_runtime_and_validate_normative_fixture() -> None:
    input_schema = _json(CONTRACT / "schemas/risk-sensitive-voi-input.schema.json")
    result_schema = _json(CONTRACT / "schemas/risk-sensitive-voi-result.schema.json")
    Draft202012Validator.check_schema(input_schema)
    Draft202012Validator.check_schema(result_schema)
    assert input_schema == RISK_SENSITIVE_VOI_INPUT_SCHEMA_V1
    assert result_schema == RISK_SENSITIVE_VOI_RESULT_SCHEMA_V1
    Draft202012Validator(input_schema).validate(_input())
    Draft202012Validator(result_schema).validate(_json(EXPECTED))
    validate_risk_sensitive_voi_semantics(_input())


def test_exact_expected_value_contract_reconciles_risk_and_constraints() -> None:
    result = risk_sensitive_constrained_voi(_input()).to_contract_dict()
    assert result == _json(EXPECTED)
    assert result["baseline"]["selected_policy_id"] == "steady"
    assert result["perfect_information"]["selected_policy_by_state"] == {
        "high": "adaptive",
        "low": "steady",
        "mid": "expansive",
    }
    assert result["value"] == {
        "gross": pytest.approx(2.1),
        "information_cost": 0.5,
        "net": pytest.approx(1.6),
        "unit": "million_nzd_npv",
    }
    assert result["enumeration"]["exact"] is True
    assert result["enumeration"]["mapping_count_evaluated"] == 27
    assert all(
        item["shadow_value_status"] == "not_a_local_shadow_price"
        for item in result["shadow_value_evidence"]
    )


@pytest.mark.parametrize(
    ("kind", "expected_gross"),
    [
        ("expected_value", 2.1),
        ("expected_utility", 2.1),
        ("lower_tail_cvar", 1.2),
        ("minimax_regret", 0.0),
    ],
)
def test_supported_risk_functionals_use_the_same_feasible_policy_problem(
    kind: str, expected_gross: float
) -> None:
    payload = _input()
    payload["objective"]["kind"] = kind
    if kind == "lower_tail_cvar":
        payload["objective"]["confidence_level"] = 0.5
    else:
        payload["objective"].pop("confidence_level", None)
    if kind == "minimax_regret":
        payload["objective"]["regret_reference_by_state"] = {
            "high": 10.0,
            "mid": 9.0,
            "low": 11.0,
        }
    else:
        payload["objective"].pop("regret_reference_by_state", None)
    result = risk_sensitive_constrained_voi(payload).to_contract_dict()
    assert result["value"]["gross"] == pytest.approx(expected_gross)
    assert result["objective"]["kind"] == kind


def test_complete_ties_are_retained_with_lexicographic_presentation() -> None:
    payload = _input()
    steady = next(item for item in payload["policies"] if item["policy_id"] == "steady")
    clone = deepcopy(steady)
    clone["policy_id"] = "steady_clone"
    clone["label"] = "Steady clone"
    payload["policies"].append(clone)
    payload["assurance"]["max_policy_mappings"] = 256
    result = risk_sensitive_constrained_voi(payload).to_contract_dict()
    assert result["baseline"]["selected_policy_id"] == "steady"
    assert result["baseline"]["tied_policy_ids"] == ["steady", "steady_clone"]
    assert result["perfect_information"]["tied_policy_mappings"]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda p: p["states"][0].update(probability=0.4), "sum to 1"),
        (
            lambda p: p["policies"][0]["objective_by_state"].pop("low"),
            "objective state keys",
        ),
        (
            lambda p: p["information_action"]["cost"].update(unit="people"),
            "cost unit",
        ),
        (
            lambda p: p["assurance"].update(max_policy_mappings=2),
            "exceeds max_policy_mappings",
        ),
    ],
)
def test_semantic_pathologies_fail_closed(mutate: Any, message: str) -> None:
    payload = _input()
    mutate(payload)
    with pytest.raises(InputError, match=message):
        risk_sensitive_constrained_voi(payload)


def test_no_feasible_baseline_policy_is_explicit() -> None:
    payload = _input()
    payload["constraints"][0]["limit"] = 0.0
    with pytest.raises(InputError, match="no feasible baseline policy"):
        risk_sensitive_constrained_voi(payload)


def test_cli_and_public_exports_execute_the_experimental_contract(
    tmp_path: Path,
) -> None:
    output = tmp_path / "risk.json"
    run = CliRunner().invoke(
        app,
        [
            "--format",
            "json",
            "calculate-risk-sensitive-voi",
            str(INPUT),
            "--output",
            str(output),
        ],
    )
    assert run.exit_code == 0, run.output
    payload = json.loads(run.stdout)
    assert payload == json.loads(output.read_text(encoding="utf-8"))
    assert payload["method_maturity"] == "experimental"
    assert voiage.RiskSensitiveVoiResult is RiskSensitiveVoiResult
    assert voiage.risk_sensitive_constrained_voi is risk_sensitive_constrained_voi


def test_capabilities_are_honest_about_language_and_promotion_gates() -> None:
    capability = _json(CONTRACT / "capabilities.json")
    assert capability["planned_version"] == "v1.3.0"
    assert capability["maturity"] == "experimental"
    assert capability["surfaces"]["python"]["status"] == "executable"
    assert capability["surfaces"]["rust"]["status"] == "unsupported"
    assert "independent scientific review" in capability["remaining_gates"]


def test_evidence_hashes_pin_the_experimental_delivery_artifacts() -> None:
    evidence = _json(CONTRACT / "fixtures/evidence.json")
    assert evidence["stable_claim_allowed"] is False
    for artifact in evidence["artifacts"]:
        path = ROOT / artifact["path"]
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]
