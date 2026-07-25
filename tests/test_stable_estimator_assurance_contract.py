"""Governance tests for the v1.1 stable estimator assurance contract."""

from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator

ASSURANCE_PATH = Path("specs/v1/stable-estimator-assurance.json")
SCHEMA_PATH = Path("specs/v1/stable-estimator-assurance.schema.json")
FREEZE_PATH = Path("specs/software-landscape/v1.1-scientific-freeze-candidate.json")


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_assurance_contract_conforms_to_its_published_schema() -> None:
    Draft202012Validator(_load(SCHEMA_PATH)).validate(_load(ASSURANCE_PATH))


def test_assurance_contract_is_bound_to_the_approved_scientific_freeze() -> None:
    assurance = _load(ASSURANCE_PATH)
    freeze = _load(FREEZE_PATH)

    assert assurance["contract_version"] == "1.1.0"
    assert assurance["status"] == "normative"
    assert assurance["scientific_freeze"] == {
        "candidate_digest": freeze["candidate_digest"],
        "approval_path": (
            "specs/software-landscape/v1.1-scientific-freeze-approval.json"
        ),
        "scope": "scientific-contract-only",
        "implementation_gate_waived": False,
    }


def test_every_stable_numerical_family_has_one_complete_policy() -> None:
    assurance = _load(ASSURANCE_PATH)
    profiles = assurance["profiles"]
    by_id = {profile["method_id"]: profile for profile in profiles}
    required = {
        "net-benefit",
        "expected-loss",
        "evpi",
        "evppi-regression",
        "evsi-nested-mc",
        "evsi-regression",
        "evsi-moment-matching",
        "enbs",
        "ceaf",
        "dominance",
        "structural-voi",
    }

    assert set(by_id) == required
    assert len(profiles) == len(by_id)
    for profile in profiles:
        assert profile["maturity"] == "stable"
        assert profile["implementation_state"] in {
            "conformant",
            "requires-assurance-evidence",
        }
        assert profile["comparison_policy"] in {
            "exact-structure-and-declared-numeric-tolerance",
            "declared-numeric-tolerance",
        }
        assert profile["tie_policy"]
        assert profile["fallback_policy"] in {
            "fail-closed",
            "explicit-opt-in-and-diagnosed",
        }
        assert profile["clipping_policy"]
        assert profile["failure_policy"]
        assert profile["required_diagnostics"]
        assert Path(profile["implementation"]).is_file()


def test_every_stable_freeze_method_is_profiled_or_explicitly_delegated() -> None:
    assurance = _load(ASSURANCE_PATH)
    freeze = _load(FREEZE_PATH)
    covered = {profile["method_id"] for profile in assurance["profiles"]}
    delegated = set(assurance["scope"]["delegated"])
    stable = {method["method_id"] for method in freeze["stable_methods"]}

    assert covered.isdisjoint(delegated)
    assert covered | delegated == stable


def test_required_diagnostics_use_stable_binding_error_codes() -> None:
    assurance = _load(ASSURANCE_PATH)
    stable_codes = {
        "invalid_input",
        "dimension_mismatch",
        "backend_unavailable",
        "numerical_failure",
        "serialization_failure",
    }

    for profile in assurance["profiles"]:
        assert set(profile["required_diagnostics"]) <= stable_codes


def test_global_policy_freezes_tolerances_failures_and_fallback_visibility() -> None:
    policy = _load(ASSURANCE_PATH)["global_policy"]

    assert policy["float_type"] == "binary64"
    assert policy["absolute_tolerance"] == 1e-10
    assert policy["relative_tolerance"] == 1e-8
    assert policy["non_finite_input"] == "reject"
    assert policy["non_finite_output"] == "reject"
    assert policy["overflow"] == "fail-closed"
    assert policy["fallback_default"] == "disabled"
    assert policy["fallback_visibility"] == "degraded-diagnostic-required"
    assert policy["unsupported_capability"] == "typed-error"
