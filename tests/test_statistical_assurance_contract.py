"""Governance tests for stable estimator statistical-assurance reporting."""

from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator

POLICY_PATH = Path("specs/v1/stable-estimator-statistical-assurance.json")
POLICY_SCHEMA_PATH = Path("specs/v1/stable-estimator-statistical-assurance.schema.json")
ENVELOPE_SCHEMA_PATH = Path(
    "specs/v1/schemas/statistical-assurance-envelope.schema.json"
)
ESTIMATOR_PATH = Path("specs/v1/stable-estimator-assurance.json")


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_statistical_policy_and_runtime_envelope_conform_to_their_schemas() -> None:
    policy = _load(POLICY_PATH)
    Draft202012Validator(_load(POLICY_SCHEMA_PATH)).validate(policy)
    Draft202012Validator.check_schema(_load(ENVELOPE_SCHEMA_PATH))


def test_statistical_policy_covers_every_stable_estimator_profile_once() -> None:
    policy = _load(POLICY_PATH)
    estimator_contract = _load(ESTIMATOR_PATH)
    profiles = policy["profiles"]

    assert len(profiles) == len({profile["method_id"] for profile in profiles})
    assert {profile["method_id"] for profile in profiles} == {
        profile["method_id"] for profile in estimator_contract["profiles"]
    }


def test_stochastic_estimators_cannot_omit_replay_or_error_reporting() -> None:
    profiles = {
        profile["method_id"]: profile for profile in _load(POLICY_PATH)["profiles"]
    }
    stochastic = {
        "evppi-regression",
        "evsi-nested-mc",
        "evsi-regression",
        "evsi-moment-matching",
    }

    for method_id in stochastic:
        profile = profiles[method_id]
        assert profile["reporting_class"] != "deterministic"
        assert profile["bias_policy"] != "not-applicable"
        assert profile["variance_policy"] != "not-applicable"
        assert profile["monte_carlo_error_policy"] != "not-applicable"
        assert profile["convergence_policy"] != "not-applicable"
        assert profile["rng_identity_policy"] != "not-applicable"
        assert profile["replication_policy"] != "not-applicable"
        assert profile["budget_policy"] != "not-applicable"
        assert profile["stopping_policy"] != "not-applicable"
        assert profile["numerical_error_policy"] != "not-applicable"


def test_effective_sample_size_is_required_only_when_statistically_meaningful() -> None:
    profiles = _load(POLICY_PATH)["profiles"]
    allowed = {
        "not-applicable-independent-unweighted-draws",
        "required-for-weighted-or-correlated-draws",
    }

    for profile in profiles:
        assert profile["effective_sample_size_policy"] in allowed
