"""Governance tests for stable estimator statistical-assurance reporting."""

from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator
import pytest

from voiage.exceptions import InputError
from voiage.statistical_assurance import summarize_evsi_replications

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


def test_independent_evsi_replications_produce_typed_convergence_evidence() -> None:
    result = summarize_evsi_replications(
        [1.0, 1.1, 0.9, 1.0],
        [11, 12, 13, 14],
        reporting_class="nested-monte-carlo",
        relative_tolerance=0.2,
    )

    assert result.estimate == pytest.approx(1.0)
    assert result.replication_seeds == (11, 12, 13, 14)
    assert result.assurance.replications == 4
    assert result.assurance.convergence is not None
    assert result.assurance.convergence.converged
    assert result.assurance.monte_carlo_standard_error is not None


def test_evsi_replication_summary_rejects_duplicate_seeds() -> None:
    with pytest.raises(InputError, match="unique seeds"):
        summarize_evsi_replications(
            [1.0, 1.1],
            [11, 11],
            reporting_class="nested-monte-carlo",
        )
