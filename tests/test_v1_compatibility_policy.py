"""Governance checks for the normative v1 compatibility policy."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError
import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

POLICY_PATH = Path("specs/v1/compatibility-policy.json")
SCHEMA_PATH = Path("specs/v1/compatibility-policy.schema.json")


def _policy() -> dict[str, object]:
    return json.loads(POLICY_PATH.read_text(encoding="utf-8"))


def _schema() -> dict[str, object]:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def _validator() -> Draft202012Validator:
    schema = _schema()
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def test_policy_validates_against_closed_draft_2020_12_schema() -> None:
    _validator().validate(_policy())


@pytest.mark.parametrize(
    "mutate",
    [
        lambda policy: policy.update({"unreviewed": True}),
        lambda policy: policy["surfaces"]["python"].update({"unknown": True}),
        lambda policy: policy["deprecation"].pop("minimum_months"),
        lambda policy: policy["deprecation"].update({"minimum_months": "six"}),
        lambda policy: policy["surfaces"]["bindings"].update(
            {"covered": ["python", "python"]}
        ),
        lambda policy: policy["surfaces"]["cli"]["exit_codes"].update({"success": 1}),
    ],
    ids=[
        "unknown-root-property",
        "unknown-nested-property",
        "missing-required-property",
        "wrong-property-type",
        "duplicate-binding",
        "contradictory-exit-code",
    ],
)
def test_policy_schema_rejects_malformed_examples(
    mutate: Callable[[dict[str, Any]], object],
) -> None:
    malformed = deepcopy(_policy())
    mutate(malformed)

    with pytest.raises(ValidationError):
        _validator().validate(malformed)


def test_policy_defines_every_stable_compatibility_surface() -> None:
    policy = _policy()

    assert set(policy) == {
        "$schema",
        "policy_version",
        "applies_from",
        "semantic_versioning",
        "surfaces",
        "deprecation",
        "excluded_stabilities",
        "excluded_surfaces_still_require",
        "promotion_requirements",
        "compatibility_evidence",
    }
    assert policy["policy_version"] == "1.0.0"
    assert set(policy["surfaces"]) == {
        "rust",
        "python",
        "c_abi",
        "bindings",
        "cli",
        "schemas",
        "serialized_outputs",
    }
    assert all(
        surface["breaking_change"] == "major" for surface in policy["surfaces"].values()
    )


def test_policy_freezes_deprecation_clock_and_warning_behavior() -> None:
    deprecation = _policy()["deprecation"]

    assert deprecation["minimum_minor_releases"] == 2
    assert deprecation["minimum_months"] == 6
    assert deprecation["python_warning_class"] == "FutureWarning"
    assert deprecation["replacement_required_before_removal"] is True
    assert deprecation["removal_release"] == "next-major-only"


def test_policy_excludes_nonstable_surfaces_and_requires_fixtures() -> None:
    policy = _policy()

    assert policy["excluded_stabilities"] == ["provisional", "experimental"]
    assert policy["compatibility_evidence"]["fixture_required"] is True
    assert policy["compatibility_evidence"]["all_supported_bindings"] is True
    assert (
        policy["compatibility_evidence"]["breaking_change_requires_migration_guide"]
        is True
    )


def test_policy_freezes_abi_cli_schema_and_serialization_details() -> None:
    surfaces = _policy()["surfaces"]

    assert surfaces["c_abi"]["symbol_versioning"] == "required"
    assert surfaces["c_abi"]["opaque_handles_preferred"] is True
    assert surfaces["cli"]["exit_codes"] == {
        "success": 0,
        "runtime_failure": 1,
        "invalid_input": 2,
    }
    assert surfaces["schemas"]["stable_ids_and_paths"] is True
    assert surfaces["serialized_outputs"]["json_numbers"] == "finite-only"
    assert surfaces["serialized_outputs"]["encoding"] == "utf-8"


def test_nonstable_surfaces_retain_safety_governance() -> None:
    assert set(_policy()["excluded_surfaces_still_require"]) == {
        "security",
        "validation",
        "provenance",
        "licensing",
        "data-safety",
        "maturity-metadata",
    }


def test_policy_objects_reject_unreviewed_structural_drift() -> None:
    policy = _policy()

    assert set(policy["semantic_versioning"]) == {"major", "minor", "patch"}
    assert set(policy["deprecation"]) == {
        "minimum_minor_releases",
        "minimum_months",
        "clock_rule",
        "python_warning_class",
        "diagnostic_code_required",
        "replacement_required_before_removal",
        "migration_documentation_required",
        "removal_release",
    }
    assert set(policy["compatibility_evidence"]) == {
        "fixture_required",
        "all_supported_bindings",
        "breaking_change_requires_migration_guide",
    }
