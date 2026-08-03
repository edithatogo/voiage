"""Fail-closed validation for H8 source observations and remediation readiness."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError

CONTRACT_ROOT = Path("specs/frontier/sampling-acquisition-harm/v1")
SOURCE_PATH = CONTRACT_ROOT / "source-observation-refresh-20260803.json"
DELTA_PATH = CONTRACT_ROOT / "remediation-readiness-delta-20260803.json"
REGISTER_PATH = CONTRACT_ROOT / "remediation-register.json"
SCHEMA_ROOT = CONTRACT_ROOT / "schemas"

EXPECTED_SOURCE_IDS = {
    "heath-2024-trial-design-voi",
    "belmont-report",
    "45-cfr-46",
    "ich-e6-r3-final",
    "camilleri-2022-safe-active-learning",
    "bottero-2022-safe-exploration",
}
EXPECTED_REPOSITORY_FINDINGS = {
    "H8D-API-GOV-01",
    "H8D-API-GOV-02",
    "H8D-GP-01",
}
EXPECTED_SOURCE_FINDINGS = {
    "H8D-ED-04",
    "H8D-EST-002",
    "H8D-API-GOV-03",
    "H8D-GP-02",
    "H8D-DS-05",
}


class SamplingHarmSourceReadinessError(ValueError):
    """Raised when source or remediation preparation overstates readiness."""


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SamplingHarmSourceReadinessError(f"{path} must contain an object")
    return value


def _validate_schema(value: object, schema: Mapping[str, Any], label: str) -> None:
    try:
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema, format_checker=FormatChecker()).validate(value)
    except JsonSchemaValidationError as error:
        location = "/".join(str(item) for item in error.absolute_path) or "$"
        raise SamplingHarmSourceReadinessError(
            f"{label} invalid at {location}: {error.message}"
        ) from error


def _string_set(values: Sequence[object], label: str) -> set[str]:
    if not all(isinstance(value, str) for value in values):
        raise SamplingHarmSourceReadinessError(f"{label} contains a non-string")
    result = set(values)
    if len(result) != len(values):
        raise SamplingHarmSourceReadinessError(f"{label} contains duplicates")
    return result


def validate_sampling_harm_source_readiness(
    source_observation: Mapping[str, Any],
    remediation_delta: Mapping[str, Any],
    remediation_register: Mapping[str, Any],
    *,
    source_schema: Mapping[str, Any],
    delta_schema: Mapping[str, Any],
) -> dict[str, int | str]:
    """Validate the exact source inventory and the 19-finding pending partition."""
    _validate_schema(source_observation, source_schema, "source observation")
    _validate_schema(remediation_delta, delta_schema, "remediation delta")

    sources = source_observation["sources"]
    source_ids = _string_set(
        [item["stable_source_id"] for item in sources], "source inventory"
    )
    if source_ids != EXPECTED_SOURCE_IDS:
        raise SamplingHarmSourceReadinessError("source inventory mismatch")

    by_id = {item["stable_source_id"]: item for item in sources}
    stable_ids = {
        source_id
        for source_id, item in by_id.items()
        if item["drift_assessment"] == "byte_stable_same_representation"
    }
    expected_stable = {
        "ich-e6-r3-final",
        "camilleri-2022-safe-active-learning",
        "bottero-2022-safe-exploration",
    }
    if stable_ids != expected_stable:
        raise SamplingHarmSourceReadinessError("byte-stable source set mismatch")
    for source_id in stable_ids:
        item = by_id[source_id]
        if item["prior_observation_sha256"] != item["current_observation_sha256"]:
            raise SamplingHarmSourceReadinessError(
                f"byte-stable digest mismatch: {source_id}"
            )
    if (
        by_id["belmont-report"]["prior_observation_sha256"] is not None
        or by_id["belmont-report"]["current_observation_sha256"] is not None
    ):
        raise SamplingHarmSourceReadinessError(
            "CLI-blocked Belmont observation must not claim a digest"
        )
    if (
        by_id["45-cfr-46"]["representation"] != "application/xml"
        or "not_byte_comparable" not in by_id["45-cfr-46"]["drift_assessment"]
    ):
        raise SamplingHarmSourceReadinessError(
            "eCFR representation change must remain non-comparable"
        )

    findings = remediation_register.get("findings")
    if not isinstance(findings, list):
        raise SamplingHarmSourceReadinessError("remediation findings are absent")
    register_ids = _string_set(
        [item.get("finding_id") for item in findings], "remediation register"
    )
    if len(register_ids) != 19:
        raise SamplingHarmSourceReadinessError(
            "remediation register must retain nineteen findings"
        )
    if any(item.get("disposition_status") != "pending" for item in findings):
        raise SamplingHarmSourceReadinessError("a finding is not pending")

    groups = remediation_delta["groups"]
    repository_ids = _string_set(
        groups["repository_implemented_awaiting_independent_rereview"],
        "repository readiness group",
    )
    source_finding_ids = _string_set(
        groups["source_review_prerequisite"], "source prerequisite group"
    )
    human_ids = _string_set(
        groups["candidate_and_human_review_prerequisite"],
        "candidate and human prerequisite group",
    )
    if repository_ids != EXPECTED_REPOSITORY_FINDINGS:
        raise SamplingHarmSourceReadinessError("repository finding set mismatch")
    if source_finding_ids != EXPECTED_SOURCE_FINDINGS:
        raise SamplingHarmSourceReadinessError("source finding set mismatch")
    if (
        (repository_ids & source_finding_ids)
        or (repository_ids & human_ids)
        or (source_finding_ids & human_ids)
    ):
        raise SamplingHarmSourceReadinessError("finding readiness groups overlap")
    if repository_ids | source_finding_ids | human_ids != register_ids:
        raise SamplingHarmSourceReadinessError(
            "finding readiness groups do not partition the register"
        )
    critical = next(
        (item for item in findings if item.get("finding_id") == "H8D-DS-03"),
        None,
    )
    if critical is None or critical.get("severity") != "Critical":
        raise SamplingHarmSourceReadinessError("Critical H8D-DS-03 is not preserved")

    return {
        "sources": len(source_ids),
        "findings": len(register_ids),
        "pending": len(register_ids),
        "replacement_packet": remediation_delta["replacement_packet"]["status"],
    }


def validate_repository_sampling_harm_source_readiness(
    repository_root: Path,
) -> dict[str, int | str]:
    """Load and validate the canonical repository artifacts."""
    root = repository_root.resolve()
    return validate_sampling_harm_source_readiness(
        _load_object(root / SOURCE_PATH),
        _load_object(root / DELTA_PATH),
        _load_object(root / REGISTER_PATH),
        source_schema=_load_object(
            root / SCHEMA_ROOT / "source-observation-refresh.schema.json"
        ),
        delta_schema=_load_object(
            root / SCHEMA_ROOT / "remediation-readiness-delta.schema.json"
        ),
    )
