"""Fail-closed validation for H8-D human commissioning preparation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import (
    SchemaError as JsonSchemaSchemaError,
)
from jsonschema.exceptions import (
    ValidationError as JsonSchemaValidationError,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

CONTRACT_PATH = Path("specs/frontier/sampling-acquisition-harm/v1")
PREFLIGHT_PATH = CONTRACT_PATH / "human-commissioning-preflight-20260803.json"
SCHEMA_PATH = CONTRACT_PATH / "schemas/human-commissioning-preflight.schema.json"
DECISION_PATH = CONTRACT_PATH / "candidate-context-decision-20260803.json"
DECISION_SCHEMA_PATH = CONTRACT_PATH / "schemas/candidate-context-decision.schema.json"
DELTA_PATH = CONTRACT_PATH / "remediation-readiness-delta-20260803.json"
REVIEWER_INTAKE_PATH = CONTRACT_PATH / "reviewer-intake-readiness.json"
SOURCE_INTAKE_PATH = CONTRACT_PATH / "source-review-intake-readiness.json"

EXPECTED_OPTION_IDS = {
    "reviewed_exclusion_generic_kernel",
    "bounded_non_authorizing_candidate",
    "defer_and_retain_unsupported",
}
EXPECTED_SOURCE_FINDINGS = {
    "H8D-ED-04",
    "H8D-EST-002",
    "H8D-API-GOV-03",
    "H8D-GP-02",
    "H8D-DS-05",
}
EXPECTED_BLOCKERS = {
    "candidate-context-decision",
    "independent-source-retrieval",
    "source-rights-review",
    "jurisdiction-applicability",
    "eligible-h8-d-reviewers",
    "replacement-packet-freeze",
    "nineteen-finding-disposition",
}
EXPECTED_BINDINGS = {
    "remediation-readiness-delta-20260803.json",
    "reviewer-intake-readiness.json",
    "source-review-intake-readiness.json",
    "source-observation-refresh-20260803.json",
}
EXPECTED_REMAINING_BLOCKERS = EXPECTED_BLOCKERS - {"candidate-context-decision"}


class SamplingHarmHumanCommissioningError(ValueError):
    """Raised when commissioning preparation is incomplete or overclaims authority."""


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SamplingHarmHumanCommissioningError(f"{path} must contain an object")
    return value


def _validate_schema(value: object, schema: dict[str, Any]) -> None:
    try:
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(value)
    except (JsonSchemaSchemaError, JsonSchemaValidationError) as error:
        location = "/".join(str(item) for item in error.absolute_path) or "$"
        raise SamplingHarmHumanCommissioningError(
            f"preflight invalid at {location}: {error.message}"
        ) from error


def _raw_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_sampling_harm_human_commissioning(
    preflight: Mapping[str, Any],
    delta: Mapping[str, Any],
    reviewer_intake: Mapping[str, Any],
    source_intake: Mapping[str, Any],
    *,
    schema: dict[str, Any],
    contract_root: Path,
) -> dict[str, Any]:
    """Validate the canonical blocked commissioning state and exact inputs."""
    _validate_schema(preflight, schema)

    bindings = preflight["input_bindings"]
    paths = [item["path"] for item in bindings]
    for value in paths:
        path = Path(value)
        if path.is_absolute() or len(path.parts) != 1:
            raise SamplingHarmHumanCommissioningError("input binding path is unsafe")
    if len(paths) != len(set(paths)) or set(paths) != EXPECTED_BINDINGS:
        raise SamplingHarmHumanCommissioningError("input binding inventory mismatch")
    for item in bindings:
        path = Path(item["path"])
        if _raw_sha256(contract_root / path) != item["sha256"]:
            raise SamplingHarmHumanCommissioningError(
                f"input digest mismatch: {item['path']}"
            )

    context = preflight["candidate_context"]
    options = context["options"]
    option_ids = [item["option_id"] for item in options]
    if (
        len(option_ids) != len(set(option_ids))
        or set(option_ids) != EXPECTED_OPTION_IDS
    ):
        raise SamplingHarmHumanCommissioningError("candidate option inventory mismatch")
    recommended = [item["option_id"] for item in options if item["recommended"]]
    if recommended != [context["recommended_option_id"]]:
        raise SamplingHarmHumanCommissioningError(
            "exactly one candidate option must be recommended"
        )

    summary = delta.get("summary")
    if (
        not isinstance(summary, dict)
        or summary.get("total_findings") != 19
        or summary.get("pending") != 19
    ):
        raise SamplingHarmHumanCommissioningError(
            "nineteen pending findings are required"
        )
    groups = delta.get("groups")
    if not isinstance(groups, dict):
        raise SamplingHarmHumanCommissioningError("finding readiness groups are absent")
    source_findings = groups.get("source_review_prerequisite")
    if set(source_findings or []) != EXPECTED_SOURCE_FINDINGS:
        raise SamplingHarmHumanCommissioningError(
            "source prerequisite binding mismatch"
        )
    if set(preflight["source_prerequisite_finding_ids"]) != EXPECTED_SOURCE_FINDINGS:
        raise SamplingHarmHumanCommissioningError(
            "source prerequisite binding mismatch"
        )

    scientific_roles = reviewer_intake.get("required_scientific_roles")
    confirmation_roles = reviewer_intake.get("human_confirmation_roles")
    if not isinstance(scientific_roles, list) or not isinstance(
        confirmation_roles, list
    ):
        raise SamplingHarmHumanCommissioningError("reviewer intake roles are absent")
    intake_roles = scientific_roles + confirmation_roles
    if any(
        item.get("eligible") is not False
        or item.get("assignment_status") != "unassigned"
        for item in intake_roles
    ):
        raise SamplingHarmHumanCommissioningError(
            "canonical intake unexpectedly claims an eligible reviewer"
        )
    expected_roles = {item["role_id"] for item in intake_roles}
    preflight_roles = {item["role_id"] for item in preflight["required_human_roles"]}
    if (
        len(preflight["required_human_roles"]) != len(preflight_roles)
        or preflight_roles != expected_roles
    ):
        raise SamplingHarmHumanCommissioningError("reviewer role binding mismatch")
    expected_stages = {item["role_id"]: "h8_d" for item in scientific_roles} | {
        item["role_id"]: "h8_g" for item in confirmation_roles
    }
    if any(
        item["stage"] != expected_stages[item["role_id"]]
        for item in preflight["required_human_roles"]
    ):
        raise SamplingHarmHumanCommissioningError("reviewer stage binding mismatch")
    if reviewer_intake.get("receipt_received") is not False or any(
        value is not False for value in reviewer_intake.get("authority", {}).values()
    ):
        raise SamplingHarmHumanCommissioningError(
            "canonical reviewer intake unexpectedly claims authority"
        )

    source_flags = (
        "rights_review_complete",
        "applicability_review_complete",
        "drift_review_complete",
        "independent_retrieval_receipt",
        "source_authority",
    )
    if any(source_intake.get(field) is not False for field in source_flags):
        raise SamplingHarmHumanCommissioningError(
            "canonical source intake unexpectedly claims authority"
        )
    if set(source_intake.get("finding_ids", [])) != EXPECTED_SOURCE_FINDINGS:
        raise SamplingHarmHumanCommissioningError("source intake finding mismatch")

    if set(preflight["blocker_ids"]) != EXPECTED_BLOCKERS:
        raise SamplingHarmHumanCommissioningError("commissioning blocker mismatch")
    if any(preflight["preconditions"].values()) or any(
        preflight["authority_boundary"].values()
    ):
        raise SamplingHarmHumanCommissioningError(
            "commissioning preflight claims unavailable authority"
        )
    privacy = preflight["privacy_and_security"]
    if any(
        privacy[field]
        for field in (
            "credentials_in_repository",
            "personal_contact_details_in_repository",
            "raw_signatures_in_repository",
        )
    ):
        raise SamplingHarmHumanCommissioningError(
            "commissioning preflight permits sensitive repository data"
        )

    return {
        "commissioning_status": preflight["commissioning_status"],
        "candidate_decision": context["decision_status"],
        "recommended_option": context["recommended_option_id"],
        "pending_findings": summary["pending"],
        "source_prerequisites": len(EXPECTED_SOURCE_FINDINGS),
        "eligible_reviewers": 0,
        "ready": False,
    }


def load_and_validate_sampling_harm_human_commissioning(
    repository_root: Path,
) -> dict[str, Any]:
    """Load and validate the repository's canonical commissioning preflight."""
    root = repository_root.resolve()
    contract = root / CONTRACT_PATH
    return validate_sampling_harm_human_commissioning(
        _load_object(root / PREFLIGHT_PATH),
        _load_object(root / DELTA_PATH),
        _load_object(root / REVIEWER_INTAKE_PATH),
        _load_object(root / SOURCE_INTAKE_PATH),
        schema=_load_object(root / SCHEMA_PATH),
        contract_root=contract,
    )


def validate_sampling_harm_candidate_decision(
    decision: Mapping[str, Any],
    preflight: Mapping[str, Any],
    delta: Mapping[str, Any],
    reviewer_intake: Mapping[str, Any],
    source_intake: Mapping[str, Any],
    *,
    schema: dict[str, Any],
) -> dict[str, Any]:
    """Validate the accountable option-one receipt without advancing H8-D."""
    try:
        _validate_schema(decision, schema)
    except SamplingHarmHumanCommissioningError as error:
        raise SamplingHarmHumanCommissioningError(
            f"decision receipt invalid: {error}"
        ) from error

    context = preflight.get("candidate_context")
    if not isinstance(context, dict):
        raise SamplingHarmHumanCommissioningError("candidate context is absent")
    option_ids = {
        item.get("option_id")
        for item in context.get("options", [])
        if isinstance(item, dict)
    }
    selected = decision["selected_option_id"]
    if (
        context.get("decision_status") != "decision_required"
        or context.get("selected_option_id") is not None
        or selected not in option_ids
        or selected != context.get("recommended_option_id")
    ):
        raise SamplingHarmHumanCommissioningError(
            "decision receipt does not supersede the exact preflight option"
        )

    remaining = decision["remaining_blocker_ids"]
    if (
        len(remaining) != len(set(remaining))
        or set(remaining) != EXPECTED_REMAINING_BLOCKERS
    ):
        raise SamplingHarmHumanCommissioningError(
            "remaining blocker inventory mismatch"
        )

    authority = decision["authority_boundary"]
    if authority.get("candidate_selected") is not True or any(
        value is not False
        for key, value in authority.items()
        if key != "candidate_selected"
    ):
        raise SamplingHarmHumanCommissioningError(
            "decision receipt claims unavailable authority"
        )

    summary = delta.get("summary")
    if not isinstance(summary, dict) or summary.get("pending") != 19:
        raise SamplingHarmHumanCommissioningError(
            "nineteen pending findings are required"
        )
    scientific_roles = reviewer_intake.get("required_scientific_roles")
    if not isinstance(scientific_roles, list) or any(
        item.get("eligible") is not False
        or item.get("assignment_status") != "unassigned"
        for item in scientific_roles
    ):
        raise SamplingHarmHumanCommissioningError(
            "candidate decision unexpectedly advances reviewer eligibility"
        )
    source_flags = (
        "rights_review_complete",
        "applicability_review_complete",
        "drift_review_complete",
        "independent_retrieval_receipt",
        "source_authority",
    )
    if any(source_intake.get(field) is not False for field in source_flags):
        raise SamplingHarmHumanCommissioningError(
            "candidate decision unexpectedly advances source authority"
        )

    return {
        "commissioning_status": "blocked_prerequisites",
        "candidate_decision": decision["decision_status"],
        "selected_option": selected,
        "remaining_blockers": len(remaining),
        "pending_findings": summary["pending"],
        "source_review_ready": False,
        "eligible_reviewers": 0,
        "ready": False,
    }


def load_and_validate_sampling_harm_candidate_decision(
    repository_root: Path,
) -> dict[str, Any]:
    """Load the selected candidate receipt after validating its preflight."""
    root = repository_root.resolve()
    load_and_validate_sampling_harm_human_commissioning(root)
    return validate_sampling_harm_candidate_decision(
        _load_object(root / DECISION_PATH),
        _load_object(root / PREFLIGHT_PATH),
        _load_object(root / DELTA_PATH),
        _load_object(root / REVIEWER_INTAKE_PATH),
        _load_object(root / SOURCE_INTAKE_PATH),
        schema=_load_object(root / DECISION_SCHEMA_PATH),
    )
