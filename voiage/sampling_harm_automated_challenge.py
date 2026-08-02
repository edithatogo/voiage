"""Fail-closed validation for the H8-D/H8-E automated challenge evidence."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError
from referencing import Registry, Resource

from voiage.scientific_review_evidence import canonical_json_sha256

CONTRACT_ROOT = Path("specs/frontier/sampling-acquisition-harm/v1")
REVIEW_ROOT = CONTRACT_ROOT / "reviews"
SYNTHESIS_PATH = REVIEW_ROOT / "h8d-automated-challenge-synthesis-20260803.json"
SYNTHESIS_SCHEMA = CONTRACT_ROOT / "schemas/automated-challenge-synthesis.schema.json"
SCIENTIFIC_SCHEMA_ROOT = Path("specs/frontier/governance/scientific-review/v1/schemas")
FROZEN_CANDIDATE_COMMIT = "8d6c67879050f161258ed95d878a72e2bb6b22dd"
FROZEN_CANDIDATE_TREE = "18289bd04081f6a6810cb91ef2beec7decafe61f"
TRUSTED_PACKAGE_COMMIT = "d00e0e20752f44c52581dbb7ee45ce27c9b7d6dd"
FROZEN_MANIFEST_SHA256 = (
    "4f18ac9b08717416e54133849c1a381b4245543d7e5dd85f51efa1cd789164c5"
)
FROZEN_PACKET_SHA256 = (
    "e1298da5a609ee9ed7a8cc8509ab117a2d2fd384d20d464c48ec32f4f033f29b"
)
REQUIRED_ROLES = (
    "estimand_domain",
    "estimator_assurance",
    "cross_language_api",
    "governance_publication",
    "domain_specialist",
)
FALSE_GATE_FIELDS = (
    "independent_eligibility_satisfied",
    "source_review_satisfied",
    "h8d_satisfied",
    "h8e_satisfied",
    "h8f_satisfied",
    "h8g_satisfied",
    "h8h_satisfied",
)


class SamplingHarmAutomatedChallengeError(ValueError):
    """Raised when automated challenge evidence is incomplete or overclaims."""


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise SamplingHarmAutomatedChallengeError(
            f"cannot load JSON object: {path}"
        ) from error
    if not isinstance(value, dict):
        raise SamplingHarmAutomatedChallengeError(
            f"JSON value is not an object: {path}"
        )
    return value


def _validate_schema(
    value: object,
    schema: dict[str, Any],
    *,
    label: str,
    registry: Registry[Any] | None = None,
) -> None:
    try:
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(
            schema, registry=registry or Registry(), format_checker=FormatChecker()
        ).validate(value)
    except JsonSchemaValidationError as error:
        location = "/".join(str(item) for item in error.absolute_path) or "$"
        raise SamplingHarmAutomatedChallengeError(
            f"{label} invalid at {location}: {error.message}"
        ) from error


def _safe_report_path(value: object) -> Path:
    if not isinstance(value, str):
        raise SamplingHarmAutomatedChallengeError("role-report path must be a string")
    path = PurePosixPath(value)
    expected_prefix = PurePosixPath(REVIEW_ROOT.as_posix())
    if (
        path.is_absolute()
        or ".." in path.parts
        or str(path) != value
        or path.parent != expected_prefix
    ):
        raise SamplingHarmAutomatedChallengeError(f"unsafe role-report path: {value}")
    return Path(path)


def validate_sampling_harm_automated_challenge(
    synthesis: Mapping[str, Any], *, repository_root: Path
) -> dict[str, object]:
    """Validate role reports, synthesis bindings, findings and authority limits."""
    root = repository_root.resolve()
    synthesis_schema = _load_object(root / SYNTHESIS_SCHEMA)
    _validate_schema(synthesis, synthesis_schema, label="automated challenge synthesis")

    declared_digest = synthesis["synthesis_sha256"]
    actual_digest = canonical_json_sha256(
        synthesis, excluded_json_pointers={"/synthesis_sha256"}
    )
    if declared_digest != actual_digest:
        raise SamplingHarmAutomatedChallengeError("synthesis canonical digest mismatch")

    bindings = synthesis["bindings"]
    expected_bindings = {
        "candidate_commit": FROZEN_CANDIDATE_COMMIT,
        "candidate_tree": FROZEN_CANDIDATE_TREE,
        "trusted_package_commit": TRUSTED_PACKAGE_COMMIT,
    }
    for field, expected in expected_bindings.items():
        if bindings[field] != {"algorithm": "sha1", "value": expected}:
            raise SamplingHarmAutomatedChallengeError(f"unexpected {field} binding")
    if bindings["artifact_manifest"]["sha256"] != FROZEN_MANIFEST_SHA256:
        raise SamplingHarmAutomatedChallengeError(
            "unexpected artifact manifest binding"
        )
    if bindings["review_packet"]["sha256"] != FROZEN_PACKET_SHA256:
        raise SamplingHarmAutomatedChallengeError("unexpected review packet binding")

    if tuple(synthesis["required_roles"]) != REQUIRED_ROLES:
        raise SamplingHarmAutomatedChallengeError("required role inventory mismatch")
    report_refs = synthesis["role_reports"]
    if tuple(item["role"] for item in report_refs) != REQUIRED_ROLES:
        raise SamplingHarmAutomatedChallengeError(
            "role-report order or inventory mismatch"
        )

    common = _load_object(root / SCIENTIFIC_SCHEMA_ROOT / "common.schema.json")
    role_schema = _load_object(
        root / SCIENTIFIC_SCHEMA_ROOT / "role-report.schema.json"
    )
    registry = Registry().with_resource(
        str(common["$id"]), Resource.from_contents(common)
    )
    source_findings: dict[str, tuple[str, str]] = {}
    for reference in report_refs:
        path = _safe_report_path(reference["path"])
        report = _load_object(root / path)
        _validate_schema(
            report,
            role_schema,
            label=f"role report {reference['role']}",
            registry=registry,
        )
        digest = canonical_json_sha256(
            report, excluded_json_pointers={"/report_sha256"}
        )
        if digest != report["report_sha256"] or digest != reference["report_sha256"]:
            raise SamplingHarmAutomatedChallengeError(
                f"role-report digest mismatch: {reference['role']}"
            )
        if not reference["digest_verified"]:
            raise SamplingHarmAutomatedChallengeError("digest_verified must be true")
        if (
            report["reviewer_role"] != reference["role"]
            or report["report_id"] != reference["report_id"]
            or report["verdict"] != reference["verdict"]
            or reference["reviewer_eligible"]
        ):
            raise SamplingHarmAutomatedChallengeError(
                f"role-report projection mismatch: {reference['role']}"
            )
        if report["candidate_commit"]["value"] != FROZEN_CANDIDATE_COMMIT:
            raise SamplingHarmAutomatedChallengeError(
                "report candidate commit mismatch"
            )
        if report["candidate_tree"]["value"] != FROZEN_CANDIDATE_TREE:
            raise SamplingHarmAutomatedChallengeError("report candidate tree mismatch")
        if report["packet_sha256"] != FROZEN_PACKET_SHA256:
            raise SamplingHarmAutomatedChallengeError("report packet binding mismatch")
        for finding_id in report["finding_ids"]:
            if finding_id in source_findings:
                raise SamplingHarmAutomatedChallengeError("duplicate source finding id")
            source_findings[finding_id] = (reference["role"], digest)

    findings = synthesis["findings"]
    finding_ids = [item["finding_id"] for item in findings]
    if len(finding_ids) != len(set(finding_ids)) or set(finding_ids) != set(
        source_findings
    ):
        raise SamplingHarmAutomatedChallengeError("synthesized finding union mismatch")
    for finding in findings:
        role, digest = source_findings[finding["finding_id"]]
        if finding["source_role"] != role or finding["source_report_sha256"] != digest:
            raise SamplingHarmAutomatedChallengeError("finding provenance mismatch")
        if finding["disposition"] != "pending":
            raise SamplingHarmAutomatedChallengeError(
                "automated synthesis cannot disposition findings"
            )

    severities = [item["normalized_severity"].lower() for item in findings]
    summary = synthesis["finding_summary"]
    expected_summary = {
        "total": len(findings),
        "critical": severities.count("critical"),
        "high": severities.count("high"),
        "medium": severities.count("medium"),
        "pending": len(findings),
        "resolved": 0,
    }
    if summary != expected_summary:
        raise SamplingHarmAutomatedChallengeError("finding summary mismatch")
    if expected_summary != {
        "total": 19,
        "critical": 1,
        "high": 15,
        "medium": 3,
        "pending": 19,
        "resolved": 0,
    }:
        raise SamplingHarmAutomatedChallengeError(
            "unexpected challenge finding profile"
        )

    actor = synthesis["actor"]
    if actor["human"] or actor["authorizing"] or actor["actor_type"] != "agent":
        raise SamplingHarmAutomatedChallengeError(
            "synthesis actor overclaims authority"
        )
    status = synthesis["gate_status"]
    if not status["required_role_reports_complete"]:
        raise SamplingHarmAutomatedChallengeError("role-shaped coverage is incomplete")
    for field in FALSE_GATE_FIELDS:
        if status[field]:
            raise SamplingHarmAutomatedChallengeError(f"{field} must remain false")
    if synthesis["reviewer_eligibility"]["independent_eligibility_satisfied"]:
        raise SamplingHarmAutomatedChallengeError(
            "reviewer eligibility must remain false"
        )
    if synthesis["source_review"]["source_review_satisfied"]:
        raise SamplingHarmAutomatedChallengeError("source review must remain false")
    if any(synthesis["authority"].values()):
        raise SamplingHarmAutomatedChallengeError(
            "all authority flags must remain false"
        )

    return {
        "synthesis_sha256": actual_digest,
        "role_report_count": len(report_refs),
        "finding_count": len(findings),
        "h8d_satisfied": False,
        "h8e_satisfied": False,
    }


def load_and_validate_sampling_harm_automated_challenge(
    path: Path, *, repository_root: Path
) -> dict[str, object]:
    """Load one synthesis and validate it against repository evidence."""
    return validate_sampling_harm_automated_challenge(
        _load_object(path), repository_root=repository_root
    )
