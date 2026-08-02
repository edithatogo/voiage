"""Fail-closed validation for governed scientific-review evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
import json
from pathlib import Path, PurePosixPath
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError
from referencing import Registry, Resource

SCHEMA_ROOT = (
    Path(__file__).resolve().parents[1]
    / "specs/frontier/governance/scientific-review/v1/schemas"
)
EVIDENCE_KINDS = (
    "adjudication",
    "artifact-manifest",
    "delta-classification",
    "disagreement",
    "disposition",
    "finding",
    "promotion-receipt",
    "review-packet",
    "reviewer-attestation",
    "role-report",
    "scientific-approval",
)
REQUIRED_REVIEWER_ROLES = {
    "estimand_domain",
    "estimator_assurance",
    "cross_language_api",
    "governance_publication",
}
METADATA_ONLY_PREFIXES = (
    ".github/ISSUE_TEMPLATE/",
    "conductor/",
    "docs/",
)
METADATA_ONLY_FILES = {
    "CHANGELOG.md",
    "README.md",
    "roadmap.md",
    "todo.md",
}


class ScientificReviewEvidenceError(ValueError):
    """Raised when scientific-review evidence is incomplete or inconsistent."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ScientificReviewEvidenceError(
            f"cannot load schema {path}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise ScientificReviewEvidenceError(f"schema {path} must be an object")
    return payload


def _schema_resources() -> tuple[dict[str, dict[str, Any]], Registry]:
    common_path = SCHEMA_ROOT / "common.schema.json"
    common = _load_json(common_path)
    schemas: dict[str, dict[str, Any]] = {}
    resources: list[tuple[str, Resource[Any]]] = []
    for schema in [common]:
        Draft202012Validator.check_schema(schema)
        resources.append((str(schema["$id"]), Resource.from_contents(schema)))
    for kind in EVIDENCE_KINDS:
        schema = _load_json(SCHEMA_ROOT / f"{kind}.schema.json")
        Draft202012Validator.check_schema(schema)
        schemas[kind] = schema
        resources.append((str(schema["$id"]), Resource.from_contents(schema)))
    return schemas, Registry().with_resources(resources)


def load_scientific_review_schemas() -> dict[str, dict[str, Any]]:
    """Load and meta-validate every public scientific-review evidence schema."""
    schemas, _ = _schema_resources()
    return schemas


def validate_scientific_review_evidence(kind: str, payload: object) -> None:
    """Validate one evidence artifact against its versioned JSON Schema."""
    schemas, registry = _schema_resources()
    if kind not in schemas:
        raise ScientificReviewEvidenceError(
            f"unknown scientific-review evidence kind: {kind}"
        )
    try:
        Draft202012Validator(
            schemas[kind], registry=registry, format_checker=FormatChecker()
        ).validate(payload)
    except JsonSchemaValidationError as error:
        location = "/".join(str(item) for item in error.absolute_path) or "$"
        raise ScientificReviewEvidenceError(
            f"{kind} evidence invalid at {location}: {error.message}"
        ) from error


def _as_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ScientificReviewEvidenceError(f"{label} must be an object")
    return value


def _as_sequence(value: object, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ScientificReviewEvidenceError(f"{label} must be an array")
    return value


def _eligible_independent(person: Mapping[str, Any]) -> bool:
    return (
        bool(person.get("independent"))
        and person.get("conflict_status") != "disqualifying"
    )


def _check_binding(
    evidence: Mapping[str, Any], canonical: Mapping[str, Any], label: str
) -> None:
    for field in ("candidate_commit", "candidate_tree", "packet_sha256"):
        if field in evidence and evidence[field] != canonical[field]:
            raise ScientificReviewEvidenceError(
                f"{label} has inconsistent candidate/packet binding for {field}"
            )


def _is_metadata_only(path: str) -> bool:
    normalized = PurePosixPath(path)
    if normalized.is_absolute() or ".." in normalized.parts:
        return False
    text = str(normalized)
    return (
        text in METADATA_ONLY_FILES
        or text.endswith(".md")
        or text.startswith(METADATA_ONLY_PREFIXES)
    )


def _parse_timestamp(value: object, label: str) -> datetime:
    if not isinstance(value, str):
        raise ScientificReviewEvidenceError(f"{label} must be an RFC 3339 timestamp")
    try:
        return datetime.fromisoformat(value)
    except ValueError as error:
        raise ScientificReviewEvidenceError(
            f"{label} must be an RFC 3339 timestamp"
        ) from error


def validate_scientific_review_bundle(bundle: object) -> None:
    """Validate a complete evidence set and its cross-artifact invariants."""
    root = _as_mapping(bundle, "scientific-review bundle")
    if root.get("schema_version") != "1.0.0":
        raise ScientificReviewEvidenceError("bundle schema_version must be 1.0.0")
    evidence = _as_mapping(root.get("evidence"), "bundle.evidence")
    singleton_kinds = {
        "artifact-manifest",
        "review-packet",
        "adjudication",
        "scientific-approval",
        "promotion-receipt",
    }
    repeated_kinds = set(EVIDENCE_KINDS) - singleton_kinds
    for kind in singleton_kinds:
        validate_scientific_review_evidence(kind, evidence.get(kind))
    for kind in repeated_kinds:
        for item in _as_sequence(evidence.get(kind), f"bundle.evidence.{kind}"):
            validate_scientific_review_evidence(kind, item)

    packet = _as_mapping(evidence["review-packet"], "review-packet")
    canonical = {
        field: packet[field]
        for field in ("candidate_commit", "candidate_tree", "packet_sha256")
    }
    for kind, value in evidence.items():
        items = value if kind in repeated_kinds else [value]
        for index, item in enumerate(_as_sequence(items, kind)):
            _check_binding(_as_mapping(item, f"{kind}[{index}]"), canonical, kind)

    reports = _as_sequence(evidence["role-report"], "role-report")
    roles = {
        _as_mapping(report, "role-report").get("reviewer_role") for report in reports
    }
    missing_roles = REQUIRED_REVIEWER_ROLES - roles
    if missing_roles:
        raise ScientificReviewEvidenceError(
            f"required reviewer roles missing: {', '.join(sorted(missing_roles))}"
        )
    attestations = {
        _as_mapping(_as_mapping(value, "attestation")["reviewer"], "reviewer")[
            "identity"
        ]
        for value in _as_sequence(
            evidence["reviewer-attestation"], "reviewer-attestation"
        )
    }
    for report in reports:
        reviewer = _as_mapping(
            _as_mapping(report, "role-report")["reviewer"], "reviewer"
        )
        if not _eligible_independent(reviewer):
            raise ScientificReviewEvidenceError(
                "role reports require eligible independent reviewers"
            )
        if reviewer["identity"] not in attestations:
            raise ScientificReviewEvidenceError(
                f"role report reviewer lacks a matching attestation: {reviewer['identity']}"
            )

    findings = {
        item["finding_id"]: item
        for item in (
            _as_mapping(value, "finding")
            for value in _as_sequence(evidence["finding"], "finding")
        )
    }
    dispositions = {
        item["finding_id"]: item
        for item in (
            _as_mapping(value, "disposition")
            for value in _as_sequence(evidence["disposition"], "disposition")
        )
    }
    for finding_id, finding in findings.items():
        if finding["severity"] not in {"critical", "high"}:
            continue
        disposition = dispositions.get(finding_id)
        if (
            disposition is None
            or disposition["disposition"] not in {"fixed", "reviewed_exclusion"}
            or not disposition["independently_verified"]
        ):
            raise ScientificReviewEvidenceError(
                f"unresolved Critical/High finding blocks acceptance: {finding_id}"
            )

    approval = _as_mapping(evidence["scientific-approval"], "scientific-approval")
    approver = _as_mapping(approval["approver"], "scientific-approval.approver")
    if not _eligible_independent(approver):
        raise ScientificReviewEvidenceError(
            "scientific approval requires an eligible independent approver"
        )
    adjudication = _as_mapping(evidence["adjudication"], "adjudication")
    chair = _as_mapping(adjudication["chair"], "adjudication.chair")
    if not _eligible_independent(chair):
        raise ScientificReviewEvidenceError(
            "adjudication requires an eligible independent chair"
        )
    scope = packet["scope"]
    for label, artifact in (
        ("adjudication", adjudication),
        ("scientific-approval", approval),
        (
            "promotion-receipt",
            _as_mapping(evidence["promotion-receipt"], "promotion-receipt"),
        ),
    ):
        if artifact["scope"] != scope:
            raise ScientificReviewEvidenceError(
                f"{label} scope differs from review-packet scope"
            )
    if approval["decision"] != adjudication["decision"]:
        raise ScientificReviewEvidenceError(
            "scientific approval decision differs from adjudication decision"
        )
    open_scientific_dissent = any(
        item["scientific_validity_dissent"] and item["status"] != "resolved"
        for item in (
            _as_mapping(value, "disagreement")
            for value in _as_sequence(evidence["disagreement"], "disagreement")
        )
    )
    if (
        open_scientific_dissent
        and approval["decision"] == "scientifically_acceptable_experimental"
    ):
        raise ScientificReviewEvidenceError(
            "unresolved scientific-validity dissent blocks positive approval"
        )

    for value in _as_sequence(evidence["delta-classification"], "delta-classification"):
        delta = _as_mapping(value, "delta-classification")
        if delta["classification"] == "bounded_metadata_only":
            if not all(_is_metadata_only(path) for path in delta["changed_paths"]):
                raise ScientificReviewEvidenceError(
                    "bounded delta contains a non-metadata-only path"
                )
            signatures = _as_sequence(delta["signatures"], "delta signatures")
            roles = {
                _as_mapping(signature, "delta signature")["reviewer_role"]
                for signature in signatures
            }
            if len(signatures) < 2 or not {
                "governance_publication",
                "affected_scientific",
            }.issubset(roles):
                raise ScientificReviewEvidenceError(
                    "bounded delta requires two independent governance/scientific signatures"
                )
            if not all(
                _eligible_independent(
                    _as_mapping(
                        _as_mapping(signature, "signature")["reviewer"], "reviewer"
                    )
                )
                for signature in signatures
            ):
                raise ScientificReviewEvidenceError(
                    "bounded delta signatures must be eligible and independent"
                )

    promotion = _as_mapping(evidence["promotion-receipt"], "promotion-receipt")
    if promotion["scientific_approval_sha256"] != approval["approval_sha256"]:
        raise ScientificReviewEvidenceError(
            "promotion receipt is not bound to the scientific approval digest"
        )
    if promotion["decision"] == "promote":
        if approval["decision"] != "scientifically_acceptable_experimental":
            raise ScientificReviewEvidenceError(
                "promotion requires a scientifically acceptable decision"
            )
        if approval.get("superseded_by"):
            raise ScientificReviewEvidenceError(
                "superseded scientific approval cannot authorize promotion"
            )
        if _parse_timestamp(
            approval["expires_at"], "approval.expires_at"
        ) <= _parse_timestamp(promotion["decision_at"], "promotion.decision_at"):
            raise ScientificReviewEvidenceError(
                "expired scientific approval cannot authorize promotion"
            )
