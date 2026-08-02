"""Fail-closed validation for governed scientific-review evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
from typing import Any, cast

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
SCHEMA_VERSION = "1.1.0"
SELF_DIGEST_FIELDS = {
    "artifact-manifest": "manifest_sha256",
    "review-packet": "packet_sha256",
    "reviewer-attestation": "attestation_sha256",
    "role-report": "report_sha256",
    "adjudication": "adjudication_sha256",
    "scientific-approval": "approval_sha256",
    "promotion-receipt": "receipt_sha256",
}
HUMAN_RECEIPT_METHODS = {
    "signed_commit",
    "authenticated_github",
    "external_authoritative_receipt",
}
GIT_EXECUTABLE = shutil.which("git")


class ScientificReviewEvidenceError(ValueError):
    """Raised when scientific-review evidence is incomplete or inconsistent."""


def _remove_json_pointer(payload: dict[str, Any], pointer: str) -> None:
    """Remove one simple RFC 6901 pointer from a copied JSON object."""
    if not pointer.startswith("/"):
        raise ValueError(f"JSON pointer must be absolute: {pointer}")
    parts = [
        part.replace("~1", "/").replace("~0", "~") for part in pointer[1:].split("/")
    ]
    current: Any = payload
    for part in parts[:-1]:
        if not isinstance(current, dict) or part not in current:
            return
        current = current[part]
    if isinstance(current, dict):
        current.pop(parts[-1], None)


def canonical_json_sha256(
    payload: Mapping[str, Any], *, excluded_json_pointers: set[str] | None = None
) -> str:
    """Return SHA-256 over canonical UTF-8 JSON with selected fields excluded."""
    canonical_payload = deepcopy(dict(payload))
    for pointer in sorted(excluded_json_pointers or set()):
        _remove_json_pointer(canonical_payload, pointer)
    encoded = json.dumps(
        canonical_payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _declared_digest(kind: str, payload: Mapping[str, Any]) -> str:
    field = SELF_DIGEST_FIELDS[kind]
    excluded = {f"/{field}"}
    if "human_receipt" in payload:
        excluded.add("/human_receipt/payload_sha256")
    return canonical_json_sha256(payload, excluded_json_pointers=excluded)


def bind_scientific_review_bundle(bundle: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy with canonical bindings and declared digests refreshed."""
    bound = deepcopy(dict(bundle))
    bound["schema_version"] = SCHEMA_VERSION
    evidence = bound["evidence"]
    for value in evidence.values():
        for item in value if isinstance(value, list) else [value]:
            item["schema_version"] = SCHEMA_VERSION

    packet = evidence["review-packet"]
    candidate_binding = {
        "candidate_commit": packet["candidate_commit"],
        "candidate_tree": packet["candidate_tree"],
    }
    for value in evidence.values():
        for item in value if isinstance(value, list) else [value]:
            for field, field_value in candidate_binding.items():
                if field in item:
                    item[field] = deepcopy(field_value)

    manifest = evidence["artifact-manifest"]
    manifest["manifest_sha256"] = _declared_digest("artifact-manifest", manifest)
    packet["artifact_manifest_sha256"] = manifest["manifest_sha256"]
    packet["packet_sha256"] = _declared_digest("review-packet", packet)
    for value in evidence.values():
        for item in value if isinstance(value, list) else [value]:
            if "packet_sha256" in item:
                item["packet_sha256"] = packet["packet_sha256"]

    for kind in (
        "reviewer-attestation",
        "role-report",
        "adjudication",
        "scientific-approval",
    ):
        value = evidence[kind]
        for item in value if isinstance(value, list) else [value]:
            digest = _declared_digest(kind, item)
            item[SELF_DIGEST_FIELDS[kind]] = digest
            if "human_receipt" in item:
                item["human_receipt"]["payload_sha256"] = digest

    approval = evidence["scientific-approval"]
    promotion = evidence["promotion-receipt"]
    promotion["scientific_approval_sha256"] = approval["approval_sha256"]
    promotion_digest = _declared_digest("promotion-receipt", promotion)
    promotion["receipt_sha256"] = promotion_digest
    promotion["human_receipt"]["payload_sha256"] = promotion_digest
    bound["expected_finding_ids"] = sorted(
        item["finding_id"] for item in evidence["finding"]
    )
    bound["expected_disagreement_ids"] = sorted(
        item["disagreement_id"] for item in evidence["disagreement"]
    )
    return bound


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
    return cast("Mapping[str, Any]", value)


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


def _is_administrative_only(path: str, changed_fields: Sequence[Any]) -> bool:
    """Return whether a delta is limited to an explicitly safe admin field."""
    normalized = PurePosixPath(path)
    if normalized.is_absolute() or ".." in normalized.parts:
        return False
    text = str(normalized)
    fields = set(changed_fields)
    if text.startswith("conductor/tracks/") and text.endswith("/metadata.json"):
        return bool(fields) and fields <= {"/updated_at"}
    if "governance-readback" in normalized.name and text.startswith("conductor/"):
        return bool(fields) and fields <= {"/observed_at", "/retrieved_at"}
    return False


def _parse_timestamp(value: object, label: str) -> datetime:
    if not isinstance(value, str):
        raise ScientificReviewEvidenceError(f"{label} must be an RFC 3339 timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ScientificReviewEvidenceError(
            f"{label} must be an RFC 3339 timestamp"
        ) from error
    if parsed.tzinfo is None:
        raise ScientificReviewEvidenceError(f"{label} must include a UTC offset")
    return parsed


def _verify_human_receipt(
    payload: Mapping[str, Any],
    digest: str,
    label: str,
    *,
    expected_identity: str | None = None,
) -> None:
    receipt = _as_mapping(payload.get("human_receipt"), f"{label}.human_receipt")
    if receipt.get("verification_method") not in HUMAN_RECEIPT_METHODS:
        raise ScientificReviewEvidenceError(
            f"{label} uses an unsupported human receipt verification method"
        )
    if receipt.get("payload_sha256") != digest:
        raise ScientificReviewEvidenceError(
            f"{label} human receipt is not bound to its canonical digest"
        )
    if receipt.get("verification_status") != "verified":
        raise ScientificReviewEvidenceError(f"{label} human receipt is not verified")
    if (
        expected_identity is not None
        and receipt.get("signer_identity") != expected_identity
    ):
        raise ScientificReviewEvidenceError(
            f"{label} human receipt signer does not match the accountable identity"
        )


def _verify_declared_digests(evidence: Mapping[str, Any]) -> None:
    for kind, field in SELF_DIGEST_FIELDS.items():
        value = evidence[kind]
        for index, item in enumerate(value if isinstance(value, list) else [value]):
            artifact = _as_mapping(item, f"{kind}[{index}]")
            expected = _declared_digest(kind, artifact)
            if artifact.get(field) != expected:
                raise ScientificReviewEvidenceError(
                    f"{kind} canonical digest does not match {field}"
                )
            if "human_receipt" in artifact:
                _verify_human_receipt(artifact, expected, kind)
    manifest = _as_mapping(evidence["artifact-manifest"], "artifact-manifest")
    packet = _as_mapping(evidence["review-packet"], "review-packet")
    if packet["artifact_manifest_sha256"] != manifest["manifest_sha256"]:
        raise ScientificReviewEvidenceError(
            "review-packet is not bound to the canonical artifact-manifest digest"
        )


def _git_output(repository_root: Path, *arguments: str, binary: bool = False) -> Any:
    if GIT_EXECUTABLE is None:
        raise ScientificReviewEvidenceError(
            "cannot verify frozen Git evidence: git is unavailable"
        )
    command = [GIT_EXECUTABLE, "-C", str(repository_root), *arguments]
    try:
        return subprocess.check_output(command, text=not binary)  # noqa: S603
    except (OSError, subprocess.CalledProcessError) as error:
        raise ScientificReviewEvidenceError(
            f"cannot verify frozen Git evidence with {' '.join(arguments)}"
        ) from error


def _verify_repository_evidence(
    evidence: Mapping[str, Any], repository_root: Path
) -> None:
    packet = _as_mapping(evidence["review-packet"], "review-packet")
    commit = _as_mapping(packet["candidate_commit"], "candidate_commit")
    tree = _as_mapping(packet["candidate_tree"], "candidate_tree")
    object_format = str(
        _git_output(repository_root, "rev-parse", "--show-object-format")
    ).strip()
    if commit["algorithm"] != object_format or tree["algorithm"] != object_format:
        raise ScientificReviewEvidenceError(
            "candidate Git OID algorithm differs from repository object format"
        )
    actual_tree = str(
        _git_output(repository_root, "rev-parse", f"{commit['value']}^{{tree}}")
    ).strip()
    if actual_tree != tree["value"]:
        raise ScientificReviewEvidenceError(
            "candidate tree does not match the frozen candidate commit"
        )
    manifest = _as_mapping(evidence["artifact-manifest"], "artifact-manifest")
    seen_paths: set[str] = set()
    for item in _as_sequence(manifest["artifacts"], "manifest.artifacts"):
        artifact = _as_mapping(item, "manifest artifact")
        path = PurePosixPath(str(artifact["path"]))
        if path.is_absolute() or ".." in path.parts:
            raise ScientificReviewEvidenceError("manifest artifact path is unsafe")
        if str(path) in seen_paths:
            raise ScientificReviewEvidenceError(
                "manifest artifact paths must be unique"
            )
        seen_paths.add(str(path))
        content = _git_output(
            repository_root,
            "show",
            f"{commit['value']}:{path}",
            binary=True,
        )
        if hashlib.sha256(content).hexdigest() != artifact["sha256"]:
            raise ScientificReviewEvidenceError(
                f"manifest artifact bytes do not match frozen tree: {path}"
            )


def validate_scientific_review_bundle(
    bundle: object,
    *,
    repository_root: Path | None = None,
    at_time: datetime | None = None,
) -> None:
    """Validate a complete evidence set and its cross-artifact invariants."""
    root = _as_mapping(bundle, "scientific-review bundle")
    if root.get("schema_version") != SCHEMA_VERSION:
        raise ScientificReviewEvidenceError(
            f"bundle schema_version must be {SCHEMA_VERSION}"
        )
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

    for kind in ("artifact-manifest", "review-packet"):
        artifact = _as_mapping(evidence[kind], kind)
        field = SELF_DIGEST_FIELDS[kind]
        if artifact[field] != _declared_digest(kind, artifact):
            raise ScientificReviewEvidenceError(
                f"{kind} canonical digest does not match {field}"
            )

    packet = _as_mapping(evidence["review-packet"], "review-packet")
    canonical = {
        field: packet[field]
        for field in ("candidate_commit", "candidate_tree", "packet_sha256")
    }
    for kind, value in evidence.items():
        items = value if kind in repeated_kinds else [value]
        for index, item in enumerate(_as_sequence(items, kind)):
            _check_binding(_as_mapping(item, f"{kind}[{index}]"), canonical, kind)

    expected_findings = set(
        _as_sequence(root.get("expected_finding_ids"), "expected_finding_ids")
    )
    expected_disagreements = set(
        _as_sequence(root.get("expected_disagreement_ids"), "expected_disagreement_ids")
    )

    reports = _as_sequence(evidence["role-report"], "role-report")
    roles = {
        _as_mapping(report, "role-report").get("reviewer_role") for report in reports
    }
    missing_roles = REQUIRED_REVIEWER_ROLES - roles
    if missing_roles:
        raise ScientificReviewEvidenceError(
            f"required reviewer roles missing: {', '.join(sorted(missing_roles))}"
        )
    attestation_items = [
        _as_mapping(value, "attestation")
        for value in _as_sequence(
            evidence["reviewer-attestation"], "reviewer-attestation"
        )
    ]
    attestations = {
        _as_mapping(item["reviewer"], "reviewer")["identity"]: item
        for item in attestation_items
    }
    if len(attestations) != len(attestation_items):
        raise ScientificReviewEvidenceError(
            "reviewer attestations require unique identities"
        )
    report_ids: set[str] = set()
    reports_by_id: dict[str, Mapping[str, Any]] = {}
    reported_findings: set[str] = set()
    report_reviewer_ids: set[str] = set()
    for report_value in reports:
        report = _as_mapping(report_value, "role-report")
        if report["report_id"] in report_ids:
            raise ScientificReviewEvidenceError("role report IDs must be unique")
        report_ids.add(report["report_id"])
        reports_by_id[report["report_id"]] = report
        reported_findings.update(report["finding_ids"])
        reviewer = _as_mapping(report["reviewer"], "reviewer")
        if not _eligible_independent(reviewer):
            raise ScientificReviewEvidenceError(
                "role reports require eligible independent reviewers"
            )
        identity = reviewer["identity"]
        report_reviewer_ids.add(identity)
        attestation = attestations.get(identity)
        if attestation is None:
            raise ScientificReviewEvidenceError(
                f"role report reviewer lacks a matching attestation: {identity}"
            )
        if (
            reviewer != attestation["reviewer"]
            or report["scope"] != attestation["scope"]
        ):
            raise ScientificReviewEvidenceError(
                f"role report must match its reviewer attestation exactly: {identity}"
            )
        if reviewer["actor_type"] == "human":
            _verify_human_receipt(
                report,
                _declared_digest("role-report", report),
                "role-report",
                expected_identity=identity,
            )
            _verify_human_receipt(
                attestation,
                _declared_digest("reviewer-attestation", attestation),
                "reviewer-attestation",
                expected_identity=identity,
            )

    finding_items = [
        _as_mapping(value, "finding")
        for value in _as_sequence(evidence["finding"], "finding")
    ]
    findings = {item["finding_id"]: item for item in finding_items}
    if len(findings) != len(finding_items):
        raise ScientificReviewEvidenceError("finding IDs must be unique")
    if set(findings) != expected_findings:
        raise ScientificReviewEvidenceError(
            "finding inventory differs from expected_finding_ids"
        )
    if reported_findings != set(findings):
        raise ScientificReviewEvidenceError(
            "role-report finding inventory is incomplete"
        )
    disposition_items = [
        _as_mapping(value, "disposition")
        for value in _as_sequence(evidence["disposition"], "disposition")
    ]
    dispositions = {item["finding_id"]: item for item in disposition_items}
    if len(dispositions) != len(disposition_items):
        raise ScientificReviewEvidenceError(
            "each finding may have only one disposition"
        )
    excluded_capabilities: set[str] = set()
    for finding_id, finding in findings.items():
        disposition = dispositions.get(finding_id)
        if (
            disposition is not None
            and disposition["disposition"] == "reviewed_exclusion"
        ):
            excluded = set(disposition["excluded_capabilities"])
            if not excluded or not excluded <= set(packet["scope"]["capabilities"]):
                raise ScientificReviewEvidenceError(
                    f"reviewed exclusion requires an excluded capability: {finding_id}"
                )
            excluded_capabilities.update(excluded)
        if finding["severity"] == "low":
            if disposition is None:
                raise ScientificReviewEvidenceError(
                    f"Low finding requires an explicit disposition: {finding_id}"
                )
            continue
        if disposition is None or not disposition["independently_verified"]:
            category = (
                "Critical/High"
                if finding["severity"] in {"critical", "high"}
                else finding["severity"].title()
            )
            raise ScientificReviewEvidenceError(
                f"unresolved {category} finding blocks acceptance: {finding_id}"
            )
        if finding["severity"] in {"critical", "high"} and disposition[
            "disposition"
        ] not in {"fixed", "reviewed_exclusion"}:
            raise ScientificReviewEvidenceError(
                f"unresolved Critical/High finding blocks acceptance: {finding_id}"
            )
        if finding["severity"] == "medium":
            rereviews = set(disposition["rereview_report_ids"])
            disposition_at = _parse_timestamp(
                disposition["decided_at"], f"disposition[{finding_id}].decided_at"
            )
            affected_roles = set(disposition["affected_reviewer_roles"])
            rereview_roles = {
                reports_by_id[report_id]["reviewer_role"]
                for report_id in rereviews
                if report_id in reports_by_id
            }
            if (
                not rereviews
                or not rereviews <= report_ids
                or not affected_roles
                or not affected_roles <= rereview_roles
                or any(
                    _parse_timestamp(
                        reports_by_id[report_id]["signed_at"],
                        f"role-report[{report_id}].signed_at",
                    )
                    <= disposition_at
                    for report_id in rereviews
                )
            ):
                raise ScientificReviewEvidenceError(
                    f"Medium finding requires affected-role re-review: {finding_id}"
                )

    unknown_dispositions = set(dispositions) - set(findings)
    if unknown_dispositions:
        raise ScientificReviewEvidenceError(
            "finding disposition inventory contains unknown finding IDs"
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
    if chair["actor_type"] != "human" or approver["actor_type"] != "human":
        raise ScientificReviewEvidenceError(
            "adjudication chair and scientific approver must be human"
        )
    _verify_human_receipt(
        adjudication,
        _declared_digest("adjudication", adjudication),
        "adjudication",
        expected_identity=chair["identity"],
    )
    _verify_human_receipt(
        approval,
        _declared_digest("scientific-approval", approval),
        "scientific-approval",
        expected_identity=approver["identity"],
    )
    for identity, person in (
        (chair["identity"], chair),
        (approver["identity"], approver),
    ):
        attestation = attestations.get(identity)
        if attestation is None or attestation["reviewer"] != person:
            raise ScientificReviewEvidenceError(
                f"decision maker lacks an exactly matching attestation: {identity}"
            )
        _verify_human_receipt(
            attestation,
            _declared_digest("reviewer-attestation", attestation),
            "reviewer-attestation",
            expected_identity=identity,
        )
    orchestrator = packet["orchestrator_identity"]
    prohibited = (
        set(packet["author_identities"])
        | set(packet["remediator_identities"])
        | {orchestrator}
    )
    decision_ids = {chair["identity"], approver["identity"]}
    if (
        len(report_reviewer_ids) != len(reports)
        or len(decision_ids) != 2
        or report_reviewer_ids & decision_ids
        or (report_reviewer_ids | decision_ids) & prohibited
    ):
        raise ScientificReviewEvidenceError(
            "scientific review separation of duties is violated"
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
    disagreement_items = [
        _as_mapping(value, "disagreement")
        for value in _as_sequence(evidence["disagreement"], "disagreement")
    ]
    disagreement_ids = {item["disagreement_id"] for item in disagreement_items}
    if (
        len(disagreement_ids) != len(disagreement_items)
        or disagreement_ids != expected_disagreements
    ):
        raise ScientificReviewEvidenceError(
            "disagreement inventory differs from expected_disagreement_ids"
        )
    for label, artifact in (
        ("adjudication", adjudication),
        ("scientific-approval", approval),
        (
            "promotion-receipt",
            _as_mapping(evidence["promotion-receipt"], "promotion-receipt"),
        ),
    ):
        if set(artifact["dissent_refs"]) != disagreement_ids:
            raise ScientificReviewEvidenceError(
                f"{label} dissent references do not preserve the disagreement inventory"
            )
    open_scientific_dissent = any(
        item["scientific_validity_dissent"] and item["status"] != "resolved"
        for item in disagreement_items
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
        changed_paths = list(delta["changed_paths"])
        hash_items = [
            _as_mapping(item, "changed artifact hash")
            for item in _as_sequence(
                delta["changed_artifact_hashes"], "changed_artifact_hashes"
            )
        ]
        hash_paths = [item["path"] for item in hash_items]
        if (
            len(hash_paths) != len(set(hash_paths))
            or set(changed_paths) != set(hash_paths)
            or any(item["before_sha256"] == item["after_sha256"] for item in hash_items)
        ):
            raise ScientificReviewEvidenceError(
                "delta changed-path inventory must exactly match changed artifact hashes"
            )
        if delta["classification"] == "bounded_metadata_only":
            if not all(
                _is_administrative_only(item["path"], item["changed_fields"])
                for item in hash_items
            ):
                raise ScientificReviewEvidenceError(
                    "bounded delta violates the administrative allowlist"
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
            signer_ids = {
                _as_mapping(
                    _as_mapping(signature, "signature")["reviewer"], "reviewer"
                )["identity"]
                for signature in signatures
            }
            if len(signer_ids) != len(signatures):
                raise ScientificReviewEvidenceError(
                    "bounded delta signatures must be from distinct people"
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
            if any(
                attestations.get(
                    _as_mapping(
                        _as_mapping(signature, "signature")["reviewer"], "reviewer"
                    )["identity"]
                )
                is None
                or attestations[
                    _as_mapping(
                        _as_mapping(signature, "signature")["reviewer"], "reviewer"
                    )["identity"]
                ]["reviewer"]
                != _as_mapping(
                    _as_mapping(signature, "signature")["reviewer"], "reviewer"
                )
                for signature in signatures
            ):
                raise ScientificReviewEvidenceError(
                    "bounded delta signer must match a reviewer attestation exactly"
                )

    promotion = _as_mapping(evidence["promotion-receipt"], "promotion-receipt")
    maintainer = _as_mapping(promotion["maintainer"], "promotion-receipt.maintainer")
    if maintainer["actor_type"] != "human":
        raise ScientificReviewEvidenceError(
            "promotion decision requires a human maintainer"
        )
    _verify_human_receipt(
        promotion,
        _declared_digest("promotion-receipt", promotion),
        "promotion-receipt",
        expected_identity=maintainer["identity"],
    )
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
        if excluded_capabilities & set(promotion["scope"]["capabilities"]):
            raise ScientificReviewEvidenceError(
                "promotion scope includes a reviewed excluded capability"
            )
        if _parse_timestamp(
            approval["expires_at"], "approval.expires_at"
        ) <= _parse_timestamp(promotion["decision_at"], "promotion.decision_at"):
            raise ScientificReviewEvidenceError(
                "expired scientific approval cannot authorize promotion"
            )
    now = at_time or datetime.now(UTC)
    for label, decision in (
        ("adjudication", adjudication),
        ("approval", approval),
        ("promotion", promotion),
    ):
        decision_at = _parse_timestamp(decision["decision_at"], f"{label}.decision_at")
        expires_at = _parse_timestamp(decision["expires_at"], f"{label}.expires_at")
        if expires_at <= decision_at:
            raise ScientificReviewEvidenceError(
                f"{label}.expires_at must be after decision_at"
            )
        if decision.get("superseded_by"):
            raise ScientificReviewEvidenceError(f"{label} decision is superseded")
        if (
            approval["decision"] == "scientifically_acceptable_experimental"
            and expires_at <= now
        ):
            raise ScientificReviewEvidenceError(f"{label} decision is expired")

    _verify_declared_digests(evidence)
    if root.get("fixture_status") != "synthetic_contract_example_not_an_approval":
        if repository_root is None:
            raise ScientificReviewEvidenceError(
                "non-synthetic evidence requires repository_root verification"
            )
        _verify_repository_evidence(evidence, repository_root)
