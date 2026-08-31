"""Validate the historical agent-only assurance record without granting authority."""

from __future__ import annotations

from datetime import UTC, date, datetime
import hashlib
import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import ValidationError

from voiage.sampling_harm_automated_challenge import (
    CONTRACT_ROOT,
    FROZEN_CANDIDATE_COMMIT,
    FROZEN_CANDIDATE_TREE,
    FROZEN_PACKET_SHA256,
    REQUIRED_ROLES,
    SYNTHESIS_PATH,
    SamplingHarmAutomatedChallengeError,
    validate_sampling_harm_automated_challenge,
)
from voiage.sampling_harm_source_readiness import (
    SamplingHarmSourceReadinessError,
    validate_repository_sampling_harm_source_readiness,
)
from voiage.scientific_review_evidence import canonical_json_sha256

ASSURANCE_PATH = CONTRACT_ROOT / "agent-assurance-review-20260804.json"
REPORT_FILENAMES = (
    "h8d-estimand-domain-agent-20260803.json",
    "h8d-estimator-assurance-automated-20260803.json",
    "h8d-cross-language-api-governance-agent-20260803.json",
    "h8d-governance-publication-agent-20260803.json",
    "h8d-domain-specialist-agent-20260803.json",
)
PANEL_ROLES = {
    "source_retrieval_rights_observer",
    "voi_estimand_scientist",
    "domain_ethics_scope_analyst",
    "adversarial_governance_auditor",
    "synthesis_orchestrator",
}
SUPERSESSION_EVENTS = {
    "source_drift",
    "candidate_change",
    "substantive_finding_remediation",
    "new_jurisdiction",
}
PROHIBITED_CLAIMS = {
    "eligible_conflict_free_human_review",
    "credentialed_scientific_or_domain_ethics_expertise",
    "publisher_permission_or_legal_clearance",
    "regulatory_or_ethics_approval",
    "universal_kernel_validity_or_exclusion_authority",
    "real_study_authorization",
    "release_or_publication_readiness",
}


class SamplingHarmAgentAssuranceError(ValueError):
    """Raised when the agent-only record is inconsistent, expired or overclaims."""


def _object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, ValueError) as error:
        raise SamplingHarmAgentAssuranceError(
            f"cannot read JSON object: {path}"
        ) from error
    if not isinstance(value, dict):
        raise SamplingHarmAgentAssuranceError(f"expected JSON object: {path}")
    return value


def _schema(root: Path, value: dict[str, Any], name: str) -> None:
    schema = _object(root / CONTRACT_ROOT / "schemas" / f"{name}.schema.json")
    try:
        Draft202012Validator(schema, format_checker=FormatChecker()).validate(value)
    except ValidationError as error:
        raise SamplingHarmAgentAssuranceError(
            f"{name} schema: {error.message}"
        ) from error


def _reference(root: Path, value: str, expected: Path) -> Path:
    path = root / CONTRACT_ROOT / value
    if (
        value != expected.relative_to(CONTRACT_ROOT).as_posix()
        or path.resolve() != (root / expected).absolute()
        or not path.resolve().is_relative_to(root)
    ):
        raise SamplingHarmAgentAssuranceError(
            f"unexpected or redirected reference: {value}"
        )
    return path


def _raw_reference(
    root: Path, reference: dict[str, Any], expected: Path
) -> dict[str, Any]:
    path = _reference(root, reference["path"], expected)
    value = _object(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != reference["sha256"]:
        raise SamplingHarmAgentAssuranceError(
            f"raw reference digest mismatch: {expected}"
        )
    return value


def load_and_validate_sampling_harm_agent_assurance(
    *,
    repository_root: Path,
    now: datetime | None = None,
    superseded_by: tuple[str, ...] = (),
) -> dict[str, object]:
    """Check this exact historical contract; known supersession fails closed.

    The injected clock supports reproducible tests. Validation does not refresh
    the expired H8-C governance snapshot or establish independent review.
    """
    root = repository_root.resolve()
    record = _object(root / ASSURANCE_PATH)
    _schema(root, record, "agent-assurance-review")
    current = now or datetime.now(UTC)
    if current.tzinfo is None or current.utcoffset() is None:
        raise SamplingHarmAgentAssuranceError("validation time must be timezone-aware")
    if superseded_by:
        raise SamplingHarmAgentAssuranceError("agent assurance has been superseded")
    if set(record["expiry"]["supersede_on"]) != SUPERSESSION_EVENTS:
        raise SamplingHarmAgentAssuranceError("supersession policy mismatch")
    observed = record.get("observed_at")
    if not isinstance(observed, str):
        raise SamplingHarmAgentAssuranceError("observation time is required")
    observed_at = datetime.fromisoformat(observed)
    review_by = date.fromisoformat(record["expiry"]["review_by"])
    if review_by != date(2026, 11, 30):
        raise SamplingHarmAgentAssuranceError("historical review deadline changed")
    if current < observed_at or current.astimezone(UTC).date() > review_by:
        raise SamplingHarmAgentAssuranceError(
            "agent assurance is future-dated or expired"
        )
    if record["candidate_binding"] != {
        "candidate_commit": FROZEN_CANDIDATE_COMMIT,
        "candidate_tree": FROZEN_CANDIDATE_TREE,
        "packet_sha256": FROZEN_PACKET_SHA256,
    }:
        raise SamplingHarmAgentAssuranceError("candidate binding mismatch")
    if set(record["panel_roles"]) != PANEL_ROLES:
        raise SamplingHarmAgentAssuranceError("panel role inventory mismatch")
    if set(record["prohibited_claims"]) != PROHIBITED_CLAIMS or record[
        "source_assessment"
    ] != {
        "receipt_artifact": "agent_observed_only",
        "rights_status": "unresolved",
        "applicability_status": "provisional_narrow_scope_only",
        "source_authority": False,
    }:
        raise SamplingHarmAgentAssuranceError(
            "historical source or claim boundary changed"
        )
    reports = record["panel_reports"]
    if tuple(item["role"] for item in reports) != REQUIRED_ROLES:
        raise SamplingHarmAgentAssuranceError("report role inventory mismatch")

    synthesis = _raw_reference(root, record["synthesis"], SYNTHESIS_PATH)
    _schema(root, synthesis, "automated-challenge-synthesis")
    synthesized_reports = synthesis.get("role_reports", [])
    # Resolve every manifest reference before the composed validator opens files.
    for reference, synthesized, filename in zip(
        reports, synthesized_reports, REPORT_FILENAMES, strict=True
    ):
        expected = CONTRACT_ROOT / "reviews" / filename
        _reference(root, reference["path"], expected)
        if synthesized["path"] != expected.as_posix():
            raise SamplingHarmAgentAssuranceError(
                "historical synthesis report path changed"
            )
    try:
        validate_sampling_harm_automated_challenge(synthesis, repository_root=root)
    except SamplingHarmAutomatedChallengeError as error:
        raise SamplingHarmAgentAssuranceError(
            f"invalid challenge evidence: {error}"
        ) from error
    # The composed validator checks every report's schema before field access.
    for reference, filename in zip(reports, REPORT_FILENAMES, strict=True):
        report = _object(root / CONTRACT_ROOT / "reviews" / filename)
        digest = canonical_json_sha256(
            report, excluded_json_pointers={"/report_sha256"}
        )
        if digest != reference["sha256"]:
            raise SamplingHarmAgentAssuranceError("canonical report digest mismatch")
        if report["reviewer"]["actor_type"] != "agent":
            raise SamplingHarmAgentAssuranceError(
                "report reviewer must remain an agent"
            )
    dissent = synthesis["dissent"]
    if not dissent or any(
        not item["statement"].strip()
        or not item["source_roles"]
        or not set(item["source_roles"]).issubset(REQUIRED_ROLES)
        for item in dissent
    ):
        raise SamplingHarmAgentAssuranceError("dissent is absent or unbound")
    _raw_reference(
        root,
        record["source_receipts"],
        CONTRACT_ROOT / "source-observation-refresh-20260803.json",
    )
    findings = record["findings"]
    register = _raw_reference(
        root,
        {"path": findings["register_path"], "sha256": findings["register_sha256"]},
        CONTRACT_ROOT / "remediation-register.json",
    )
    _schema(root, register, "remediation-register")
    if register["bindings"]["synthesis_sha256"] != synthesis["synthesis_sha256"]:
        raise SamplingHarmAgentAssuranceError("register synthesis binding mismatch")
    ids = [item["finding_id"] for item in register["findings"]]
    if (
        len(ids) != 19
        or len(set(ids)) != 19
        or set(ids) != {item["finding_id"] for item in synthesis["findings"]}
        or any(item["disposition_status"] != "pending" for item in register["findings"])
    ):
        raise SamplingHarmAgentAssuranceError("pending finding inventory mismatch")
    try:
        validate_repository_sampling_harm_source_readiness(root)
    except (SamplingHarmSourceReadinessError, ValueError, OSError) as error:
        raise SamplingHarmAgentAssuranceError(
            f"invalid source readiness: {error}"
        ) from error
    return {
        "status": "valid_historical_agent_only_assurance",
        "evaluated_at": current.isoformat(),
        "review_by": review_by.isoformat(),
        "historical_packet_only": True,
        "qualified_replacement_packet": False,
        "role_report_count": len(reports),
        "pending_findings": len(ids),
        "human_review": "not_performed",
        "source_authority": False,
        "finding_disposition": False,
        "runtime": False,
        "real_study": False,
    }
