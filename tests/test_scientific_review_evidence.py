"""Scientific-review evidence must be structured, bound, and fail closed."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

import voiage.scientific_review_evidence as review_evidence
from voiage.scientific_review_evidence import (
    ScientificReviewEvidenceError,
    load_scientific_review_schemas,
    validate_scientific_review_bundle,
    validate_scientific_review_evidence,
)

ROOT = Path(__file__).parents[1]
FIXTURE = (
    ROOT
    / "specs/frontier/governance/scientific-review/v1/fixtures/valid-review-bundle.json"
)


def _bundle() -> dict[str, object]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_all_scientific_review_evidence_schemas_are_valid() -> None:
    schemas = load_scientific_review_schemas()

    assert set(schemas) == {
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
    }


@pytest.mark.parametrize(
    ("schema_contents", "message"),
    [
        (None, "cannot load schema"),
        ("{", "cannot load schema"),
        ("[]", "must be an object"),
    ],
)
def test_schema_loading_fails_closed_for_missing_or_malformed_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    schema_contents: str | None,
    message: str,
) -> None:
    schema_root = tmp_path / "schemas"
    schema_root.mkdir()
    if schema_contents is not None:
        (schema_root / "common.schema.json").write_text(
            schema_contents, encoding="utf-8"
        )
    monkeypatch.setattr(review_evidence, "SCHEMA_ROOT", schema_root)

    with pytest.raises(ScientificReviewEvidenceError, match=message):
        load_scientific_review_schemas()


def test_valid_review_bundle_is_candidate_bound_and_complete() -> None:
    bundle = _bundle()

    validate_scientific_review_bundle(bundle)
    for kind, payload in bundle["evidence"].items():
        if kind in {"reviewer-attestation", "role-report"}:
            for item in payload:
                validate_scientific_review_evidence(kind, item)
        elif kind not in {
            "finding",
            "disagreement",
            "disposition",
            "delta-classification",
        }:
            validate_scientific_review_evidence(kind, payload)


@pytest.mark.parametrize(
    "field", ["candidate_commit", "candidate_tree", "packet_sha256"]
)
def test_mixed_candidate_or_packet_binding_is_rejected(field: str) -> None:
    bundle = _bundle()
    bundle["evidence"]["scientific-approval"][field] = "f" * 64

    with pytest.raises(ScientificReviewEvidenceError, match="binding"):
        validate_scientific_review_bundle(bundle)


def test_boolean_only_or_conflicted_approval_is_rejected() -> None:
    approval = _bundle()["evidence"]["scientific-approval"]
    boolean_only = {
        "approved": True,
        "candidate_commit": approval["candidate_commit"],
    }

    with pytest.raises(ScientificReviewEvidenceError):
        validate_scientific_review_evidence("scientific-approval", boolean_only)

    conflicted = deepcopy(approval)
    conflicted["approver"]["conflict_status"] = "disqualifying"
    conflicted["approver"]["independent"] = False
    with pytest.raises(ScientificReviewEvidenceError, match="eligible independent"):
        validate_scientific_review_bundle(
            {
                **_bundle(),
                "evidence": {
                    **_bundle()["evidence"],
                    "scientific-approval": conflicted,
                },
            }
        )


def test_unknown_kind_and_malformed_bundle_shapes_are_rejected() -> None:
    with pytest.raises(ScientificReviewEvidenceError, match="unknown"):
        validate_scientific_review_evidence("boolean-approval", {"approved": True})
    with pytest.raises(ScientificReviewEvidenceError, match="must be an object"):
        validate_scientific_review_bundle(None)

    bundle = _bundle()
    bundle["schema_version"] = "2.0.0"
    with pytest.raises(ScientificReviewEvidenceError, match="schema_version"):
        validate_scientific_review_bundle(bundle)

    bundle = _bundle()
    bundle["evidence"]["disagreement"] = "not-an-array"
    with pytest.raises(ScientificReviewEvidenceError, match="must be an array"):
        validate_scientific_review_bundle(bundle)


def test_required_roles_and_unresolved_high_findings_block_acceptance() -> None:
    bundle = _bundle()
    bundle["evidence"]["role-report"] = bundle["evidence"]["role-report"][:-1]
    with pytest.raises(ScientificReviewEvidenceError, match="required reviewer roles"):
        validate_scientific_review_bundle(bundle)

    bundle = _bundle()
    bundle["evidence"]["finding"] = [
        {
            **bundle["evidence"]["finding"][0],
            "severity": "high",
            "finding_id": "F-HIGH-OPEN",
        }
    ]
    bundle["evidence"]["disposition"] = []
    with pytest.raises(ScientificReviewEvidenceError, match="Critical/High"):
        validate_scientific_review_bundle(bundle)


def test_independently_verified_high_finding_is_accepted() -> None:
    bundle = _bundle()
    bundle["evidence"]["finding"][0]["severity"] = "high"

    validate_scientific_review_bundle(bundle)


def test_every_report_requires_a_matching_attestation_and_scope() -> None:
    bundle = _bundle()
    bundle["evidence"]["reviewer-attestation"] = bundle["evidence"][
        "reviewer-attestation"
    ][:-1]
    with pytest.raises(ScientificReviewEvidenceError, match="attestation"):
        validate_scientific_review_bundle(bundle)

    bundle = _bundle()
    bundle["evidence"]["adjudication"]["scope"]["family"] = "different-family"
    with pytest.raises(ScientificReviewEvidenceError, match="scope"):
        validate_scientific_review_bundle(bundle)


def test_open_scientific_dissent_blocks_positive_approval() -> None:
    bundle = _bundle()
    bundle["evidence"]["disagreement"] = [
        {
            "evidence_kind": "disagreement",
            "schema_version": "1.0.0",
            "disagreement_id": "DISSENT-1",
            "candidate_commit": "a" * 64,
            "candidate_tree": "b" * 64,
            "packet_sha256": "e" * 64,
            "topic": "scientific validity",
            "positions": [
                {
                    "reviewer_identity": "reviewer-estimand",
                    "position": "valid",
                    "evidence_refs": ["REPORT-ESTIMAND"],
                },
                {
                    "reviewer_identity": "reviewer-estimator",
                    "position": "not valid",
                    "evidence_refs": ["REPORT-ESTIMATOR"],
                },
            ],
            "scientific_validity_dissent": True,
            "resolution_evidence_required": "Independent reconstruction",
            "status": "open",
            "recorded_at": "2026-08-02T00:00:00Z",
        }
    ]
    with pytest.raises(ScientificReviewEvidenceError, match="dissent"):
        validate_scientific_review_bundle(bundle)


def test_ineligible_reviewers_and_inconsistent_decisions_are_rejected() -> None:
    bundle = _bundle()
    bundle["evidence"]["role-report"][0]["reviewer"]["independent"] = False
    with pytest.raises(ScientificReviewEvidenceError, match="role reports"):
        validate_scientific_review_bundle(bundle)

    bundle = _bundle()
    bundle["evidence"]["adjudication"]["chair"]["conflict_status"] = "disqualifying"
    with pytest.raises(ScientificReviewEvidenceError, match="chair"):
        validate_scientific_review_bundle(bundle)

    bundle = _bundle()
    bundle["evidence"]["adjudication"]["decision"] = "reviewed_exclusion"
    with pytest.raises(ScientificReviewEvidenceError, match="decision differs"):
        validate_scientific_review_bundle(bundle)


def test_bounded_delta_requires_metadata_only_paths_and_two_signatures() -> None:
    bundle = _bundle()
    delta = bundle["evidence"]["delta-classification"][0]
    delta["changed_paths"] = ["voiage/methods/utility_information.py"]
    with pytest.raises(ScientificReviewEvidenceError, match="metadata-only"):
        validate_scientific_review_bundle(bundle)

    bundle = _bundle()
    bundle["evidence"]["delta-classification"][0]["signatures"] = [
        bundle["evidence"]["delta-classification"][0]["signatures"][0]
    ]
    with pytest.raises(ScientificReviewEvidenceError, match="two independent"):
        validate_scientific_review_bundle(bundle)

    bundle = _bundle()
    bundle["evidence"]["delta-classification"][0]["signatures"][0]["reviewer"][
        "independent"
    ] = False
    with pytest.raises(ScientificReviewEvidenceError, match="eligible and independent"):
        validate_scientific_review_bundle(bundle)


@pytest.mark.parametrize("changed_path", ["/absolute.md", "docs/../review.md"])
def test_bounded_delta_rejects_absolute_and_parent_traversal_paths(
    changed_path: str,
) -> None:
    bundle = _bundle()
    bundle["evidence"]["delta-classification"][0]["changed_paths"] = [changed_path]

    with pytest.raises(ScientificReviewEvidenceError, match="metadata-only"):
        validate_scientific_review_bundle(bundle)


def test_full_invalidation_delta_does_not_use_metadata_only_exception() -> None:
    bundle = _bundle()
    delta = bundle["evidence"]["delta-classification"][0]
    delta["classification"] = "full_invalidation"
    delta["changed_paths"] = ["voiage/scientific_review_evidence.py"]

    validate_scientific_review_bundle(bundle)


def test_promotion_digest_decision_and_supersession_are_fail_closed() -> None:
    bundle = _bundle()
    bundle["evidence"]["promotion-receipt"]["scientific_approval_sha256"] = "f" * 64
    with pytest.raises(ScientificReviewEvidenceError, match="approval digest"):
        validate_scientific_review_bundle(bundle)

    bundle = _bundle()
    bundle["evidence"]["promotion-receipt"]["decision"] = "promote"
    bundle["evidence"]["scientific-approval"]["decision"] = (
        "conditional_remediation_and_rereview"
    )
    bundle["evidence"]["adjudication"]["decision"] = (
        "conditional_remediation_and_rereview"
    )
    with pytest.raises(
        ScientificReviewEvidenceError, match="scientifically acceptable"
    ):
        validate_scientific_review_bundle(bundle)

    bundle = _bundle()
    bundle["evidence"]["promotion-receipt"]["decision"] = "promote"
    bundle["evidence"]["scientific-approval"]["superseded_by"] = "APPROVAL-2"
    with pytest.raises(ScientificReviewEvidenceError, match="superseded"):
        validate_scientific_review_bundle(bundle)


def test_expired_or_superseded_approval_cannot_authorize_promotion() -> None:
    bundle = _bundle()
    bundle["evidence"]["scientific-approval"]["expires_at"] = "2026-08-01T00:00:00Z"
    bundle["evidence"]["promotion-receipt"]["decision"] = "promote"
    bundle["evidence"]["promotion-receipt"]["decision_at"] = "2026-08-02T00:00:00Z"

    with pytest.raises(ScientificReviewEvidenceError, match="expired"):
        validate_scientific_review_bundle(bundle)


def test_current_approval_can_authorize_promotion() -> None:
    bundle = _bundle()
    bundle["evidence"]["promotion-receipt"]["decision"] = "promote"

    validate_scientific_review_bundle(bundle)


@pytest.mark.parametrize("timestamp", [None, "not-a-timestamp"])
def test_timestamp_parser_fails_closed_for_non_timestamp_values(
    timestamp: object,
) -> None:
    with pytest.raises(ScientificReviewEvidenceError, match="RFC 3339"):
        review_evidence._parse_timestamp(timestamp, "approval.expires_at")
