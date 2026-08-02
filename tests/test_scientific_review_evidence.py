"""Scientific-review evidence must be structured, bound, and fail closed."""

from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import pytest

import voiage.scientific_review_evidence as review_evidence
from voiage.scientific_review_evidence import (
    ScientificReviewEvidenceError,
    bind_scientific_review_bundle,
    canonical_json_sha256,
    load_scientific_review_schemas,
    validate_scientific_review_bundle,
    validate_scientific_review_evidence,
)

ROOT = Path(__file__).parents[1]
FIXTURE = (
    ROOT
    / "specs/frontier/governance/scientific-review/v1/fixtures/valid-review-bundle.json"
)
GIT = shutil.which("git")


def _bundle() -> dict[str, object]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _rebind(bundle: dict[str, object]) -> dict[str, object]:
    return bind_scientific_review_bundle(bundle)


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


def test_contract_uses_typed_git_oids_and_canonical_content_digests() -> None:
    bundle = _bundle()
    packet = bundle["evidence"]["review-packet"]

    assert bundle["schema_version"] == "1.1.0"
    assert packet["candidate_commit"] == {"algorithm": "sha1", "value": "a" * 40}
    assert packet["candidate_tree"] == {"algorithm": "sha1", "value": "b" * 40}
    assert packet["packet_sha256"] == canonical_json_sha256(
        packet, excluded_json_pointers={"/packet_sha256"}
    )

    legacy = deepcopy(packet)
    legacy["candidate_commit"] = "a" * 64
    with pytest.raises(ScientificReviewEvidenceError):
        validate_scientific_review_evidence("review-packet", legacy)


def test_fabricated_declared_digest_is_rejected() -> None:
    bundle = _bundle()
    bundle["evidence"]["review-packet"]["packet_sha256"] = "f" * 64

    with pytest.raises(ScientificReviewEvidenceError, match="canonical digest"):
        validate_scientific_review_bundle(bundle)


def test_manifest_bytes_are_verified_against_the_frozen_git_tree(
    tmp_path: Path,
) -> None:
    assert GIT is not None
    repository = tmp_path / "repo"
    repository.mkdir()
    subprocess.run([GIT, "init", "-q", str(repository)], check=True)
    subprocess.run(
        [GIT, "-C", str(repository), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(
        [GIT, "-C", str(repository), "config", "user.name", "Contract Test"],
        check=True,
    )
    artifact = repository / "specs/example.json"
    artifact.parent.mkdir()
    artifact.write_text('{"fixture":true}\n', encoding="utf-8")
    subprocess.run([GIT, "-C", str(repository), "add", "."], check=True)
    subprocess.run(
        [GIT, "-C", str(repository), "commit", "-q", "-m", "fixture"],
        check=True,
    )
    commit = subprocess.check_output(
        [GIT, "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    tree = subprocess.check_output(
        [GIT, "-C", str(repository), "rev-parse", "HEAD^{tree}"], text=True
    ).strip()

    bundle = _bundle()
    bundle["fixture_status"] = "candidate_bound_contract_test"
    for value in bundle["evidence"].values():
        items = value if isinstance(value, list) else [value]
        for item in items:
            if "candidate_commit" in item:
                item["candidate_commit"] = {"algorithm": "sha1", "value": commit}
                item["candidate_tree"] = {"algorithm": "sha1", "value": tree}
    bundle["evidence"]["artifact-manifest"]["artifacts"][0]["sha256"] = hashlib.sha256(
        artifact.read_bytes()
    ).hexdigest()
    bundle = _rebind(bundle)
    validate_scientific_review_bundle(bundle, repository_root=repository)

    artifact.write_text('{"fixture":false}\n', encoding="utf-8")
    # The working tree is irrelevant; the frozen commit remains authoritative.
    validate_scientific_review_bundle(bundle, repository_root=repository)

    bundle["evidence"]["artifact-manifest"]["artifacts"][0]["sha256"] = "f" * 64
    bundle = _rebind(bundle)
    with pytest.raises(ScientificReviewEvidenceError, match="artifact bytes"):
        validate_scientific_review_bundle(bundle, repository_root=repository)


def test_report_attestation_and_decision_roles_cannot_collapse() -> None:
    bundle = _bundle()
    report = bundle["evidence"]["role-report"][0]
    report["reviewer"]["qualifications"] = ["different claim"]
    bundle = _rebind(bundle)
    with pytest.raises(ScientificReviewEvidenceError, match="attestation exactly"):
        validate_scientific_review_bundle(bundle)

    bundle = _bundle()
    bundle["evidence"]["scientific-approval"]["approver"] = deepcopy(
        bundle["evidence"]["adjudication"]["chair"]
    )
    bundle["evidence"]["scientific-approval"]["human_receipt"]["signer_identity"] = (
        "review-chair"
    )
    bundle = _rebind(bundle)
    with pytest.raises(ScientificReviewEvidenceError, match="separation of duties"):
        validate_scientific_review_bundle(bundle)


def test_medium_finding_requires_disposition_and_fresh_rereview() -> None:
    bundle = _bundle()
    finding = bundle["evidence"]["finding"][0]
    finding["severity"] = "medium"
    bundle["evidence"]["disposition"] = []
    bundle = _rebind(bundle)
    with pytest.raises(ScientificReviewEvidenceError, match="Medium finding"):
        validate_scientific_review_bundle(bundle)

    bundle = _bundle()
    bundle["evidence"]["finding"][0]["severity"] = "medium"
    bundle["evidence"]["disposition"][0]["rereview_report_ids"] = []
    bundle = _rebind(bundle)
    with pytest.raises(ScientificReviewEvidenceError, match="affected-role re-review"):
        validate_scientific_review_bundle(bundle)

    bundle = _bundle()
    bundle["evidence"]["finding"][0]["severity"] = "medium"
    bundle["evidence"]["disposition"][0]["rereview_report_ids"] = ["REPORT-ESTIMAND"]
    bundle = _rebind(bundle)
    with pytest.raises(ScientificReviewEvidenceError, match="affected-role re-review"):
        validate_scientific_review_bundle(bundle)


def test_reviewed_exclusion_must_bind_removed_capabilities() -> None:
    bundle = _bundle()
    disposition = bundle["evidence"]["disposition"][0]
    disposition["disposition"] = "reviewed_exclusion"
    disposition["excluded_capabilities"] = []
    bundle = _rebind(bundle)

    with pytest.raises(ScientificReviewEvidenceError, match="excluded capability"):
        validate_scientific_review_bundle(bundle)


def test_bounded_delta_is_field_allowlisted_and_signers_are_distinct() -> None:
    bundle = _bundle()
    delta = bundle["evidence"]["delta-classification"][0]
    delta["changed_paths"] = ["docs/synthetic-review.md"]
    delta["changed_artifact_hashes"][0]["path"] = "docs/synthetic-review.md"
    delta["changed_artifact_hashes"][0]["changed_fields"] = ["/title"]
    bundle = _rebind(bundle)
    with pytest.raises(ScientificReviewEvidenceError, match="administrative allowlist"):
        validate_scientific_review_bundle(bundle)

    bundle = _bundle()
    delta = bundle["evidence"]["delta-classification"][0]
    delta["signatures"][1]["reviewer"]["identity"] = delta["signatures"][0]["reviewer"][
        "identity"
    ]
    bundle = _rebind(bundle)
    with pytest.raises(ScientificReviewEvidenceError, match="distinct people"):
        validate_scientific_review_bundle(bundle)


def test_changed_path_and_hash_inventory_must_match_exactly() -> None:
    bundle = _bundle()
    bundle["evidence"]["delta-classification"][0]["changed_paths"].append(
        "conductor/tracks/other/metadata.json"
    )
    bundle = _rebind(bundle)

    with pytest.raises(ScientificReviewEvidenceError, match="changed-path inventory"):
        validate_scientific_review_bundle(bundle)


def test_human_receipts_expiry_and_supersession_fail_closed() -> None:
    bundle = _bundle()
    approval = bundle["evidence"]["scientific-approval"]
    assert approval["approver"]["actor_type"] == "human"
    assert approval["human_receipt"]["verification_method"] in {
        "signed_commit",
        "authenticated_github",
        "external_authoritative_receipt",
    }

    approval["human_receipt"]["signer_identity"] = "different-person"
    bundle = _rebind(bundle)
    with pytest.raises(ScientificReviewEvidenceError, match="signer"):
        validate_scientific_review_bundle(bundle)

    bundle = _bundle()
    approval = bundle["evidence"]["scientific-approval"]
    approval["expires_at"] = approval["decision_at"]
    bundle = _rebind(bundle)
    with pytest.raises(ScientificReviewEvidenceError, match="after decision_at"):
        validate_scientific_review_bundle(bundle)

    bundle = _bundle()
    bundle["evidence"]["scientific-approval"]["superseded_by"] = "APPROVAL-2"
    bundle = _rebind(bundle)
    with pytest.raises(ScientificReviewEvidenceError, match="superseded"):
        validate_scientific_review_bundle(
            bundle, at_time=datetime(2026, 8, 3, tzinfo=UTC)
        )


def test_unknown_finding_disposition_is_rejected() -> None:
    bundle = _bundle()
    extra = deepcopy(bundle["evidence"]["disposition"][0])
    extra["disposition_id"] = "D-UNKNOWN"
    extra["finding_id"] = "F-UNKNOWN"
    bundle["evidence"]["disposition"].append(extra)
    bundle = _rebind(bundle)

    with pytest.raises(ScientificReviewEvidenceError, match="unknown finding"):
        validate_scientific_review_bundle(bundle)


@pytest.mark.parametrize(
    "field", ["candidate_commit", "candidate_tree", "packet_sha256"]
)
def test_mixed_candidate_or_packet_binding_is_rejected(field: str) -> None:
    bundle = _bundle()
    bundle["evidence"]["scientific-approval"][field] = (
        {"algorithm": "sha1", "value": "f" * 40}
        if field != "packet_sha256"
        else "f" * 64
    )

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
    bundle["evidence"]["role-report"][0]["finding_ids"] = ["F-HIGH-OPEN"]
    bundle = _rebind(bundle)
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
    bundle = _rebind(bundle)
    with pytest.raises(ScientificReviewEvidenceError, match="scope"):
        validate_scientific_review_bundle(bundle)


def test_open_scientific_dissent_blocks_positive_approval() -> None:
    bundle = _bundle()
    bundle["evidence"]["disagreement"] = [
        {
            "evidence_kind": "disagreement",
            "schema_version": "1.1.0",
            "disagreement_id": "DISSENT-1",
            "candidate_commit": {"algorithm": "sha1", "value": "a" * 40},
            "candidate_tree": {"algorithm": "sha1", "value": "b" * 40},
            "packet_sha256": bundle["evidence"]["review-packet"]["packet_sha256"],
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
    bundle = _rebind(bundle)
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
    bundle = _rebind(bundle)
    with pytest.raises(ScientificReviewEvidenceError, match="decision differs"):
        validate_scientific_review_bundle(bundle)


def test_bounded_delta_requires_metadata_only_paths_and_two_signatures() -> None:
    bundle = _bundle()
    delta = bundle["evidence"]["delta-classification"][0]
    delta["changed_paths"] = ["voiage/methods/utility_information.py"]
    delta["changed_artifact_hashes"][0]["path"] = delta["changed_paths"][0]
    with pytest.raises(ScientificReviewEvidenceError, match="administrative allowlist"):
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
    bundle["evidence"]["delta-classification"][0]["changed_artifact_hashes"][0][
        "path"
    ] = changed_path

    with pytest.raises(ScientificReviewEvidenceError, match="administrative allowlist"):
        validate_scientific_review_bundle(bundle)


def test_full_invalidation_delta_does_not_use_metadata_only_exception() -> None:
    bundle = _bundle()
    delta = bundle["evidence"]["delta-classification"][0]
    delta["classification"] = "full_invalidation"
    delta["changed_paths"] = ["voiage/scientific_review_evidence.py"]
    delta["changed_artifact_hashes"][0]["path"] = delta["changed_paths"][0]

    validate_scientific_review_bundle(bundle)


def test_promotion_digest_decision_and_supersession_are_fail_closed() -> None:
    bundle = _bundle()
    bundle["evidence"]["promotion-receipt"]["scientific_approval_sha256"] = "f" * 64
    bundle["evidence"]["promotion-receipt"]["human_receipt"]["payload_sha256"] = (
        canonical_json_sha256(
            bundle["evidence"]["promotion-receipt"],
            excluded_json_pointers={
                "/receipt_sha256",
                "/human_receipt/payload_sha256",
            },
        )
    )
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
    bundle = _rebind(bundle)
    with pytest.raises(
        ScientificReviewEvidenceError, match="scientifically acceptable"
    ):
        validate_scientific_review_bundle(bundle)

    bundle = _bundle()
    bundle["evidence"]["promotion-receipt"]["decision"] = "promote"
    bundle["evidence"]["scientific-approval"]["superseded_by"] = "APPROVAL-2"
    bundle = _rebind(bundle)
    with pytest.raises(ScientificReviewEvidenceError, match="superseded"):
        validate_scientific_review_bundle(bundle)


def test_expired_or_superseded_approval_cannot_authorize_promotion() -> None:
    bundle = _bundle()
    bundle["evidence"]["scientific-approval"]["expires_at"] = "2026-08-01T00:00:00Z"
    bundle["evidence"]["promotion-receipt"]["decision"] = "promote"
    bundle["evidence"]["promotion-receipt"]["decision_at"] = "2026-08-02T00:00:00Z"
    bundle = _rebind(bundle)

    with pytest.raises(ScientificReviewEvidenceError, match="expired"):
        validate_scientific_review_bundle(bundle)


def test_current_approval_can_authorize_promotion() -> None:
    bundle = _bundle()
    bundle["evidence"]["promotion-receipt"]["decision"] = "promote"
    bundle = _rebind(bundle)

    validate_scientific_review_bundle(bundle)


@pytest.mark.parametrize("timestamp", [None, "not-a-timestamp"])
def test_timestamp_parser_fails_closed_for_non_timestamp_values(
    timestamp: object,
) -> None:
    with pytest.raises(ScientificReviewEvidenceError, match="RFC 3339"):
        review_evidence._parse_timestamp(timestamp, "approval.expires_at")
