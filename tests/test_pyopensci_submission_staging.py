"""Contracts for the unposted pyOpenSci submission staging packet."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import shutil

import pytest

from scripts.validate_pyopensci_submission_staging import validate_staging_packet

ROOT = Path(__file__).parents[1]
STAGING = ROOT / "specs" / "submission-readiness" / "pyopensci-submission-staging.json"
TEMPLATE = (
    ROOT / "specs" / "submission-readiness" / "pyopensci-submission-template.json"
)
CANDIDATE = (
    ROOT / "specs" / "submission-readiness" / "pyopensci-submission-candidate.json"
)
DRAFT = ROOT / "docs" / "release" / "pyopensci-submission-draft.md"
PUBLICATION_RECEIPT = (
    ROOT
    / "conductor"
    / "archive"
    / "quality_release_automation_20260723"
    / "release-2.1.0-publication-receipt-20260821.json"
)


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _staged_packet(tmp_path: Path, draft: str) -> Path:
    staged_root = tmp_path / "repo"
    relative_files = (
        TEMPLATE.relative_to(ROOT),
        CANDIDATE.relative_to(ROOT),
        PUBLICATION_RECEIPT.relative_to(ROOT),
    )
    for relative in relative_files:
        destination = staged_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)

    staged_draft = staged_root / DRAFT.relative_to(ROOT)
    staged_draft.parent.mkdir(parents=True, exist_ok=True)
    staged_draft.write_text(draft, encoding="utf-8")

    staging = _load(STAGING)
    staging["draft"]["sha256"] = hashlib.sha256(staged_draft.read_bytes()).hexdigest()
    staged_manifest = staged_root / STAGING.relative_to(ROOT)
    staged_manifest.parent.mkdir(parents=True, exist_ok=True)
    staged_manifest.write_text(json.dumps(staging), encoding="utf-8")
    return staged_root


def test_staging_packet_validates() -> None:
    """The canonical repository packet passes its fail-closed validator."""
    assert validate_staging_packet(ROOT) == []


def test_template_provenance_is_exact() -> None:
    """The draft remains bound to the inspected upstream template bytes."""
    template = _load(TEMPLATE)
    upstream = template["upstream"]

    assert template["state"] == "reference_only_unposted"
    assert template["submission_performed"] is False
    assert upstream["repository"] == "pyOpenSci/software-submission"
    assert upstream["commit"] == "a1f31b8aab21128faee96ee548d256d5cffc3ba9"
    assert upstream["content_sha256"] == (
        "43b69c9633967e16bcb68435ba0306911f266de554eaf63345c852407d63aea4"
    )


def test_candidate_matches_publication_receipt() -> None:
    """The recommended package is the current evidence-bound public release."""
    candidate = _load(CANDIDATE)
    receipt = _load(PUBLICATION_RECEIPT)
    recommended = candidate["recommended_candidate"]

    assert candidate["state"] == "selected_for_local_staging_maintainer_confirmed"
    assert candidate["maintainer_version_confirmation"] == "confirmed"
    assert candidate["submission_performed"] is False
    assert recommended["version"] == receipt["release"]["version"] == "2.1.0"
    assert recommended["commit"] == receipt["release"]["commit"]
    assert candidate["artifact_sha256"] == receipt["reviewed_digests"]
    assert candidate["joss_handoff"]["state"] == (
        "blocked_pending_refresh_and_external_evidence"
    )


def test_staging_manifest_records_version_without_external_action() -> None:
    """Version selection cannot imply other attestations, posting, or acceptance."""
    staging = _load(STAGING)

    assert staging["state"] == "prepared_local_unposted"
    assert staging["candidate_version"] == "2.1.0"
    assert staging["candidate_confirmation"] == "confirmed_maintainer"
    assert staging["human_attestations"]["submitted_version"] == "confirmed"
    assert all(
        state == "pending"
        for key, state in staging["human_attestations"].items()
        if key != "submitted_version"
    )
    assert all(performed is False for performed in staging["external_actions"].values())
    assert staging["external_outcomes"] == {
        "pyopensci_review": "not_started",
        "pyopensci_acceptance": "pending_external",
        "joss_referral": "not_started",
        "joss_acceptance": "pending_external",
    }


def test_draft_is_unposted_and_contains_current_template_sections() -> None:
    """The local Markdown draft is complete but visibly non-submissive."""
    draft = DRAFT.read_text(encoding="utf-8")
    template = _load(TEMPLATE)

    assert "UNPOSTED LOCAL DRAFT" in draft
    assert "Submission performed: **No**" in draft
    for section in template["required_sections"]:
        assert f"## {section}" in draft
    assert (
        "Version submitted: 2.1.0 (confirmed by maintainer; submission not performed)"
        in draft
    )
    assert "- [ ] I agree to abide by" in draft
    assert "- [ ] I have read and will commit" in draft
    assert "- [ ] Do you wish to automatically submit" in draft
    assert "- [ ] Last but not least please fill out our pre-review survey" in draft
    assert not re.search(
        r"\b(?:TBD|TODO|FILL(?:\s+THIS)?\s+IN)\b", draft, re.IGNORECASE
    )


def test_validator_rejects_false_submission_state(tmp_path: Path) -> None:
    """A local packet cannot be relabelled as submitted without evidence."""
    staging = _load(STAGING)
    staging["external_actions"]["pyopensci_issue_created"] = True
    staged_root = tmp_path / "repo"
    staged_manifest = (
        staged_root
        / "specs"
        / "submission-readiness"
        / "pyopensci-submission-staging.json"
    )
    staged_manifest.parent.mkdir(parents=True)
    staged_manifest.write_text(json.dumps(staging), encoding="utf-8")

    findings = validate_staging_packet(staged_root, require_all_files=False)

    assert "external actions must remain false in an unposted packet" in findings


def test_validator_rejects_missing_human_attestation(tmp_path: Path) -> None:
    """Deleting an unchecked human gate cannot make the packet valid."""
    staging = _load(STAGING)
    del staging["human_attestations"]["pre_review_survey"]
    staged_root = tmp_path / "repo"
    staged_manifest = (
        staged_root
        / "specs"
        / "submission-readiness"
        / "pyopensci-submission-staging.json"
    )
    staged_manifest.parent.mkdir(parents=True)
    staged_manifest.write_text(json.dumps(staging), encoding="utf-8")

    findings = validate_staging_packet(staged_root, require_all_files=False)

    assert "human attestation key set is incomplete or unexpected" in findings


def test_validator_rejects_missing_external_action(tmp_path: Path) -> None:
    """Deleting a false external-action receipt cannot weaken the boundary."""
    staging = _load(STAGING)
    del staging["external_actions"]["pyopensci_contact_made"]
    staged_root = tmp_path / "repo"
    staged_manifest = (
        staged_root
        / "specs"
        / "submission-readiness"
        / "pyopensci-submission-staging.json"
    )
    staged_manifest.parent.mkdir(parents=True)
    staged_manifest.write_text(json.dumps(staging), encoding="utf-8")

    findings = validate_staging_packet(staged_root, require_all_files=False)

    assert "external action key set is incomplete or unexpected" in findings


@pytest.mark.parametrize(
    "marker",
    [
        "I agree to abide by",
        "I have read and will commit",
        "Do you wish to automatically submit",
        "Maintainer confirmation pending. If confirmed",
        "I have read the pyOpenSci author guide",
        "Last but not least please fill out our pre-review survey",
    ],
)
def test_validator_rejects_checked_human_attestation(
    tmp_path: Path, marker: str
) -> None:
    """No pending human checkbox can be checked by editing and rebinding."""
    draft = DRAFT.read_text(encoding="utf-8")
    draft = draft.replace(f"- [ ] {marker}", f"- [x] {marker}", 1)
    staged_root = _staged_packet(tmp_path, draft)

    findings = validate_staging_packet(staged_root)

    assert "draft human-attestation markers must remain uniquely unchecked" in findings


def test_validator_rejects_checked_human_attestation_duplicate(
    tmp_path: Path,
) -> None:
    """A checked duplicate cannot hide behind the required unchecked marker."""
    draft = DRAFT.read_text(encoding="utf-8")
    draft += "\n- [x] I have read the pyOpenSci author guide.\n"
    staged_root = _staged_packet(tmp_path, draft)

    findings = validate_staging_packet(staged_root)

    assert "draft human-attestation markers must remain uniquely unchecked" in findings


def test_validator_rejects_unqualified_submitted_version(tmp_path: Path) -> None:
    """The draft must preserve the confirmation and non-submission boundary."""
    draft = DRAFT.read_text(encoding="utf-8").replace(
        "Version submitted: 2.1.0 (confirmed by maintainer; submission not performed)",
        "Version submitted: 2.1.0",
        1,
    )
    staged_root = _staged_packet(tmp_path, draft)

    findings = validate_staging_packet(staged_root)

    assert (
        "draft submitted version must match the confirmed maintainer selection"
        in findings
    )
