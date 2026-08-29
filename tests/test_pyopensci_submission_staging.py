"""Contracts for the unposted pyOpenSci submission staging packet."""

from __future__ import annotations

import json
from pathlib import Path
import re

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
    / "tracks"
    / "quality_release_automation_20260723"
    / "release-2.1.0-publication-receipt-20260821.json"
)


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


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

    assert candidate["state"] == (
        "recommended_for_local_staging_maintainer_confirmation_pending"
    )
    assert candidate["maintainer_version_confirmation"] == "pending"
    assert candidate["submission_performed"] is False
    assert recommended["version"] == receipt["release"]["version"] == "2.1.0"
    assert recommended["commit"] == receipt["release"]["commit"]
    assert candidate["artifact_sha256"] == receipt["reviewed_digests"]
    assert candidate["joss_handoff"]["state"] == (
        "blocked_pending_refresh_and_external_evidence"
    )


def test_staging_manifest_keeps_human_and_external_states_pending() -> None:
    """A prepared draft cannot imply attestations, posting, or acceptance."""
    staging = _load(STAGING)

    assert staging["state"] == "prepared_local_unposted"
    assert staging["candidate_version"] == "2.1.0"
    assert staging["candidate_confirmation"] == "pending_maintainer"
    assert all(state == "pending" for state in staging["human_attestations"].values())
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
        "Version submitted: 2.1.0 (recommended; maintainer confirmation pending)"
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
