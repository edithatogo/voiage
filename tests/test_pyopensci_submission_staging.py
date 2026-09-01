"""Contracts for the unposted pyOpenSci submission staging packet."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import shutil

import pytest

from scripts.validate_pyopensci_submission_staging import (
    DRAFT_ATTESTATION_MARKERS,
    EXPECTED_HUMAN_ATTESTATIONS,
    validate_staging_packet,
)

ROOT = Path(__file__).parents[1]
STAGING = ROOT / "specs" / "submission-readiness" / "pyopensci-submission-staging.json"
TEMPLATE = (
    ROOT / "specs" / "submission-readiness" / "pyopensci-submission-template.json"
)
CANDIDATE = (
    ROOT / "specs" / "submission-readiness" / "pyopensci-submission-candidate.json"
)
DRAFT = ROOT / "docs" / "release" / "pyopensci-submission-draft.md"
RECEIPT = (
    ROOT
    / "conductor/tracks/v2_2_release_and_venue_submissions_20260830"
    / "release-2.2.0-publication-receipt-20260830.json"
)


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _staged_packet(tmp_path: Path, draft: str) -> Path:
    staged_root = tmp_path / "repo"
    relative_files = [
        TEMPLATE.relative_to(ROOT),
        CANDIDATE.relative_to(ROOT),
        RECEIPT.relative_to(ROOT),
    ]
    confirmation = _load(STAGING).get("maintainer_confirmation")
    if confirmation:
        relative_files.append(Path(confirmation["path"]))
    withdrawal = _load(STAGING).get("withdrawal_receipt")
    if withdrawal:
        relative_files.append(Path(withdrawal["path"]))
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
    assert upstream["commit"] == "df24b7c63a589ff5d82a30e42f1d11b8aa1b5927"
    baseline_path = next(
        path
        for state in ("tracks", "archive")
        if (
            path := ROOT
            / "conductor"
            / state
            / "v2_2_release_and_venue_submissions_20260830"
            / "release-submission-live-baseline-20260830.json"
        ).is_file()
    )
    baseline = _load(baseline_path)["live_official_inputs"][
        "pyopensci_submission_template"
    ]
    assert upstream["commit"] == baseline["latest_path_commit"]
    assert upstream["blob_sha"] == baseline["blob_sha"]
    assert upstream["content_sha256"] == (
        "43b69c9633967e16bcb68435ba0306911f266de554eaf63345c852407d63aea4"
    )


def test_candidate_is_bound_to_verified_publication() -> None:
    """The selected package carries exact public evidence, not venue approval."""
    candidate = _load(CANDIDATE)
    recommended = candidate["recommended_candidate"]

    assert candidate["state"] == ("published_release_maintainer_confirmed")
    assert candidate["maintainer_version_confirmation"] == "confirmed"
    assert candidate["submission_performed"] is False
    assert recommended["version"] == "2.2.0"
    receipt = _load(RECEIPT)
    assert recommended["commit"] == receipt["release"]["commit"]
    assert recommended["published_at"] == receipt["github"]["published_at"]
    assert (
        recommended["publication_receipt"]["sha256"]
        == hashlib.sha256(RECEIPT.read_bytes()).hexdigest()
    )
    assert recommended["tag_signature_verified"] is True
    assert recommended["immutable_github_release"] is True
    assert candidate["artifact_sha256"] == receipt["reviewed_digests"]
    assert candidate["joss_handoff"]["state"] == (
        "blocked_pending_refresh_and_external_evidence"
    )
    assert candidate["joss_handoff"]["permanent_arxiv_identifier"] == (
        "deferred_until_after_journal_submission_not_pre_joss_gate"
    )


def test_current_venue_projections_follow_withdrawal_and_journal_first_order() -> None:
    """Current guidance supersedes, without rewriting, the dated gate snapshot."""
    targets = _load(ROOT / "specs" / "submission-readiness" / "targets.json")
    by_id = {target["id"]: target for target in targets["targets"]}
    pyopensci_decision = by_id["pyopensci"]["next_decision"]
    arxiv_decision = by_id["arxiv"]["next_decision"]

    assert "requests #271 and #272 are withdrawn" in pyopensci_decision
    assert "private pre-review survey" in pyopensci_decision
    assert "human-written submission body" in pyopensci_decision
    assert "write later review communication personally is confirmed" in (
        pyopensci_decision
    )
    assert "authenticated pyOpenSci submission" in pyopensci_decision
    assert "resolve contact-capacity eligibility" not in pyopensci_decision
    assert "deferred until an actual journal submission" in arxiv_decision

    readiness = (ROOT / "docs" / "release" / "pyopensci-readiness.md").read_text(
        encoding="utf-8"
    )
    assert "withdrawn and verified closed on 31 August 2026" in readiness
    assert "still require contact-capacity clarification" not in readiness

    historical_checklist = (
        ROOT
        / "conductor/tracks/v2_2_release_and_venue_submissions_20260830"
        / "venue-human-action-checklist-20260831.md"
    )
    assert hashlib.sha256(historical_checklist.read_bytes()).hexdigest() == (
        "47db7c5550defa0ee226481105677833618046078647289f24c3d9bb7489f425"
    )
    assert "Open; labels `presubmission`" in historical_checklist.read_text(
        encoding="utf-8"
    )

    supersession = (
        ROOT
        / "conductor/tracks/v2_2_release_and_venue_submissions_20260830"
        / "venue-human-action-checklist-supersession-20260902.md"
    ).read_text(encoding="utf-8")
    assert "immutable" in supersession
    assert "factual snapshot" in supersession.replace("\n", " ")
    assert "not current action guidance" in supersession.replace("\n", " ")
    assert "must not trigger repeated declarations" in supersession

    active_spec = (
        ROOT / "conductor/tracks/v2_2_release_and_venue_submissions_20260830/spec.md"
    ).read_text(encoding="utf-8")
    assert "requests #271 and #272 were closed as not planned" in active_spec
    assert "contact-capacity clarification for open issues #271 and #272" not in (
        active_spec
    )


def test_validator_rejects_rebound_arxiv_before_joss_gate(tmp_path: Path) -> None:
    """Rebinding the candidate hash cannot restore the superseded arXiv gate."""
    staged_root = _staged_packet(tmp_path, DRAFT.read_text(encoding="utf-8"))
    candidate_path = staged_root / CANDIDATE.relative_to(ROOT)
    candidate = _load(candidate_path)
    candidate["joss_handoff"]["permanent_arxiv_identifier"] = "pending"
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
    staging_path = staged_root / STAGING.relative_to(ROOT)
    staging = _load(staging_path)
    staging["candidate"]["sha256"] = hashlib.sha256(
        candidate_path.read_bytes()
    ).hexdigest()
    staging_path.write_text(json.dumps(staging), encoding="utf-8")

    findings = validate_staging_packet(staged_root)

    assert "JOSS handoff must preserve the journal-first arXiv deferral" in findings


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_binding",
        "missing_file",
        "path",
        "digest",
        "state",
        "reason",
        "approval",
    ],
)
def test_validator_rejects_invalid_withdrawal_evidence(
    tmp_path: Path, mutation: str
) -> None:
    """Withdrawal claims require immutable, semantically valid issue evidence."""
    staged_root = _staged_packet(tmp_path, DRAFT.read_text(encoding="utf-8"))
    staging_path = staged_root / STAGING.relative_to(ROOT)
    staging = _load(staging_path)
    binding = staging["withdrawal_receipt"]
    receipt_path = staged_root / binding["path"]

    if mutation == "missing_binding":
        del staging["withdrawal_receipt"]
    elif mutation == "missing_file":
        receipt_path.unlink()
    elif mutation == "path":
        copied = receipt_path.with_name("copied-withdrawal-receipt.json")
        shutil.copyfile(receipt_path, copied)
        binding["path"] = str(copied.relative_to(staged_root))
    elif mutation == "digest":
        binding["sha256"] = "0" * 64
    else:
        receipt = _load(receipt_path)
        if mutation == "state":
            receipt["issues"][0]["after_state"] = "open"
        elif mutation == "reason":
            receipt["issues"][1]["after_state_reason"] = "completed"
        else:
            receipt["editorial_approval_inferred"] = True
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        binding["sha256"] = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    staging_path.write_text(json.dumps(staging), encoding="utf-8")

    findings = validate_staging_packet(staged_root)

    assert any(finding.startswith("withdrawal receipt") for finding in findings)


def test_staging_manifest_records_version_without_external_action() -> None:
    """Version selection cannot imply other attestations, posting, or acceptance."""
    staging = _load(STAGING)

    assert staging["state"] == "prepared_local_unposted"
    assert staging["candidate_version"] == "2.2.0"
    assert staging["candidate_confirmation"] == "confirmed_maintainer"
    assert staging["human_attestations"]["submitted_version"] == "confirmed"
    assert staging["human_attestations"]["joss_partnership_option"] == "confirmed"
    assert staging["human_attestations"]["pre_review_survey"] == "pending"
    if "maintainer_confirmation" in staging:
        assert staging["action_gates"] == {
            "pre_review_survey": "pending",
            "human_written_submission_text": "pending",
        }
    assert all(performed is False for performed in staging["external_actions"].values())
    assert staging["external_outcomes"] == {
        "pyopensci_review": "not_started",
        "pyopensci_acceptance": "pending_external",
        "joss_referral": "not_started",
        "joss_acceptance": "pending_external",
    }


@pytest.mark.parametrize("mutation", ["published", "digests", "missing_receipt"])
def test_validator_rejects_rebound_prepublication_claims(
    tmp_path: Path, mutation: str
) -> None:
    """Rebinding hashes cannot legitimize publication claims or deleted gates."""
    staged_root = _staged_packet(tmp_path, DRAFT.read_text(encoding="utf-8"))
    candidate = _load(CANDIDATE)
    candidate["state"] = "release_candidate_prepublication_maintainer_confirmed"
    recommended = candidate["recommended_candidate"]
    for key in ("commit", "tree", "tag_object", "published_at", "publication_receipt"):
        recommended[key] = None
    for key in (
        "tag_signature_verified",
        "immutable_github_release",
        "latest_on_pypi_when_observed",
    ):
        recommended[key] = False
    candidate["artifact_sha256"] = {}
    if mutation == "published":
        candidate["recommended_candidate"]["published_at"] = "2026-08-30T00:00:00Z"
    elif mutation == "digests":
        candidate["artifact_sha256"] = {"unverified.whl": "0" * 64}
    else:
        del candidate["recommended_candidate"]["publication_receipt"]
    candidate_path = staged_root / CANDIDATE.relative_to(ROOT)
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
    staging_path = staged_root / STAGING.relative_to(ROOT)
    staging = _load(staging_path)
    staging["candidate"]["sha256"] = hashlib.sha256(
        candidate_path.read_bytes()
    ).hexdigest()
    staging_path.write_text(json.dumps(staging), encoding="utf-8")

    findings = validate_staging_packet(staged_root)

    assert any(
        finding.startswith("prepublication candidate must not") for finding in findings
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "commit",
        "digests",
        "missing_receipt",
        "non_json",
        "escape",
        "draft",
        "failed_run",
        "yanked",
    ],
)
def test_validator_rejects_inconsistent_published_evidence(
    tmp_path: Path, mutation: str
) -> None:
    """Rebound candidate hashes cannot bypass independent receipt checks."""
    staged_root = _staged_packet(tmp_path, DRAFT.read_text(encoding="utf-8"))
    candidate = _load(CANDIDATE)
    recommended = candidate["recommended_candidate"]
    receipt_path = staged_root / RECEIPT.relative_to(ROOT)
    receipt = _load(receipt_path)
    if mutation == "commit":
        recommended["commit"] = "0" * 40
    elif mutation == "digests":
        candidate["artifact_sha256"] = {"fake.whl": "0" * 64}
    elif mutation == "missing_receipt":
        receipt_path.unlink()
    elif mutation == "non_json":
        renamed = receipt_path.with_suffix(".txt")
        receipt_path.rename(renamed)
        recommended["publication_receipt"]["path"] = str(
            renamed.relative_to(staged_root)
        )
    elif mutation == "escape":
        recommended["publication_receipt"] = {
            "path": "../escape.json",
            "sha256": "0" * 64,
        }
    else:
        if mutation == "draft":
            receipt["github"]["draft"] = True
        elif mutation == "failed_run":
            receipt["workflows"]["publication"]["conclusion"] = "failure"
        else:
            receipt["pypi"]["artifacts"][0]["yanked"] = True
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        recommended["publication_receipt"]["sha256"] = hashlib.sha256(
            receipt_path.read_bytes()
        ).hexdigest()
    candidate_path = staged_root / CANDIDATE.relative_to(ROOT)
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
    staging_path = staged_root / STAGING.relative_to(ROOT)
    staging = _load(staging_path)
    staging["candidate"]["sha256"] = hashlib.sha256(
        candidate_path.read_bytes()
    ).hexdigest()
    staging_path.write_text(json.dumps(staging), encoding="utf-8")

    assert validate_staging_packet(staged_root)


def test_draft_is_unposted_and_contains_current_template_sections() -> None:
    """The local Markdown draft is complete but visibly non-submissive."""
    draft = DRAFT.read_text(encoding="utf-8")
    template = _load(TEMPLATE)

    assert "UNPOSTED LOCAL DRAFT" in draft
    assert "Submission performed: **No**" in draft
    for section in template["required_sections"]:
        assert f"## {section}" in draft
    assert (
        "Version submitted: 2.2.0 (confirmed by maintainer; submission not performed)"
        in draft
    )
    mark = "x" if "maintainer_confirmation" in _load(STAGING) else " "
    assert f"- [{mark}] I agree to abide by" in draft
    assert f"- [{mark}] I have read and will commit" in draft
    assert "- [x] Do you wish to automatically submit" in draft
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
        "I confirm sustained human-led development",
        "I have personally reviewed and understood",
        "I will write the review communication personally",
        "I have verified the AI scope and scale disclosure",
        "Reviewers may open",
        "I have read the pyOpenSci author guide",
        "Last but not least please fill out our pre-review survey",
    ],
)
def test_validator_rejects_checked_human_attestation(
    tmp_path: Path, marker: str
) -> None:
    """No pending human checkbox can be checked by editing and rebinding."""
    draft = DRAFT.read_text(encoding="utf-8")
    if (
        "maintainer_confirmation" not in _load(STAGING)
        and marker == "Reviewers may open"
    ):
        marker = "Maintainer confirmation pending. If confirmed"
    if f"- [ ] {marker}" in draft:
        draft = draft.replace(f"- [ ] {marker}", f"- [x] {marker}", 1)
    else:
        draft = draft.replace(f"- [x] {marker}", f"- [ ] {marker}", 1)
    staged_root = _staged_packet(tmp_path, draft)

    findings = validate_staging_packet(staged_root)

    assert "draft human-attestation markers must match scoped confirmations" in findings


def test_validator_rejects_checked_human_attestation_duplicate(
    tmp_path: Path,
) -> None:
    """A checked duplicate cannot hide behind the required unchecked marker."""
    draft = DRAFT.read_text(encoding="utf-8")
    draft += "\n- [x] I have read the pyOpenSci author guide.\n"
    staged_root = _staged_packet(tmp_path, draft)

    findings = validate_staging_packet(staged_root)

    assert "draft human-attestation markers must match scoped confirmations" in findings


@pytest.mark.parametrize("mutation", ["unchecked", "duplicate"])
def test_validator_preserves_confirmed_joss_route(
    tmp_path: Path, mutation: str
) -> None:
    draft = DRAFT.read_text(encoding="utf-8")
    marker = "Do you wish to automatically submit"
    if mutation == "unchecked":
        draft = draft.replace(f"- [x] {marker}", f"- [ ] {marker}", 1)
    else:
        draft += f"\n- [x] {marker} to JOSS?\n"
    findings = validate_staging_packet(_staged_packet(tmp_path, draft))
    assert "draft JOSS option must match the confirmed maintainer selection" in findings


def test_validator_rejects_unqualified_submitted_version(tmp_path: Path) -> None:
    """The draft must preserve the confirmation and non-submission boundary."""
    draft = DRAFT.read_text(encoding="utf-8").replace(
        "Version submitted: 2.2.0 (confirmed by maintainer; submission not performed)",
        "Version submitted: 2.2.0",
        1,
    )
    staged_root = _staged_packet(tmp_path, draft)

    findings = validate_staging_packet(staged_root)

    assert (
        "draft submitted version must match the confirmed maintainer selection"
        in findings
    )


def test_legacy_pending_packet_still_validates(tmp_path: Path) -> None:
    """Historical pending packets remain valid without fabricated confirmations."""
    draft = DRAFT.read_text(encoding="utf-8")
    for marker in DRAFT_ATTESTATION_MARKERS:
        draft = draft.replace(f"- [x] {marker}", f"- [ ] {marker}")
    draft = draft.replace(
        "Reviewers may open", "Maintainer confirmation pending. If confirmed"
    )
    root = _staged_packet(tmp_path, draft)
    manifest = root / STAGING.relative_to(ROOT)
    staging = _load(manifest)
    staging.pop("maintainer_confirmation", None)
    staging.pop("action_gates", None)
    staging["human_attestations"] = EXPECTED_HUMAN_ATTESTATIONS
    manifest.write_text(json.dumps(staging), encoding="utf-8")
    assert validate_staging_packet(root) == []


@pytest.mark.parametrize(
    "mutation",
    ["missing", "digest", "version", "scope", "survey", "authorship", "action_gate"],
)
def test_confirmation_requires_scoped_immutable_evidence(
    tmp_path: Path, mutation: str
) -> None:
    """A commitment cannot invent completed actions or expand a user's scope."""
    root = _staged_packet(tmp_path, DRAFT.read_text(encoding="utf-8"))
    manifest = root / STAGING.relative_to(ROOT)
    staging = _load(manifest)
    if "maintainer_confirmation" not in staging:
        confirmed = sorted(
            key
            for key, state in EXPECTED_HUMAN_ATTESTATIONS.items()
            if state == "pending" and key != "pre_review_survey"
        )
        receipt_path = root / "test-maintainer-decision.json"
        receipt_path.write_text(
            json.dumps(
                {
                    "schema_version": "voiage.maintainer-venue-decision.v1",
                    "candidate_version": "2.2.0",
                    "confirmed_attestations": confirmed,
                    "source": "current_user_message",
                    "user_statement": "Hypothetical test confirmation only",
                    "boundaries": {
                        "pre_review_survey_completed": False,
                        "existing_ai_draft_human_authored": False,
                        "submission_performed": False,
                        "editorial_capacity_approved": False,
                        "arxiv_submission_performed": False,
                    },
                }
            ),
            encoding="utf-8",
        )
        staging["maintainer_confirmation"] = {
            "path": receipt_path.name,
            "sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
        }
        staging["human_attestations"].update(dict.fromkeys(confirmed, "confirmed"))
        staging["action_gates"] = {
            "pre_review_survey": "pending",
            "human_written_submission_text": "pending",
        }
        draft_path = root / DRAFT.relative_to(ROOT)
        draft = draft_path.read_text(encoding="utf-8").replace(
            "Maintainer confirmation pending. If confirmed", "Reviewers may open"
        )
        for marker, key in DRAFT_ATTESTATION_MARKERS.items():
            if key in confirmed:
                draft = draft.replace(f"- [ ] {marker}", f"- [x] {marker}")
        draft_path.write_text(draft, encoding="utf-8")
        staging["draft"]["sha256"] = hashlib.sha256(draft_path.read_bytes()).hexdigest()
        manifest.write_text(json.dumps(staging), encoding="utf-8")
    assert validate_staging_packet(root) == []
    binding = staging["maintainer_confirmation"]
    receipt_path = root / binding["path"]
    receipt = _load(receipt_path)
    if mutation == "missing":
        receipt_path.unlink()
    elif mutation == "digest":
        binding["sha256"] = "0" * 64
    elif mutation == "action_gate":
        staging["action_gates"]["human_written_submission_text"] = "complete"
    else:
        if mutation == "version":
            receipt["candidate_version"] = "2.1.0"
        elif mutation == "scope":
            receipt["confirmed_attestations"].append("pre_review_survey")
        elif mutation == "survey":
            receipt["boundaries"]["pre_review_survey_completed"] = True
        elif mutation == "authorship":
            receipt["boundaries"]["existing_ai_draft_human_authored"] = True
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        binding["sha256"] = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(staging), encoding="utf-8")
    assert validate_staging_packet(root)
