"""Contracts for the research-software registry handoff."""

import json
from pathlib import Path

from scripts.validate_external_track_handoff import validate_handoff

TRACK = Path("conductor/archive/research_software_registry_readiness_20260721")
HANDOFF = TRACK / "handoff/registry-readiness.json"


def test_registry_handoff_preserves_release_and_external_gates() -> None:
    summary = validate_handoff(HANDOFF)

    assert summary == {
        "track_id": "research_software_registry_readiness_20260721",
        "channel": "research-software-registries",
        "status": "blocked",
        "command_count": 12,
        "evidence_count": 8,
    }


def test_registry_track_records_native_paper_issue_hierarchy() -> None:
    """The arXiv lane remains traceable from Conductor and the GitHub parent."""
    track = HANDOFF.parent.parent
    metadata = json.loads((track / "metadata.json").read_text())
    handoff = json.loads(HANDOFF.read_text())
    plan = (track / "plan.md").read_text()
    specification = (track / "spec.md").read_text()

    arxiv_issue = "https://github.com/edithatogo/voiage/issues/312"
    independent_validation_issue = "https://github.com/edithatogo/voiage/issues/471"
    submission_contract_issues = {
        f"https://github.com/edithatogo/voiage/issues/{number}"
        for number in range(614, 618)
    }
    hpc_packaging_issue = "https://github.com/edithatogo/voiage/issues/622"
    assert arxiv_issue in metadata["github_subissues"]
    assert independent_validation_issue in metadata["github_subissues"]
    assert submission_contract_issues <= set(metadata["github_subissues"])
    assert hpc_packaging_issue in metadata["github_subissues"]
    assert arxiv_issue in plan
    assert independent_validation_issue in plan
    assert arxiv_issue in specification
    assert independent_validation_issue in specification
    assert all(issue in specification for issue in submission_contract_issues)
    assert hpc_packaging_issue in plan
    assert hpc_packaging_issue in specification
    assert "- [x] Reproduce and remediate the panel's scientific EVSI finding" in plan
    index = (track / "index.md").read_text()
    assert "merged v2 scientific EVSI contract" in index
    assert "do not prevent repository archival" in index
    assert "JOSS/arXiv submission, curation, indexing, and registry acceptance" in index
    assert handoff["arxiv_preprint_evidence"]["review_pr"].endswith("/pull/311")
    assert handoff["arxiv_preprint_evidence"]["prior_submission_id"] == "7861466"
    assert handoff["arxiv_preprint_evidence"]["submission_id"] == "7870358"
    assert handoff["arxiv_preprint_evidence"]["status"] == "replacement_incomplete"
    assert handoff["arxiv_preprint_evidence"]["submission_performed"] is False
    assert "Replacement submission `7870358`" in plan
    assert "it is not submission evidence" in plan
    assert handoff["joss_submission_evidence"]["selected_route"] == "direct_joss"
    assert (
        handoff["joss_submission_evidence"]["status"]
        == "repository_ready_pending_human_and_external_evidence"
    )
    remaining = handoff["joss_submission_evidence"]["remaining_submission_gates"]
    assert not any("exact v2 release" in gate for gate in remaining)
    assert not any("AI-policy attestation" in gate for gate in remaining)
    assert not any("research-workflow use" in gate for gate in remaining)
    assert any("human community engagement" in gate for gate in remaining)
    assert not any("Open Journals PDF" in gate for gate in remaining)
    assert (
        handoff["joss_submission_evidence"]["release_bound_pdf"]["visual_review"]
        == "passed"
    )
    assert handoff["release_evidence"]["tag"] == "v2.0.0"
    assert (
        handoff["software_heritage_archival_request"]["snapshot_swhid"]
        == "swh:1:snp:31f89375852737bb9eb62ebc03fadfbc7ff70c2d"
    )


def test_registry_track_separates_repository_completion_from_external_gates() -> None:
    """Archived readiness work must not claim external registry outcomes."""
    plan = (TRACK / "plan.md").read_text(encoding="utf-8")
    metadata = json.loads((TRACK / "metadata.json").read_text(encoding="utf-8"))
    handoff = json.loads(HANDOFF.read_text(encoding="utf-8"))
    assurance = json.loads(
        Path("paper/joss-editorial-assurance.json").read_text(encoding="utf-8")
    )
    readiness = Path("docs/release/joss-submission-readiness.md").read_text(
        encoding="utf-8"
    )

    assert metadata["status"] == "completed"
    assert "- [~]" not in plan
    assert "- [ ]" not in plan
    human_gate = "all_retained_ai_outputs_reviewed_modified_and_validated"
    assert assurance["author_attestations"][human_gate]["confirmed_on"] == "2026-07-27"
    assert (
        assurance["human_review"][human_gate] == "pending_explicit_final_confirmation"
    )
    remaining = handoff["joss_submission_evidence"]["remaining_submission_gates"]
    assert not any("AI-policy attestation" in gate for gate in remaining)
    assert not any("research-workflow use" in gate for gate in remaining)
    assert any("human community engagement" in gate for gate in remaining)
    assert "maintainer confirmed review and understanding on 31 August" in readiness
    assert "subsequent changes require review against the submitted packet" in readiness
    assert "Repository checks do not establish venue acceptance" in readiness
    assert "historical same-author VOP record" in readiness
    external_gate = next(
        gate
        for gate in metadata["gates"]
        if gate["id"] == "external-registry-decisions"
    )
    assert external_gate["status"] == "pending"


def test_registry_archive_records_its_merged_delivery_pr() -> None:
    """The archive must cite the PR that delivered and archived the track."""
    metadata = json.loads((TRACK / "metadata.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        Path("conductor/github-cross-references.json").read_text(encoding="utf-8")
    )
    entry = next(
        item
        for item in manifest["tracks"]
        if item["track_id"] == "research_software_registry_readiness_20260721"
    )
    index = (TRACK / "index.md").read_text(encoding="utf-8")

    delivery_url = "https://github.com/edithatogo/voiage/pull/880"
    assert delivery_url in metadata["github_cross_reference"]["pull_requests"]
    delivery = next(item for item in entry["pull_requests"] if item["number"] == 880)
    assert delivery["status"] == "merged"
    assert "c6e142f18e86579548e3a5c29118dd1ccd9365b0" in delivery["evidence"]
    assert "5e0c4cbeb4432b19f50439831d226e1db5577539" in delivery["evidence"]
    assert "pull/880" in index
