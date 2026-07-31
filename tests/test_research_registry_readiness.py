"""Contracts for the research-software registry handoff."""

import json
from pathlib import Path

from scripts.validate_external_track_handoff import validate_handoff

HANDOFF = Path(
    "conductor/tracks/research_software_registry_readiness_20260721/"
    "handoff/registry-readiness.json"
)


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
        == "revision_in_progress_pending_human_and_external_evidence"
    )
    remaining = handoff["joss_submission_evidence"]["remaining_submission_gates"]
    assert not any("exact v2 release" in gate for gate in remaining)
    assert any("AI-policy attestation" in gate for gate in remaining)
    assert any("research-workflow use" in gate for gate in remaining)
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
