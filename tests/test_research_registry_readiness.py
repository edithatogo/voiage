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
        "command_count": 8,
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
    assert arxiv_issue in metadata["github_subissues"]
    assert independent_validation_issue in metadata["github_subissues"]
    assert arxiv_issue in plan
    assert independent_validation_issue in plan
    assert arxiv_issue in specification
    assert independent_validation_issue in specification
    assert handoff["arxiv_preprint_evidence"]["review_pr"].endswith("/pull/311")
    assert handoff["arxiv_preprint_evidence"]["submission_id"] == "7861466"
    assert "submission `7861466` is complete" in plan
    assert handoff["joss_submission_evidence"]["selected_route"] == "direct_joss"
    assert (
        handoff["joss_submission_evidence"]["status"]
        == "repository_ready_pending_external_evidence"
    )
