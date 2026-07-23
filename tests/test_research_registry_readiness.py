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
        "command_count": 4,
        "evidence_count": 4,
    }


def test_registry_track_records_native_paper_issue_hierarchy() -> None:
    """The arXiv lane remains traceable from Conductor and the GitHub parent."""
    track = HANDOFF.parent.parent
    metadata = json.loads((track / "metadata.json").read_text())
    plan = (track / "plan.md").read_text()
    specification = (track / "spec.md").read_text()

    arxiv_issue = "https://github.com/edithatogo/voiage/issues/312"
    assert arxiv_issue in metadata["github_subissues"]
    assert arxiv_issue in plan
    assert arxiv_issue in specification
    assert "PR #311" in plan
