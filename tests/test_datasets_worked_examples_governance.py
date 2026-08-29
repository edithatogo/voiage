"""Governance boundaries for dataset worked-example delivery."""

from __future__ import annotations

import json
from pathlib import Path

TRACK = Path("conductor/archive/datasets_worked_examples_20260723")


def test_superseded_status_is_consistent_without_claiming_delivery() -> None:
    """Registry, metadata, and index agree on the superseded lifecycle."""
    metadata = json.loads((TRACK / "metadata.json").read_text(encoding="utf-8"))
    index = (TRACK / "index.md").read_text(encoding="utf-8")
    normalized_index = " ".join(index.split())
    registry = Path("conductor/tracks.md").read_text(encoding="utf-8")

    assert metadata["status"] == "completed"
    assert metadata["legacy_outcome"] == "superseded"
    assert metadata["superseded_by"] == (
        "pre_submission_comprehensive_hardening_20260829"
    )
    assert "Status: in progress" in index
    assert "## [x] Track: Datasets and Executable Worked Examples" in registry
    assert "Status: superseded on 2026-08-29" in index
    assert "delivery evidence" in normalized_index
    assert "remain pending" in normalized_index


def test_governance_reconciliation_preserves_delivery_and_review_gates() -> None:
    """G1-G4 completion cannot promote G5-G15 or external evidence."""
    plan = (TRACK / "plan.md").read_text(encoding="utf-8")
    metadata = json.loads((TRACK / "metadata.json").read_text(encoding="utf-8"))
    gates = {gate["id"]: gate for gate in metadata["gates"]}

    for task in range(1, 5):
        assert f"- [x] **G{task}:**" in plan
    assert "- **Migrated:** **G5:**" in plan
    assert "- **Migrated:** **G15:**" in plan
    assert gates["scientific-and-contract-review"]["status"] == "pending"
    assert gates["hosted-required-checks"]["status"] == "pending"


def test_delivery_pull_requests_match_canonical_manifest() -> None:
    """Track metadata and the canonical manifest share live-verified PR state."""
    metadata = json.loads((TRACK / "metadata.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        Path("conductor/github-cross-references.json").read_text(encoding="utf-8")
    )
    entry = next(
        item
        for item in manifest["tracks"]
        if item["track_id"] == "datasets_worked_examples_20260723"
    )
    pull_requests = {item["number"]: item for item in entry["pull_requests"]}

    assert {item["url"] for item in entry["pull_requests"]} == set(
        metadata["github_cross_reference"]["pull_requests"]
    )
    assert pull_requests[621]["status"] == "merged"
    assert "b86a7d1aa08896eec2f83ab786c13c25a7fff3a3" in pull_requests[621]["evidence"]
    assert pull_requests[818]["status"] == "open"
