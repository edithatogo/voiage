"""Governance boundaries for dataset worked-example delivery."""

from __future__ import annotations

import json
from pathlib import Path

TRACK = Path("conductor/tracks/datasets_worked_examples_20260723")


def test_phase_one_status_is_consistent_without_claiming_delivery() -> None:
    """Registry, metadata, and index agree on bounded in-progress status."""
    metadata = json.loads((TRACK / "metadata.json").read_text(encoding="utf-8"))
    index = (TRACK / "index.md").read_text(encoding="utf-8")
    normalized_index = " ".join(index.split())
    registry = Path("conductor/tracks.md").read_text(encoding="utf-8")

    assert metadata["status"] == "in_progress"
    assert "Status: in progress" in index
    assert "## [~] Track: Datasets and Executable Worked Examples" in registry
    assert "delivery evidence" in normalized_index
    assert "remain pending" in normalized_index


def test_governance_reconciliation_preserves_delivery_and_review_gates() -> None:
    """G1-G4 completion cannot promote G5-G15 or external evidence."""
    plan = (TRACK / "plan.md").read_text(encoding="utf-8")
    metadata = json.loads((TRACK / "metadata.json").read_text(encoding="utf-8"))
    gates = {gate["id"]: gate for gate in metadata["gates"]}

    for task in range(1, 5):
        assert f"- [x] **G{task}:**" in plan
    assert "- [ ] **G5:**" in plan
    assert "- [ ] **G15:**" in plan
    assert gates["scientific-and-contract-review"]["status"] == "pending"
    assert gates["hosted-required-checks"]["status"] == "pending"
