"""Governance boundaries for information-source portfolio VOI."""

from __future__ import annotations

import json
from pathlib import Path

TRACK = Path("conductor/archive/information_source_portfolio_voi_20260801")


def test_repository_panel_does_not_satisfy_scientific_promotion_gate() -> None:
    """An agent review cannot replace independent scientific evidence."""
    metadata = json.loads((TRACK / "metadata.json").read_text(encoding="utf-8"))
    gates = {gate["id"]: gate for gate in metadata["gates"]}

    assert gates["scientific-review-panel"]["kind"] == "repository_review"
    assert gates["scientific-review-panel"]["status"] == "satisfied"
    assert gates["scientific-review"]["kind"] == "scientific_review"
    assert gates["scientific-review"]["status"] == "pending"


def test_track_surfaces_preserve_experimental_review_boundary() -> None:
    """The index and panel must retain the non-promotion boundary."""
    index = (TRACK / "index.md").read_text(encoding="utf-8")
    panel = (TRACK / "scientific-review-panel-20260801.md").read_text(encoding="utf-8")

    assert "Repository scientific review panel" in index
    assert "Independent scientific evidence" in index
    assert "does not authorize" in panel
    assert "stable promotion" in panel
