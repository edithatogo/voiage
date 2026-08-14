"""Governance boundaries for controlled live dataset probes."""

from __future__ import annotations

import json
from pathlib import Path

TRACK = Path("conductor/tracks/controlled_live_dataset_interoperability_20260801")


def test_live_probe_track_is_active_but_authorization_blocked() -> None:
    """Fail-closed tests do not authorize source access or network I/O."""
    metadata = json.loads((TRACK / "metadata.json").read_text(encoding="utf-8"))
    index = (TRACK / "index.md").read_text(encoding="utf-8")
    registry = Path("conductor/tracks.md").read_text(encoding="utf-8")
    gates = {gate["id"]: gate for gate in metadata["gates"]}

    assert metadata["status"] == "in_progress"
    assert "Status: in progress but authorization-blocked" in index
    assert "## [~] Track: Controlled Live" in registry
    assert gates["source-rights-and-use-authority"]["status"] == "pending"
    assert gates["hosted-required-checks"]["status"] == "satisfied"
    assert "does not authorize network retrieval" in index


def test_probe_assurance_does_not_complete_controlled_source_work() -> None:
    """Delivery assurance does not complete source authorization work."""
    plan = (TRACK / "plan.md").read_text(encoding="utf-8")

    assert "- [ ] **L1 / AC-01:**" in plan
    assert "- [x] **L2 / AC-01:**" in plan
    assert "- [x] **L3 / AC-01:**" in plan
    assert "- [ ] **L4 / AC-02:**" in plan
    assert "- [x] **L7 / AC-02--AC-04:**" in plan
    assert "does not satisfy L1 or authorize network I/O" in plan
