"""Tests for the Conductor GitHub cross-reference contract."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "validate_conductor_github_cross_references.py"


def _validator() -> ModuleType:
    spec = importlib.util.spec_from_file_location("conductor_github_xref", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_repository_cross_references_are_complete() -> None:
    """Every local or proposed track has a valid unique GitHub record."""
    assert _validator().validate(ROOT) == []


def test_manifest_preserves_no_pr_evidence_boundary() -> None:
    """Legacy tracks without a provable PR state that boundary explicitly."""
    manifest = json.loads(
        (ROOT / "conductor" / "github-cross-references.json").read_text()
    )
    completed_without_prs = [
        entry
        for entry in manifest["tracks"]
        if entry["lifecycle"] == "completed" and not entry["pull_requests"]
    ]
    assert completed_without_prs
    assert all(
        entry["pull_request_evidence"] == "none_found"
        for entry in completed_without_prs
    )


def test_expected_utility_track_metadata_and_manifest_share_delivery_prs() -> None:
    """The #595 track owns its VOIAGE delivery PR as well as canonical planning."""
    manifest = json.loads(
        (ROOT / "conductor" / "github-cross-references.json").read_text()
    )
    track_id = "risk_adjusted_information_pricing_20260731"
    entry = next(item for item in manifest["tracks"] if item["track_id"] == track_id)
    metadata = json.loads(
        (ROOT / "conductor" / "tracks" / track_id / "metadata.json").read_text()
    )

    manifest_urls = {item["url"] for item in entry["pull_requests"]}
    assert manifest_urls == set(metadata["github_cross_reference"]["pull_requests"])
    assert "https://github.com/edithatogo/voiage/pull/712" in manifest_urls


def test_external_landscape_track_preserves_phase_one_pr_boundaries() -> None:
    """The canonical manifest owns both planning and open delivery PRs."""
    manifest = json.loads(
        (ROOT / "conductor" / "github-cross-references.json").read_text()
    )
    track_id = "external_voi_library_feature_parity_20260723"
    entry = next(item for item in manifest["tracks"] if item["track_id"] == track_id)
    track = ROOT / "conductor" / "tracks" / track_id
    metadata = json.loads((track / "metadata.json").read_text())
    index = (track / "index.md").read_text()
    plan = (track / "plan.md").read_text()
    gates = {gate["id"]: gate for gate in metadata["gates"]}

    manifest_prs = {item["url"]: item["status"] for item in entry["pull_requests"]}
    assert set(manifest_prs) == set(metadata["github_cross_reference"]["pull_requests"])
    assert manifest_prs["https://github.com/edithatogo/voiage/pull/621"] == "merged"
    assert manifest_prs["https://github.com/edithatogo/voiage/pull/819"] == "open"
    assert "Merged planning PR #621" in index
    assert "Delivery PR #819" in index
    assert "- [x] **G4:**" in plan
    assert "- [ ] **G5:**" in plan
    assert "- [ ] **G15:**" in plan
    assert gates["scientific-and-contract-review"]["status"] == "pending"
    assert gates["hosted-required-checks"]["status"] == "pending"
