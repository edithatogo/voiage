"""Tests for the Spack external-gate handoff."""

from pathlib import Path

from scripts.validate_external_track_handoff import validate_handoff

HANDOFF = next(
    root / "spack-package-merge-followthrough_20260625/handoff/spack-evidence.json"
    for root in (Path("conductor/tracks"), Path("conductor/archive"))
    if (root / "spack-package-merge-followthrough_20260625").is_dir()
)


def test_spack_handoff_preserves_draft_recipe_and_upstream_gate() -> None:
    summary = validate_handoff(HANDOFF)
    assert summary["status"] == "blocked"
    assert summary["command_count"] == 4
    assert summary["evidence_count"] == 3
