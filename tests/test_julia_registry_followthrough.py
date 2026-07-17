"""Tests for the Julia General external-gate handoff."""

from pathlib import Path

from scripts.validate_external_track_handoff import validate_handoff

HANDOFF = next(
    root
    / "julia-general-registry-publication_20260625/handoff/julia-registry-evidence.json"
    for root in (Path("conductor/tracks"), Path("conductor/archive"))
    if (root / "julia-general-registry-publication_20260625").is_dir()
)


def test_julia_registry_handoff_preserves_local_tests_and_external_approval() -> None:
    summary = validate_handoff(HANDOFF)
    assert summary["status"] == "blocked"
    assert summary["command_count"] == 5
    assert summary["evidence_count"] == 4
