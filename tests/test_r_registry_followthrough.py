"""Tests for the R registry external-gate handoff."""

from pathlib import Path

from scripts.validate_external_track_handoff import validate_handoff

HANDOFF = next(
    root / "r-cran-runiverse-publication_20260625/handoff/r-registry-evidence.json"
    for root in (Path("conductor/tracks"), Path("conductor/archive"))
    if (root / "r-cran-runiverse-publication_20260625").is_dir()
)


def test_r_registry_handoff_preserves_local_readiness_and_external_gates() -> None:
    summary = validate_handoff(HANDOFF)
    assert summary["status"] == "blocked"
    assert summary["command_count"] == 6
    assert summary["evidence_count"] == 4
