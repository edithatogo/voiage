"""Tests for the E4S external-gate handoff."""

from pathlib import Path

from scripts.validate_external_track_handoff import validate_handoff

HANDOFF = next(
    root / "e4s-inclusion-followthrough_20260625/handoff/e4s-evidence.json"
    for root in (Path("conductor/tracks"), Path("conductor/archive"))
    if (root / "e4s-inclusion-followthrough_20260625").is_dir()
)


def test_e4s_handoff_preserves_prerequisites_and_curation_gate() -> None:
    summary = validate_handoff(HANDOFF)
    assert summary["status"] == "blocked"
    assert summary["command_count"] == 4
    assert summary["evidence_count"] == 5
