"""Tests for the HPSF external-gate handoff."""

from pathlib import Path

from scripts.validate_external_track_handoff import validate_handoff

HANDOFF = next(
    root / "hpsf-curation-submission-followthrough_20260625/handoff/hpsf-evidence.json"
    for root in (Path("conductor/tracks"), Path("conductor/archive"))
    if (root / "hpsf-curation-submission-followthrough_20260625").is_dir()
)


def test_hpsf_handoff_preserves_governance_and_review_gate() -> None:
    summary = validate_handoff(HANDOFF)
    assert summary["status"] == "blocked"
    assert summary["command_count"] == 5
    assert summary["evidence_count"] == 5
