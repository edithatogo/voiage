"""Tests for the ASIC MPW and silicon external-gate handoff."""

from pathlib import Path

from scripts.validate_external_track_handoff import validate_handoff

HANDOFF = next(
    root
    / "asic-mpw-shuttle-and-silicon-evidence_20260625/handoff/asic-silicon-evidence.json"
    for root in (Path("conductor/tracks"), Path("conductor/archive"))
    if (root / "asic-mpw-shuttle-and-silicon-evidence_20260625").is_dir()
)


def test_asic_handoff_preserves_rtl_and_silicon_gate() -> None:
    summary = validate_handoff(HANDOFF)
    assert summary["status"] == "blocked"
    assert summary["command_count"] == 4
    assert summary["evidence_count"] == 6
