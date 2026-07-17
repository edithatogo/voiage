"""Tests for the FPGA physical-runtime external-gate handoff."""

from pathlib import Path

from scripts.validate_external_track_handoff import validate_handoff

HANDOFF = next(
    root
    / "fpga-physical-board-runtime-evidence_20260625/handoff/fpga-runtime-evidence.json"
    for root in (Path("conductor/tracks"), Path("conductor/archive"))
    if (root / "fpga-physical-board-runtime-evidence_20260625").is_dir()
)


def test_fpga_handoff_preserves_pre_silicon_and_board_gate() -> None:
    summary = validate_handoff(HANDOFF)
    assert summary["status"] == "blocked"
    assert summary["command_count"] == 4
    assert summary["evidence_count"] == 4
