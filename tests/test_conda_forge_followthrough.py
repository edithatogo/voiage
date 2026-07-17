"""Tests for the Conda-Forge external-gate handoff."""

from pathlib import Path

from scripts.validate_external_track_handoff import validate_handoff

HANDOFF = Path(
    "conductor/tracks/conda-forge-feedstock-publication_20260625/handoff/conda-forge-evidence.json"
)


def test_conda_forge_handoff_preserves_feedstock_gate() -> None:
    summary = validate_handoff(HANDOFF)
    assert summary["status"] == "blocked"
    assert summary["command_count"] == 4
    assert summary["evidence_count"] == 4
