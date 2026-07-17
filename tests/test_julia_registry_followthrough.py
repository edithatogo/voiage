"""Tests for the Julia General external-gate handoff."""

from pathlib import Path

from scripts.validate_external_track_handoff import validate_handoff

HANDOFF = Path(
    "conductor/tracks/julia-general-registry-publication_20260625/handoff/julia-registry-evidence.json"
)


def test_julia_registry_handoff_preserves_local_tests_and_external_approval() -> None:
    summary = validate_handoff(HANDOFF)
    assert summary["status"] == "blocked"
    assert summary["command_count"] == 5
    assert summary["evidence_count"] == 4
