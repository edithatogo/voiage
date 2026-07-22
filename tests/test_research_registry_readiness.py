"""Contracts for the research-software registry handoff."""

from pathlib import Path

from scripts.validate_external_track_handoff import validate_handoff

HANDOFF = Path(
    "conductor/tracks/research_software_registry_readiness_20260721/"
    "handoff/registry-readiness.json"
)


def test_registry_handoff_preserves_release_and_external_gates() -> None:
    summary = validate_handoff(HANDOFF)

    assert summary == {
        "track_id": "research_software_registry_readiness_20260721",
        "channel": "research-software-registries",
        "status": "blocked",
        "command_count": 4,
        "evidence_count": 4,
    }
