"""Tests for the custom-circuit production decision handoff."""

from pathlib import Path

from scripts.validate_external_track_handoff import validate_handoff

HANDOFF = next(
    root
    / "custom-circuit-production-acceleration-review_20260625/handoff/production-acceleration-decision.json"
    for root in (Path("conductor/tracks"), Path("conductor/archive"))
    if (root / "custom-circuit-production-acceleration-review_20260625").is_dir()
)


def test_custom_circuit_review_records_no_go_boundary() -> None:
    summary = validate_handoff(HANDOFF)
    assert summary["status"] == "blocked"
    assert summary["command_count"] == 4
    assert summary["evidence_count"] == 4

    payload = HANDOFF.read_text(encoding="utf-8")
    assert '"decision": "no_go"' in payload
    assert "CPU path" in payload
