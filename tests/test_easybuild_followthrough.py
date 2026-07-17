"""Tests for the EasyBuild external-gate handoff."""

from pathlib import Path

from scripts.validate_external_track_handoff import validate_handoff

HANDOFF = next(
    root
    / "easybuild-easyconfig-merge-followthrough_20260625/handoff/easybuild-evidence.json"
    for root in (Path("conductor/tracks"), Path("conductor/archive"))
    if (root / "easybuild-easyconfig-merge-followthrough_20260625").is_dir()
)


def test_easybuild_handoff_preserves_easyconfig_and_upstream_gate() -> None:
    summary = validate_handoff(HANDOFF)
    assert summary["status"] == "blocked"
    assert summary["command_count"] == 4
    assert summary["evidence_count"] == 3
