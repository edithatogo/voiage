"""Tests for external registry publication evidence boundaries."""

import json
from pathlib import Path

import pytest

from scripts.validate_registry_publication_evidence import validate_manifest

MANIFEST = Path(
    "conductor/tracks/external-registry-publication-program_20260625/handoff/registry-manifest.json"
)


def test_registry_manifest_covers_all_channels_and_external_gates() -> None:
    summary = validate_manifest(MANIFEST)
    assert summary["channel_count"] == 13
    assert "python" in summary["channels"]
    assert "e4s" in summary["channels"]


def test_registry_manifest_rejects_unresolved_channel_without_gate(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.json"
    channel = {
        "channel": "x",
        "package": "x",
        "registry": "x",
        "status": "not_found",
        "owner": "owner",
        "next_action": "next",
        "external_gate": "external review",
        "evidence_url": "https://example.test",
        "checked_at": "2026-07-17",
    }
    channels = [{**channel, "channel": str(index)} for index in range(12)]
    channels.append({**channel, "channel": "missing-gate", "external_gate": "none"})
    manifest.write_text(json.dumps({"channels": channels}), encoding="utf-8")
    with pytest.raises(ValueError, match="external gate"):
        validate_manifest(manifest)
