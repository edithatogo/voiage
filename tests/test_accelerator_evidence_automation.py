"""Tests for the unified accelerator evidence packet validator."""

import json
from pathlib import Path

import pytest

from scripts.validate_accelerator_evidence import validate_manifest

TRACK_ID = "accelerator-evidence-automation_20260625"
MANIFEST = next(
    root / "handoff/accelerator-evidence-manifest.json"
    for root in (
        Path("conductor/tracks") / TRACK_ID,
        Path("conductor/archive") / TRACK_ID,
    )
    if (root / "handoff/accelerator-evidence-manifest.json").is_file()
)


def test_manifest_indexes_passed_and_blocked_accelerator_packets() -> None:
    index = validate_manifest(MANIFEST)
    assert [item["runtime"] for item in index["packets"]] == ["gpu", "tpu", "metal"]
    assert [item["status"] for item in index["packets"]] == [
        "passed",
        "blocked",
        "blocked",
    ]
    assert all(item["sha256"] for item in index["packets"])


def test_validator_rejects_passed_packet_without_measurement(tmp_path: Path) -> None:
    packet = tmp_path / "packet.json"
    packet.write_text(
        json.dumps({"runtime": "gpu", "status": "passed"}), encoding="utf-8"
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"packets": [{"artifact_path": packet.name}]}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="missing fields"):
        validate_manifest(manifest)
