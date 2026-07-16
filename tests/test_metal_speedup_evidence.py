"""Tests for Apple Metal evidence boundaries."""

from pathlib import Path

import pytest

from scripts.validate_metal_speedup_evidence import validate_manifest

TRACK_ID = "apple-metal-production-speedup-evidence_20260625"
MANIFEST = next(
    root / "handoff/metal-manifest.json"
    for root in (
        Path("conductor/tracks") / TRACK_ID,
        Path("conductor/archive") / TRACK_ID,
    )
    if (root / "handoff/metal-manifest.json").is_file()
)


def test_manifest_indexes_cpu_reference_and_metal_gate() -> None:
    index = validate_manifest(MANIFEST)
    assert [item["backend"] for item in index["packets"]] == ["cpu", "metal"]
    assert [item["status"] for item in index["packets"]] == ["passed", "blocked"]
    assert all(item["sha256"] for item in index["packets"])


def test_validator_rejects_metal_without_parity(tmp_path: Path) -> None:
    packet = tmp_path / "packet.json"
    packet.write_text(
        '{"backend":"metal","status":"passed","owner":"test","source_command":"x",'
        '"timestamp":"2026-07-17T00:00:00Z","workload_hash":"sha256:test",'
        '"workload_scale":"production","device_metadata":{"device":"MPS"},'
        '"warmup_iterations":1,"timing_seconds":1.0,"throughput":2.0,'
        '"cpu_baseline_seconds":2.0,"speedup":2.0,"cpu_fallback":false,'
        '"result_parity":false,"claim_level":"production_speedup"}',
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        '{"packets":[{"artifact_path":"packet.json"}]}', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="baseline and parity"):
        validate_manifest(manifest)
