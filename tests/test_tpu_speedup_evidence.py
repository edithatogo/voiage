"""Tests for TPU production evidence boundaries."""

from pathlib import Path

import pytest

from scripts.validate_tpu_speedup_evidence import validate_manifest

TRACK_ID = "tpu-production-scale-colab-evidence_20260625"
MANIFEST = Path("conductor/tracks") / TRACK_ID / "handoff/tpu-manifest.json"


def test_manifest_indexes_cpu_reference_and_tpu_gate() -> None:
    index = validate_manifest(MANIFEST)
    assert [item["backend"] for item in index["packets"]] == ["cpu", "tpu"]
    assert [item["status"] for item in index["packets"]] == ["passed", "blocked"]
    assert all(item["sha256"] for item in index["packets"])


def test_validator_rejects_tpu_without_speedup(tmp_path: Path) -> None:
    packet = tmp_path / "packet.json"
    packet.write_text(
        '{"backend":"tpu","status":"passed","owner":"test","source_command":"x",'
        '"timestamp":"2026-07-17T00:00:00Z","device_metadata":{"device":"v5e"},'
        '"workload_hash":"sha256:test","workload_scale":"production",'
        '"warmup_iterations":1,"timing_seconds":1.0,"throughput":2.0,'
        '"cpu_baseline_seconds":2.0,"speedup":1.0,"transfer_overhead_seconds":0.1,'
        '"compile_overhead_seconds":0.1,"result_parity":true,"cpu_fallback":false,'
        '"claim_level":"production_speedup"}',
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        '{"packets":[{"artifact_path":"packet.json"}]}', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="speedup"):
        validate_manifest(manifest)
