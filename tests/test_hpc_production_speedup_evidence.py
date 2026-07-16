"""Tests for production-speedup evidence boundaries."""

from pathlib import Path

import pytest

from scripts.validate_production_speedup_evidence import validate_manifest

MANIFEST = Path(
    "conductor/tracks/hpc-production-speedup-evidence-program_20260625/handoff/production-speedup-manifest.json"
)


def test_manifest_indexes_cpu_reference_and_external_gates() -> None:
    index = validate_manifest(MANIFEST)
    assert [item["backend"] for item in index["packets"]] == [
        "cpu", "gpu", "tpu", "metal", "fpga", "asic"
    ]
    assert [item["status"] for item in index["packets"]] == [
        "passed", "blocked", "blocked", "blocked", "not_available", "not_available"
    ]
    assert all(item["sha256"] for item in index["packets"])


def test_passed_packet_cannot_overclaim_representative_workload(tmp_path: Path) -> None:
    packet = tmp_path / "packet.json"
    packet.write_text(
        '{"backend":"gpu","status":"passed","owner":"test","source_command":"x",'
        '"timestamp":"2026-07-17T00:00:00Z","workload_hash":"sha256:test",'
        '"workload_scale":"representative","warmup_iterations":1,"timing_seconds":1.0,'
        '"throughput":2.0,"device_metadata":{"device":"test"},"cpu_fallback":true,'
        '"cpu_comparison_seconds":2.0,"speedup":2.0,"review_status":"reviewed"}',
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"packets":[{"artifact_path":"packet.json"}]}', encoding="utf-8")
    with pytest.raises(ValueError, match="production workload_scale"):
        validate_manifest(manifest)
