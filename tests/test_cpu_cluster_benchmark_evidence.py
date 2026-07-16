"""Tests for CPU cluster benchmark evidence boundaries."""

from pathlib import Path

import pytest

from scripts.validate_cpu_cluster_benchmark_evidence import validate_manifest
from voiage.parallel.distributed import distributed_map

TRACK_ID = "cpu-cluster-production-benchmark-evidence_20260625"
MANIFEST = next(
    root / "handoff/cpu-cluster-manifest.json"
    for root in (
        Path("conductor/tracks") / TRACK_ID,
        Path("conductor/archive") / TRACK_ID,
    )
    if (root / "handoff/cpu-cluster-manifest.json").is_file()
)


def _double(value: int) -> int:
    return value * 2


def test_manifest_indexes_reference_smoke_and_blocked_cluster_packets() -> None:
    index = validate_manifest(MANIFEST)
    assert [item["scheduler"] for item in index["packets"]] == [
        "single-process",
        "local-process",
        "dask-or-ray",
    ]
    assert [item["status"] for item in index["packets"]] == [
        "passed",
        "passed",
        "blocked",
    ]
    assert all(item["sha256"] for item in index["packets"])


def test_validator_rejects_unreviewed_production_speedup(tmp_path: Path) -> None:
    packet = tmp_path / "packet.json"
    packet.write_text(
        '{"scheduler":"local-process","status":"passed","owner":"test",'
        '"source_command":"x","timestamp":"2026-07-17T00:00:00Z",'
        '"workload_hash":"sha256:test","workload_scale":"representative",'
        '"worker_count":2,"node_count":1,"warmup_iterations":1,"timing_seconds":1.0,'
        '"throughput":2.0,"cpu_baseline_seconds":2.0,"speedup":2.0,'
        '"claim_level":"production_speedup","result_envelope_preserved":true,'
        '"diagnostics_preserved":true}',
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        '{"packets":[{"artifact_path":"packet.json"}]}', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="production workload"):
        validate_manifest(manifest)


def test_local_scheduler_preserves_ordered_result_envelope() -> None:
    assert distributed_map([1, 2, 3], _double, n_workers=2, use_processes=False) == [
        2,
        4,
        6,
    ]
