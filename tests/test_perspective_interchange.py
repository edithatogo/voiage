"""Arrow interchange, fixture durability, and performance contract tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import polars as pl
import pyarrow.parquet as pq

from scripts.benchmark_perspective_interchange import benchmark
from scripts.perspective_interchange_manifest import (
    RELATIVE_ROOT,
    artifact_hash,
    verify_artifacts,
)

ROOT = Path(__file__).resolve().parents[1]
INTERCHANGE_ROOT = ROOT / RELATIVE_ROOT


def test_golden_arrow_fixtures_and_manifest_are_current() -> None:
    """Committed Parquet and IPC artifacts must match their source and hashes."""
    assert verify_artifacts(ROOT) == []
    manifest = json.loads((INTERCHANGE_ROOT / "manifest.json").read_text())
    assert manifest["schema_version"] == "1.0.0"
    assert len(manifest["arrow_schema_fingerprint_sha256"]) == 64
    assert manifest["compatibility"]["previous_fixture_versions_readable"] == ["1.0.0"]


def test_pyarrow_polars_round_trip_preserves_values_and_types() -> None:
    """Polars must consume and reproduce every Arrow value and logical type."""
    arrow_table = pq.read_table(INTERCHANGE_ROOT / "perspective-result-v1.parquet")
    frame = pl.from_arrow(arrow_table)
    round_trip = frame.to_arrow()
    expected_schema = arrow_table.schema.remove_metadata()
    compatible = round_trip.cast(expected_schema)
    assert compatible.schema == expected_schema
    assert compatible.equals(arrow_table.replace_schema_metadata(None))


def test_arrow_fixtures_are_readable_in_a_fresh_process() -> None:
    """Golden files must cross a process boundary without package-local state."""
    program = """
import hashlib, json
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
root = Path(__import__('sys').argv[1])
parquet = pq.read_table(root / 'perspective-result-v1.parquet')
with pa.memory_map(str(root / 'perspective-result-v1.arrow'), 'r') as source:
    ipc = pa.ipc.open_file(source).read_all()
print(json.dumps({
    'equal': parquet.equals(ipc, check_metadata=True),
    'rows': parquet.num_rows,
    'fingerprint': hashlib.sha256(parquet.schema.serialize().to_pybytes()).hexdigest(),
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", program, str(INTERCHANGE_ROOT)],
        check=True,
        capture_output=True,
        text=True,
    )
    observed = json.loads(completed.stdout)
    manifest = json.loads((INTERCHANGE_ROOT / "manifest.json").read_text())
    assert observed == {
        "equal": True,
        "rows": 4,
        "fingerprint": manifest["arrow_schema_fingerprint_sha256"],
    }


def test_lf_normalized_hash_is_platform_independent(tmp_path: Path) -> None:
    """Text evidence hashes must be identical for LF and CRLF checkouts."""
    lf = tmp_path / "lf.json"
    crlf = tmp_path / "crlf.json"
    lf.write_bytes(b'{\n  "value": 1\n}\n')
    crlf.write_bytes(b'{\r\n  "value": 1\r\n}\r\n')
    expected = hashlib.sha256(lf.read_bytes()).hexdigest()
    assert artifact_hash(lf, "lf-text") == expected
    assert artifact_hash(crlf, "lf-text") == expected


def test_serialization_benchmark_contract() -> None:
    """Representative Arrow output must stay compact and materially performant."""
    metrics = benchmark(rows=4_000, repeats=3)
    assert metrics["parquet_to_jsonl_time_ratio"] < 2.0
    assert metrics["parquet_to_jsonl_size_ratio"] < 0.75
