"""Compatibility and bounded-allocation checks for normalized input digests."""

from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pyarrow as pa
from pyarrow import ipc
from pyarrow import parquet as pq
import pytest

import voiage
from voiage.contracts.normalized_input import (
    DatasetManifest,
    FieldManifest,
    NormalizedInputBundle,
    SourceProvenance,
    TableManifest,
)


def _bundle(values: np.ndarray) -> NormalizedInputBundle:
    return NormalizedInputBundle(
        manifest=DatasetManifest(
            dataset_id="digest-memory",
            tables=(
                TableManifest(
                    table_id="samples",
                    fields=(FieldManifest(field_id="value", dtype="float64"),),
                ),
            ),
            provenance=SourceProvenance(
                provider_id="test",
                source_uri="file:///redacted",
                descriptor_digest="a" * 64,
            ),
        ),
        tables={"samples": pa.table({"value": values})},
    )


@pytest.mark.parametrize(
    "column",
    [
        pa.chunked_array([[1.0, None], [3.0, 4.0]]),
        pa.chunked_array([pa.array([], type=pa.float64())]),
        pa.chunked_array([["alpha", None, "beta"]]),
        pa.chunked_array([pa.array(["a", "b", "a"]).dictionary_encode()]),
        pa.chunked_array([pa.array([[1, 2], None, [3]])]),
    ],
    ids=["chunked", "empty", "strings", "dictionary", "nested"],
)
def test_streamed_digest_preserves_existing_ipc_bytes(column: pa.ChunkedArray) -> None:
    original = _bundle(np.arange(2, dtype=np.float64))
    table = pa.table({"value": column}).replace_schema_metadata({b"origin": b"test"})
    bundle = NormalizedInputBundle(
        manifest=original.manifest, tables={"samples": table}
    )
    digest = sha256(bundle.canonical_json.encode())
    for name in sorted(bundle.tables):
        table = bundle.tables[name]
        sink = pa.BufferOutputStream()
        with ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)
        digest.update(name.encode())
        digest.update(sink.getvalue().to_pybytes())
    assert bundle.content_digest == digest.hexdigest()


def test_streamed_digest_preserves_multi_table_framing() -> None:
    original = _bundle(np.arange(3, dtype=np.float64))
    second = TableManifest(
        table_id="other", fields=(FieldManifest(field_id="label", dtype="string"),)
    )
    manifest = original.manifest.model_copy(
        update={"tables": (*original.manifest.tables, second)}
    )
    tables = {
        "samples": original.tables["samples"],
        "other": pa.table({"label": ["x"]}),
    }
    bundle = NormalizedInputBundle(manifest=manifest, tables=tables)
    digest = sha256(bundle.canonical_json.encode())
    for name in sorted(tables):
        sink = pa.BufferOutputStream()
        with ipc.new_stream(sink, tables[name].schema) as writer:
            writer.write_table(tables[name])
        digest.update(name.encode())
        digest.update(sink.getvalue().to_pybytes())
    assert bundle.content_digest == digest.hexdigest()


def test_ipc_round_trip_preserves_units_and_source_identity(tmp_path: Path) -> None:
    """The digest remains bound to declared units and source provenance."""
    provenance = SourceProvenance(
        provider_id="fixture",
        source_uri="file:///redacted/parameters.arrow",
        descriptor_digest="b" * 64,
        governance={"source_revision": "synthetic-v1"},
    )

    def manifest(unit: str) -> DatasetManifest:
        return DatasetManifest(
            dataset_id="parameter-fixture",
            tables=(
                TableManifest(
                    table_id="samples",
                    fields=(
                        FieldManifest(field_id="value", dtype="float64", unit=unit),
                    ),
                ),
            ),
            provenance=provenance,
        )

    table = pa.table({"value": [1.0, 2.0]})
    bundle = NormalizedInputBundle(manifest=manifest("kg"), tables={"samples": table})
    path = tmp_path / "parameters.arrow"
    bundle.write_ipc(path)
    restored = NormalizedInputBundle.read_ipc(path)
    assert restored.manifest == bundle.manifest
    assert restored.manifest.schema_version == "1.0.0"
    assert restored.manifest.tables[0].fields[0].unit == "kg"
    assert restored.manifest.provenance == provenance
    assert restored.content_digest == bundle.content_digest
    changed_unit = NormalizedInputBundle(
        manifest=manifest("g"), tables={"samples": table}
    )
    assert changed_unit.content_digest != bundle.content_digest
    changed_provenance = SourceProvenance(
        provider_id="fixture",
        source_uri="file:///redacted/parameters.arrow",
        descriptor_digest="c" * 64,
        governance={"source_revision": "synthetic-v1"},
    )
    changed_source = NormalizedInputBundle(
        manifest=DatasetManifest(
            dataset_id="parameter-fixture",
            tables=manifest("kg").tables,
            provenance=changed_provenance,
        ),
        tables={"samples": table},
    )
    assert changed_source.content_digest != bundle.content_digest


@pytest.mark.parametrize("format_name", ["ipc", "parquet"])
@pytest.mark.parametrize(
    "original_metadata",
    [
        None,
        {},
        {b"parameter-origin": b"synthetic-fixture", b"unit-context": b"kg"},
        {
            b"voiage.normalized.manifest": b"original-manifest",
            b"voiage.normalized.table_id": b"original-table",
            b"voiage.normalized.original_schema_metadata": b"original-envelope",
            b"voiage.normalized.original_arrow_schema": b"original-schema",
            b"binary": b"\x00\xff",
        },
    ],
    ids=["none", "empty", "ordinary", "reserved-key-collisions"],
)
def test_round_trip_preserves_original_arrow_schema_metadata(
    tmp_path: Path, format_name: str, original_metadata: dict[bytes, bytes] | None
) -> None:
    original = _bundle(np.arange(3, dtype=np.float64))
    table = original.tables["samples"].replace_schema_metadata(original_metadata)
    bundle = NormalizedInputBundle(
        manifest=original.manifest, tables={"samples": table}
    )
    path = tmp_path / ("input.arrow" if format_name == "ipc" else "input.parquet")
    getattr(bundle, f"write_{format_name}")(path)
    restored = getattr(NormalizedInputBundle, f"read_{format_name}")(path)
    assert restored.tables["samples"].schema.metadata == table.schema.metadata
    assert restored.content_digest == bundle.content_digest


@pytest.mark.parametrize("format_name", ["ipc", "parquet"])
@pytest.mark.parametrize(
    "legacy_value",
    [
        b"legacy-user-value",
        b"voiage-schema-metadata-v1:null",
        b"voiage-schema-metadata-v1:invalid",
    ],
)
def test_legacy_bundle_restores_nonreserved_schema_metadata(
    tmp_path: Path, format_name: str, legacy_value: bytes
) -> None:
    bundle = _bundle(np.arange(3, dtype=np.float64))
    metadata = {
        b"voiage.normalized.manifest": bundle.canonical_json.encode(),
        b"voiage.normalized.table_id": b"samples",
        b"source": b"legacy",
        b"voiage.normalized.original_schema_metadata": legacy_value,
    }
    table = bundle.tables["samples"].replace_schema_metadata(metadata)
    path = tmp_path / ("legacy.arrow" if format_name == "ipc" else "legacy.parquet")
    if format_name == "ipc":
        with ipc.new_file(path, table.schema) as writer:
            writer.write_table(table)
    else:
        pq.write_table(table, path)
    restored = getattr(NormalizedInputBundle, f"read_{format_name}")(path)
    assert restored.tables["samples"].schema.metadata == {
        b"source": b"legacy",
        b"voiage.normalized.original_schema_metadata": legacy_value,
    }


@pytest.mark.parametrize("format_name", ["ipc", "parquet"])
def test_marked_bundle_rejects_missing_metadata_envelope(
    tmp_path: Path, format_name: str
) -> None:
    bundle = _bundle(np.arange(3, dtype=np.float64))
    metadata = {
        b"voiage.normalized.manifest": bundle.canonical_json.encode() + b"\n ",
        b"voiage.normalized.table_id": b"samples",
    }
    table = bundle.tables["samples"].replace_schema_metadata(metadata)
    path = tmp_path / ("invalid.arrow" if format_name == "ipc" else "invalid.parquet")
    if format_name == "ipc":
        with ipc.new_file(path, table.schema) as writer:
            writer.write_table(table)
    else:
        pq.write_table(table, path)
    with pytest.raises(ValueError, match="invalid original Arrow schema metadata"):
        getattr(NormalizedInputBundle, f"read_{format_name}")(path)


@pytest.mark.parametrize("format_name", ["ipc", "parquet"])
@pytest.mark.parametrize(
    "payload",
    [
        b"not-json",
        b"\xff",
        b"{}",
        b"[1]",
        b'[["00"]]',
        b'[[0,"00"]]',
        b'[["zz","00"]]',
        b'[["00","01"],["00","02"]]',
    ],
    ids=[
        "json",
        "unicode",
        "non-list",
        "entry-type",
        "entry-length",
        "element-type",
        "hex",
        "duplicate",
    ],
)
def test_marked_bundle_rejects_malformed_metadata_envelope(
    tmp_path: Path, format_name: str, payload: bytes
) -> None:
    bundle = _bundle(np.arange(3, dtype=np.float64))
    metadata = {
        b"voiage.normalized.manifest": bundle.canonical_json.encode() + b"\n ",
        b"voiage.normalized.table_id": b"samples",
        b"voiage.normalized.original_schema_metadata": b"voiage-schema-metadata-v1:"
        + payload,
    }
    table = bundle.tables["samples"].replace_schema_metadata(metadata)
    path = tmp_path / ("invalid.arrow" if format_name == "ipc" else "invalid.parquet")
    if format_name == "ipc":
        with ipc.new_file(path, table.schema) as writer:
            writer.write_table(table)
    else:
        pq.write_table(table, path)
    with pytest.raises(ValueError, match="original Arrow schema metadata"):
        getattr(NormalizedInputBundle, f"read_{format_name}")(path)


def test_digest_does_not_buffer_the_complete_arrow_stream(tmp_path: Path) -> None:
    """Measure in a child because Arrow's default memory pool is process-wide."""
    probe = """
import importlib.util, json, sys, tracemalloc
sys.path.insert(0, sys.argv[1])
spec = importlib.util.spec_from_file_location("digest_probe", sys.argv[2])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
bundle = module._bundle(module.np.arange(2_000_000, dtype=module.np.float64))
pa = module.pa
original = pa.default_memory_pool()
observed = pa.proxy_memory_pool(original)
pa.set_memory_pool(observed)
tracemalloc.start()
try:
    expected = bundle.content_digest
    assert len(expected) == 64
    for _ in range(3):
        assert bundle.content_digest == expected
        assert observed.bytes_allocated() == 0
    _, python_peak = tracemalloc.get_traced_memory()
finally:
    tracemalloc.stop()
    pa.set_memory_pool(original)
print(json.dumps({"arrow_peak": observed.max_memory(), "python_peak": python_peak}))
"""
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            probe,
            str(Path(voiage.__file__).resolve().parent.parent),
            str(Path(__file__).resolve()),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    measurement = json.loads(result.stdout)
    assert measurement["arrow_peak"] < 2_000_000
    assert measurement["python_peak"] < 2_000_000
