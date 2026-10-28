"""Compatibility and bounded-allocation checks for normalized input digests."""

from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pyarrow as pa
from pyarrow import ipc
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
