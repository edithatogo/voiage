"""Synthetic parameter-artifact identity checks for tobac qualification."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

import pyarrow as pa
from pyarrow import parquet as pq
import pytest

from voiage.contracts.normalized_input import (
    DatasetManifest,
    FieldManifest,
    NormalizedInputBundle,
    SourceProvenance,
    TableManifest,
)


@pytest.mark.parametrize("format_name", ["ipc", "parquet"])
def test_parameter_covariance_is_bound_to_units_and_provenance(
    tmp_path: Path, format_name: str
) -> None:
    manifest = DatasetManifest(
        dataset_id="synthetic-parameter-covariance",
        tables=(
            TableManifest(
                table_id="parameters",
                fields=(
                    FieldManifest(field_id="mean", dtype="float64", unit="kg"),
                    FieldManifest(
                        field_id="covariance_row", dtype="fixed_size_list", unit="kg^2"
                    ),
                ),
            ),
        ),
        provenance=SourceProvenance(
            provider_id="synthetic-test",
            source_uri="file:///redacted/parameter-input",
            descriptor_digest="d" * 64,
            governance={"revision": "fixture-v1"},
        ),
    )

    def table(covariance: list[list[float]]) -> pa.Table:
        return pa.table(
            {
                "mean": pa.array([1.0, 2.0]),
                "covariance_row": pa.array(
                    covariance,
                    type=pa.list_(
                        pa.field(
                            "covariance_element",
                            pa.float64(),
                            metadata={b"origin": b"synthetic"},
                        ),
                        2,
                    ),
                ),
            }
        ).replace_schema_metadata({b"parameter_schema": b"synthetic-v1"})

    original_table = table([[1.0, 0.25], [0.25, 4.0]])
    original = NormalizedInputBundle(
        manifest=manifest, tables={"parameters": original_table}
    )
    path = tmp_path / (
        "parameters.arrow" if format_name == "ipc" else "parameters.parquet"
    )
    getattr(original, f"write_{format_name}")(path)
    restored = getattr(NormalizedInputBundle, f"read_{format_name}")(path)
    assert restored.tables["parameters"].equals(original_table, check_metadata=True)
    assert restored.manifest == manifest
    assert restored.content_digest == original.content_digest
    changed = NormalizedInputBundle(
        manifest=manifest, tables={"parameters": table([[1.0, 0.5], [0.5, 4.0]])}
    )
    assert changed.content_digest != original.content_digest


@pytest.mark.parametrize("corruption", ["missing", "invalid", "different-type"])
def test_parquet_rejects_invalid_stored_arrow_schema(
    tmp_path: Path, corruption: str
) -> None:
    manifest = DatasetManifest(
        dataset_id="schema-check",
        tables=(
            TableManifest(
                table_id="parameters",
                fields=(FieldManifest(field_id="mean", dtype="float64"),),
            ),
        ),
        provenance=SourceProvenance(
            provider_id="synthetic-test",
            source_uri="file:///redacted/parameter-input",
            descriptor_digest="e" * 64,
        ),
    )
    bundle = NormalizedInputBundle(
        manifest=manifest, tables={"parameters": pa.table({"mean": [1.0]})}
    )
    path = tmp_path / "parameters.parquet"
    bundle.write_parquet(path)
    stored = pq.read_table(path)
    metadata = dict(stored.schema.metadata or {})
    key = b"voiage.normalized.original_arrow_schema"
    if corruption == "missing":
        del metadata[key]
    elif corruption == "invalid":
        metadata[key] = b"not-an-arrow-schema"
    else:
        metadata[key] = (
            pa.schema([pa.field("mean", pa.int64())]).serialize().to_pybytes()
        )
    pq.write_table(stored.replace_schema_metadata(metadata), path)
    with pytest.raises(ValueError, match="original Arrow schema"):
        NormalizedInputBundle.read_parquet(path)


def test_parquet_reads_previous_envelope_when_user_schema_key_collides(
    tmp_path: Path,
) -> None:
    manifest = DatasetManifest(
        dataset_id="previous-envelope",
        tables=(
            TableManifest(
                table_id="parameters",
                fields=(FieldManifest(field_id="mean", dtype="float64"),),
            ),
        ),
        provenance=SourceProvenance(
            provider_id="synthetic-test",
            source_uri="file:///redacted/parameter-input",
            descriptor_digest="f" * 64,
        ),
    )
    source_metadata = {b"voiage.normalized.original_arrow_schema": b"user-value"}
    encoded = json.dumps(
        [[key.hex(), value.hex()] for key, value in source_metadata.items()],
        separators=(",", ":"),
    ).encode()
    wire_metadata = {
        **source_metadata,
        b"voiage.normalized.manifest": manifest.canonical_json().encode() + b"\n ",
        b"voiage.normalized.table_id": b"parameters",
        b"voiage.normalized.original_schema_metadata": b"voiage-schema-metadata-v1:"
        + encoded,
    }
    path = tmp_path / "previous.parquet"
    pq.write_table(
        pa.table({"mean": [1.0]}).replace_schema_metadata(wire_metadata), path
    )
    restored = NormalizedInputBundle.read_parquet(path)
    assert restored.tables["parameters"].schema.metadata == source_metadata


def test_parquet_round_trip_preserves_multi_chunk_column_layout_and_digest(
    tmp_path: Path,
) -> None:
    manifest = DatasetManifest(
        dataset_id="multi-chunk-parameters",
        tables=(
            TableManifest(
                table_id="parameters",
                fields=(
                    FieldManifest(field_id="mean", dtype="float64", unit="kg"),
                    FieldManifest(
                        field_id="covariance_row", dtype="fixed_size_list", unit="kg^2"
                    ),
                ),
            ),
        ),
        provenance=SourceProvenance(
            provider_id="synthetic-test",
            source_uri="file:///redacted/parameter-input",
            descriptor_digest="a" * 64,
        ),
    )
    cov_type = pa.list_(pa.field("covariance_element", pa.float64()), 2)
    table = pa.table(
        {
            "mean": pa.chunked_array([[1.0, 2.0], [3.0]]),
            "covariance_row": pa.chunked_array(
                [
                    pa.array([[1.0, 0.25], [0.25, 4.0]], type=cov_type),
                    pa.array([[2.0, 0.5]], type=cov_type),
                ],
                type=cov_type,
            ),
        }
    )
    original = NormalizedInputBundle(manifest=manifest, tables={"parameters": table})
    path = tmp_path / "multi-chunk.parquet"
    original.write_parquet(path)
    restored = NormalizedInputBundle.read_parquet(path)
    assert [col.num_chunks for col in restored.tables["parameters"].columns] == [2, 2]
    assert restored.content_digest == original.content_digest


@pytest.mark.parametrize(
    "bad_chunks",
    [b"not-json", b"{}", b"[[1, 2]]", b"[[-1, 2]]", b"[[true]]"],
)
def test_parquet_rejects_invalid_stored_column_chunks(
    tmp_path: Path, bad_chunks: bytes
) -> None:
    manifest = DatasetManifest(
        dataset_id="chunk-check",
        tables=(
            TableManifest(
                table_id="parameters",
                fields=(FieldManifest(field_id="mean", dtype="float64"),),
            ),
        ),
        provenance=SourceProvenance(
            provider_id="synthetic-test",
            source_uri="file:///redacted/parameter-input",
            descriptor_digest="b" * 64,
        ),
    )
    bundle = NormalizedInputBundle(
        manifest=manifest, tables={"parameters": pa.table({"mean": [1.0]})}
    )
    path = tmp_path / "invalid-chunks.parquet"
    bundle.write_parquet(path)
    stored = pq.read_table(path)
    metadata = dict(stored.schema.metadata or {})
    metadata[b"voiage.normalized.original_column_chunks"] = bad_chunks
    pq.write_table(stored.replace_schema_metadata(metadata), path)
    with pytest.raises(ValueError, match="original Arrow column chunks"):
        NormalizedInputBundle.read_parquet(path)


def test_parquet_round_trip_accepts_legacy_bundle_without_original_chunks_key(
    tmp_path: Path,
) -> None:
    manifest = DatasetManifest(
        dataset_id="legacy-chunks",
        tables=(
            TableManifest(
                table_id="parameters",
                fields=(FieldManifest(field_id="mean", dtype="float64"),),
            ),
        ),
        provenance=SourceProvenance(
            provider_id="synthetic-test",
            source_uri="file:///redacted/parameter-input",
            descriptor_digest="b" * 64,
        ),
    )
    bundle = NormalizedInputBundle(
        manifest=manifest, tables={"parameters": pa.table({"mean": [1.0, 2.0]})}
    )
    path = tmp_path / "legacy-chunks.parquet"
    bundle.write_parquet(path)
    stored = pq.read_table(path)
    metadata = dict(stored.schema.metadata or {})
    metadata.pop(b"voiage.normalized.original_column_chunks", None)
    pq.write_table(stored.replace_schema_metadata(metadata), path)

    reloaded = NormalizedInputBundle.read_parquet(path)
    assert reloaded.content_digest == bundle.content_digest


def test_parquet_round_trip_restores_normalized_timestamp_unit(tmp_path: Path) -> None:
    import datetime as dt

    manifest = DatasetManifest(
        dataset_id="timestamp-unit",
        tables=(
            TableManifest(
                table_id="events",
                fields=(FieldManifest(field_id="at", dtype="timestamp"),),
            ),
        ),
        provenance=SourceProvenance(
            provider_id="synthetic-test",
            source_uri="file:///redacted/events",
            descriptor_digest="c" * 64,
        ),
    )
    table = pa.table(
        {"at": pa.array([dt.datetime(2026, 1, 1)], type=pa.timestamp("s"))}
    )
    bundle = NormalizedInputBundle(manifest=manifest, tables={"events": table})
    path = tmp_path / "timestamp.parquet"
    bundle.write_parquet(path)
    restored = NormalizedInputBundle.read_parquet(path)
    assert restored.tables["events"].schema.field("at").type == pa.timestamp("s")
    assert restored.content_digest == bundle.content_digest


def test_parquet_incompatible_original_type_is_rejected(tmp_path: Path) -> None:
    manifest = DatasetManifest(
        dataset_id="type-mismatch",
        tables=(
            TableManifest(
                table_id="parameters",
                fields=(FieldManifest(field_id="mean", dtype="float64"),),
            ),
        ),
        provenance=SourceProvenance(
            provider_id="synthetic-test",
            source_uri="file:///redacted/parameter-input",
            descriptor_digest="b" * 64,
        ),
    )
    bundle = NormalizedInputBundle(
        manifest=manifest, tables={"parameters": pa.table({"mean": [1.0]})}
    )
    path = tmp_path / "mismatch.parquet"
    bundle.write_parquet(path)
    stored = pq.read_table(path)
    changed = stored.cast(
        pa.schema([pa.field("mean", pa.string())]).with_metadata(stored.schema.metadata)
    )
    pq.write_table(changed, path)
    with pytest.raises(ValueError, match="does not match Parquet data"):
        NormalizedInputBundle.read_parquet(path)


def test_parquet_type_compatibility_accepts_lossless_normalizations() -> None:
    """Dictionary and narrow/wide originals stay compatible with Parquet types."""
    from voiage.contracts.normalized_input import _parquet_type_compatible

    assert _parquet_type_compatible(pa.int64(), pa.int64())
    assert _parquet_type_compatible(pa.dictionary(pa.int32(), pa.string()), pa.string())
    assert _parquet_type_compatible(
        pa.dictionary(pa.int32(), pa.string()),
        pa.dictionary(pa.int32(), pa.large_string()),
    )
    assert _parquet_type_compatible(pa.string(), pa.large_string())
    assert _parquet_type_compatible(pa.binary(), pa.large_binary())
    assert _parquet_type_compatible(pa.date32(), pa.date64())
    assert _parquet_type_compatible(pa.timestamp("s"), pa.timestamp("ns"))
    assert not _parquet_type_compatible(pa.timestamp("s", tz="UTC"), pa.timestamp("s"))
    assert not _parquet_type_compatible(pa.int64(), pa.float64())
    assert not _parquet_type_compatible(pa.string(), pa.int64())


def test_parquet_rechunk_skips_empty_source_chunks() -> None:
    """Recorded boundaries rebuild across zero-length source chunks."""
    from voiage.contracts.normalized_input import _rechunk

    column = pa.chunked_array(
        [pa.array([], type=pa.float64()), pa.array([1.0, 2.0], type=pa.float64())]
    )
    rebuilt = _rechunk(column, [2])
    assert [chunk.to_pylist() for chunk in rebuilt] == [[1.0, 2.0]]
    empty_first = _rechunk(column, [0, 2])
    assert [len(chunk) for chunk in empty_first] == [0, 2]


def test_parquet_rejects_original_schema_that_cannot_be_recast(tmp_path: Path) -> None:
    """A lossless-looking original schema that cannot be cast still fails closed."""
    import datetime as dt

    manifest = DatasetManifest(
        dataset_id="recast-failure",
        tables=(
            TableManifest(
                table_id="events",
                fields=(FieldManifest(field_id="at", dtype="timestamp"),),
            ),
        ),
        provenance=SourceProvenance(
            provider_id="synthetic-test",
            source_uri="file:///redacted/events",
            descriptor_digest="d" * 64,
        ),
    )
    table = pa.table(
        {
            "at": pa.array(
                [dt.datetime(2026, 1, 1, 0, 0, 0, 123456, tzinfo=dt.timezone.utc)],
                type=pa.timestamp("ns", tz="UTC"),
            )
        }
    )
    bundle = NormalizedInputBundle(manifest=manifest, tables={"events": table})
    path = tmp_path / "recast.parquet"
    bundle.write_parquet(path)
    stored = pq.read_table(path)
    metadata = dict(stored.schema.metadata or {})
    metadata[b"voiage.normalized.original_arrow_schema"] = (
        pa.schema([pa.field("at", pa.timestamp("s", tz="UTC"))])
        .serialize()
        .to_pybytes()
    )
    pq.write_table(stored.replace_schema_metadata(metadata), path)
    with pytest.raises(ValueError, match="does not match Parquet data"):
        NormalizedInputBundle.read_parquet(path)
