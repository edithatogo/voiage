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
