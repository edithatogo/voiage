"""Generic DataFrame-interchange adapter; no pandas or Polars dependency."""

from __future__ import annotations

import hashlib

import pyarrow as pa
from pyarrow.interchange import from_dataframe as arrow_from_dataframe

from voiage.contracts.normalized_input import (
    DatasetManifest,
    FieldManifest,
    NormalizedInputBundle,
    SourceProvenance,
    TableManifest,
    VOIBinding,
)


def from_dataframe(
    dataframe: object,
    *,
    dataset_id: str,
    table_id: str = "data",
    bindings: tuple[VOIBinding, ...] = (),
    allow_copy: bool = True,
) -> NormalizedInputBundle:
    """Convert any dataframe-interchange producer to the normalized contract.

    Set ``allow_copy=False`` when callers need the Arrow interchange layer to
    reject conversions that cannot preserve a zero-copy boundary.
    """
    try:
        table = arrow_from_dataframe(dataframe, allow_copy=allow_copy)
    except (TypeError, ValueError, RuntimeError, pa.ArrowException) as error:
        raise ValueError(
            "input does not satisfy the dataframe interchange protocol "
            "with the requested copy policy"
        ) from error
    descriptor_digest = _table_digest(table, dataset_id=dataset_id, table_id=table_id)
    return NormalizedInputBundle(
        manifest=DatasetManifest(
            dataset_id=dataset_id,
            tables=(
                TableManifest(
                    table_id=table_id,
                    fields=tuple(
                        FieldManifest(field_id=field.name, dtype=str(field.type))
                        for field in table.schema
                    ),
                ),
            ),
            provenance=SourceProvenance(
                provider_id="dataframe-interchange",
                source_uri="urn:voiage:dataframe-interchange",
                descriptor_digest=descriptor_digest,
            ),
            bindings=bindings,
        ),
        tables={table_id: table},
    )


def _table_digest(table: pa.Table, *, dataset_id: str, table_id: str) -> str:
    """Hash canonical Arrow IPC content for direct-input provenance."""
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    hasher = hashlib.sha256()
    hasher.update(b"voiage:dataframe-interchange:v1\0")
    hasher.update(dataset_id.encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(table_id.encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(sink.getvalue().to_pybytes())
    return hasher.hexdigest()
