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
) -> NormalizedInputBundle:
    """Convert any dataframe-interchange producer to the normalized contract."""
    try:
        table = arrow_from_dataframe(dataframe)
    except (TypeError, ValueError, pa.ArrowException) as error:
        raise ValueError("input does not implement the dataframe interchange protocol") from error
    descriptor_digest = hashlib.sha256(
        f"dataframe-interchange:{dataset_id}:{table_id}".encode()
    ).hexdigest()
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
