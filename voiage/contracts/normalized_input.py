"""Format-neutral, immutable inputs for dataset-backed VOI analyses.

External descriptors are intentionally absent from this module.  Providers turn
their source-specific metadata and resources into these records; calculation
code consumes only this contract.
"""

from __future__ import annotations

from collections.abc import (
    Mapping,  # noqa: TC003 - Pydantic resolves runtime annotations
)
from hashlib import sha256
import json
from pathlib import Path  # noqa: TC003 - public API runtime annotation
from types import MappingProxyType
from typing import Annotated, Literal

import pyarrow as pa
from pyarrow import ipc
from pydantic import Field, StringConstraints, field_serializer, model_validator

from voiage.contracts.analysis import ContractModel, thaw_json
from voiage.contracts.interchange import schema_fingerprint

Identifier = Annotated[str, StringConstraints(min_length=1, strip_whitespace=True)]
Digest = Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{64}$")]


class FieldManifest(ContractModel):
    """One logical field in an input table."""

    field_id: Identifier
    dtype: Identifier
    unit: str | None = None
    nullable: bool = True
    description: str | None = None


class TableManifest(ContractModel):
    """The logical schema and identity of one materialized table."""

    table_id: Identifier
    fields: tuple[FieldManifest, ...]
    primary_key: tuple[Identifier, ...] = ()

    @model_validator(mode="after")
    def validate_fields(self) -> TableManifest:
        """Ensure each declared column and primary-key reference is unambiguous."""
        ids = tuple(field.field_id for field in self.fields)
        if len(ids) != len(set(ids)):
            raise ValueError("field identifiers must be unique within a table")
        if not set(self.primary_key).issubset(ids):
            raise ValueError("primary_key references an unknown field")
        return self


class SourceProvenance(ContractModel):
    """A redaction-safe identity for the descriptor used during ingestion."""

    provider_id: Identifier
    source_uri: str
    descriptor_digest: Digest
    retrieved_at: str | None = None
    license: str | None = None
    citation: str | None = None
    governance: Mapping[str, object] = Field(default_factory=dict)

    @field_serializer("governance")
    def serialize_governance(self, value: object) -> object:
        """Restore frozen metadata to JSON containers for serialization."""
        return thaw_json(value)


class VOIBinding(ContractModel):
    """An explicit mapping from source fields to a VOI runtime role."""

    role: Literal["net_benefit", "parameter", "cost", "outcome", "weight"]
    table_id: Identifier
    field_ids: tuple[Identifier, ...]
    strategy_names: tuple[Identifier, ...] = ()
    unit: str | None = None
    perspective: str | None = None

    @model_validator(mode="after")
    def validate_strategy_names(self) -> VOIBinding:
        """Require a non-empty mapping and aligned user-facing strategies."""
        if not self.field_ids:
            raise ValueError("a VOI binding must reference at least one field")
        if self.strategy_names and len(self.strategy_names) != len(self.field_ids):
            raise ValueError("strategy_names must match field_ids when supplied")
        return self


class DatasetManifest(ContractModel):
    """Versioned metadata sufficient to interpret a normalized dataset."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    dataset_id: Identifier
    tables: tuple[TableManifest, ...]
    provenance: SourceProvenance
    bindings: tuple[VOIBinding, ...] = ()
    extensions: Mapping[str, object] = Field(default_factory=dict)

    @field_serializer("extensions")
    def serialize_extensions(self, value: object) -> object:
        """Restore frozen extensions to their JSON representation."""
        return thaw_json(value)

    @model_validator(mode="after")
    def validate_references(self) -> DatasetManifest:
        """Reject duplicate table IDs and dangling binding references."""
        table_map = {table.table_id: table for table in self.tables}
        if len(table_map) != len(self.tables):
            raise ValueError("table identifiers must be unique")
        for binding in self.bindings:
            table = table_map.get(binding.table_id)
            if table is None:
                raise ValueError(f"binding references unknown table {binding.table_id!r}")
            available = {field.field_id for field in table.fields}
            unknown = set(binding.field_ids).difference(available)
            if unknown:
                raise ValueError(f"binding references unknown field(s): {sorted(unknown)}")
        return self

    def canonical_json(self) -> str:
        """Return deterministic JSON used for metadata and content identity."""
        return json.dumps(
            self.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )


class NormalizedInputBundle:
    """Immutable tables plus the manifest that makes them meaningful.

    The data frame representation is Apache Arrow because it has deterministic
    logical schemas and supports zero-copy handoff to Polars and other engines.
    """

    def __init__(self, *, manifest: DatasetManifest, tables: Mapping[str, pa.Table]) -> None:
        expected = {table.table_id for table in manifest.tables}
        actual = set(tables)
        if expected != actual:
            raise ValueError("materialized table names must exactly match the manifest")
        for table_manifest in manifest.tables:
            table = tables[table_manifest.table_id]
            fields = tuple(table.column_names)
            expected_fields = tuple(field.field_id for field in table_manifest.fields)
            if fields != expected_fields:
                raise ValueError(
                    f"table {table_manifest.table_id!r} columns must match the manifest order"
                )
        self.manifest = manifest
        self.tables: Mapping[str, pa.Table] = MappingProxyType(dict(tables))

    def table(self, table_id: str) -> pa.Table:
        """Return the immutable Arrow table selected by its stable identifier."""
        return self.tables[table_id]

    @property
    def canonical_json(self) -> str:
        """Return deterministic manifest JSON, excluding physical table bytes."""
        return self.manifest.canonical_json()

    @property
    def schema_fingerprint(self) -> str:
        """Fingerprint every table logical schema in deterministic name order."""
        payload = {
            name: schema_fingerprint(self.tables[name].schema)
            for name in sorted(self.tables)
        }
        return sha256(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()).hexdigest()

    @property
    def content_digest(self) -> str:
        """Digest manifest, schemas, and canonical Arrow IPC payloads."""
        digest = sha256(self.canonical_json.encode())
        for name in sorted(self.tables):
            sink = pa.BufferOutputStream()
            with ipc.new_stream(sink, self.tables[name].schema) as writer:
                writer.write_table(self.tables[name])
            digest.update(name.encode())
            digest.update(sink.getvalue().to_pybytes())
        return digest.hexdigest()

    def write_ipc(self, path: Path) -> None:
        """Write a deterministic single-table IPC file with its manifest metadata."""
        if len(self.tables) != 1:
            raise ValueError("IPC export currently requires exactly one table")
        table_name, table = next(iter(self.tables.items()))
        metadata = {
            **(table.schema.metadata or {}),
            b"voiage.normalized.manifest": self.canonical_json.encode(),
            b"voiage.normalized.table_id": table_name.encode(),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with ipc.new_file(path, table.replace_schema_metadata(metadata).schema) as writer:
            writer.write_table(table.replace_schema_metadata(metadata))

    @classmethod
    def read_ipc(cls, path: Path) -> NormalizedInputBundle:
        """Restore a normalized bundle previously written by :meth:`write_ipc`."""
        with ipc.open_file(path) as reader:
            table = reader.read_all()
        metadata = table.schema.metadata or {}
        try:
            manifest = DatasetManifest.model_validate_json(metadata[b"voiage.normalized.manifest"])
            table_name = metadata[b"voiage.normalized.table_id"].decode()
        except KeyError as error:
            raise ValueError("not a voiage normalized IPC file") from error
        return cls(manifest=manifest, tables={table_name: table.replace_schema_metadata(None)})
