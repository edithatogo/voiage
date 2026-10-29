"""Format-neutral, immutable inputs for dataset-backed VOI analyses.

External descriptors are intentionally absent from this module.  Providers turn
their source-specific metadata and resources into these records; calculation
code consumes only this contract.
"""

# pyright: reportAny=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownVariableType=false, reportUnannotatedClassAttribute=false

from __future__ import annotations

from collections.abc import (
    Mapping,  # noqa: TC003 - Pydantic resolves runtime annotations
)
from hashlib import sha256
import json
from pathlib import Path  # noqa: TC003 - public API runtime annotation
from types import MappingProxyType
from typing import Annotated, Literal, Protocol
from urllib.parse import urlsplit

import pyarrow as pa
from pyarrow import ipc
from pyarrow import parquet as pq
from pydantic import Field, StringConstraints, field_serializer, model_validator

from voiage.contracts.analysis import ContractModel, thaw_json
from voiage.contracts.interchange import schema_fingerprint

Identifier = Annotated[str, StringConstraints(min_length=1, strip_whitespace=True)]
Digest = Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{64}$")]
_MANIFEST_KEY = b"voiage.normalized.manifest"
_TABLE_ID_KEY = b"voiage.normalized.table_id"
_ORIGINAL_METADATA_KEY = b"voiage.normalized.original_schema_metadata"
_ORIGINAL_SCHEMA_KEY = b"voiage.normalized.original_arrow_schema"
_ORIGINAL_CHUNKS_KEY = b"voiage.normalized.original_column_chunks"
_METADATA_MAGIC = b"voiage-schema-metadata-v1:"
_MANIFEST_FORMAT_MARKER = b"\n "
_PARQUET_SCHEMA_FORMAT_MARKER = b"\n \n "


def _encode_schema_metadata(metadata: Mapping[bytes, bytes] | None) -> bytes:
    """Retain arbitrary original metadata, including reserved-key collisions."""
    entries = (
        None
        if metadata is None
        else [[key.hex(), value.hex()] for key, value in metadata.items()]
    )
    return _METADATA_MAGIC + json.dumps(entries, separators=(",", ":")).encode("ascii")


def _restore_schema_metadata(
    metadata: Mapping[bytes, bytes],
) -> dict[bytes, bytes] | None:
    """Restore original schema metadata, including from older bundle files."""
    encoded = metadata.get(_ORIGINAL_METADATA_KEY)
    if not metadata[_MANIFEST_KEY].endswith(_MANIFEST_FORMAT_MARKER):
        legacy = {
            key: value
            for key, value in metadata.items()
            if key not in {_MANIFEST_KEY, _TABLE_ID_KEY}
        }
        return legacy or None
    if encoded is None or not encoded.startswith(_METADATA_MAGIC):
        raise ValueError("invalid original Arrow schema metadata")
    try:
        entries = json.loads(encoded[len(_METADATA_MAGIC) :])
    except (ValueError, UnicodeDecodeError) as error:
        raise ValueError("invalid original Arrow schema metadata") from error
    if entries is None:
        return None
    if not isinstance(entries, list):
        raise ValueError("invalid original Arrow schema metadata")  # noqa: TRY004
    restored: dict[bytes, bytes] = {}
    for entry in entries:
        if (
            not isinstance(entry, list)
            or len(entry) != 2
            or not all(isinstance(part, str) for part in entry)
        ):
            raise ValueError("invalid original Arrow schema metadata")
        try:
            key, value = (bytes.fromhex(part) for part in entry)
        except ValueError as error:
            raise ValueError("invalid original Arrow schema metadata") from error
        if key in restored:
            raise ValueError("duplicate original Arrow schema metadata key")
        restored[key] = value
    return restored


def _parquet_type_compatible(original: pa.DataType, actual: pa.DataType) -> bool:
    """Return whether *actual* is a lossless Parquet normalization of *original*."""
    if original.equals(actual):
        return True
    if pa.types.is_timestamp(original) and pa.types.is_timestamp(actual):
        return original.tz == actual.tz
    if pa.types.is_dictionary(original):
        return _parquet_type_compatible(original.value_type, actual) or (
            pa.types.is_dictionary(actual)
            and _parquet_type_compatible(original.value_type, actual.value_type)
        )
    for narrow, wide in (
        (pa.string(), pa.large_string()),
        (pa.binary(), pa.large_binary()),
        (pa.date32(), pa.date64()),
    ):
        if {original, actual} == {narrow, wide}:
            return True
    return False


def _rechunk(column: pa.ChunkedArray, lengths: list[int]) -> list[pa.Array]:
    """Rebuild recorded chunk boundaries without combining the whole column."""
    sources = list(column.chunks)
    source_index = 0
    source_offset = 0
    rebuilt: list[pa.Array] = []
    for length in lengths:
        parts: list[pa.Array] = []
        remaining = length
        while remaining > 0:
            source = sources[source_index]
            take = min(remaining, len(source) - source_offset)
            if take > 0:
                parts.append(source.slice(source_offset, take))
                source_offset += take
                remaining -= take
            if source_offset >= len(source):
                source_index += 1
                source_offset = 0
        rebuilt.append(
            pa.concat_arrays(parts) if parts else pa.array([], type=column.type)
        )
    return rebuilt


def _restore_column_chunks(
    table: pa.Table, metadata: Mapping[bytes, bytes]
) -> list[pa.ChunkedArray]:
    """Restore original column chunk boundaries when recorded in metadata."""
    raw = metadata.get(_ORIGINAL_CHUNKS_KEY)
    if raw is None:
        return list(table.columns)
    try:
        layout = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as error:
        raise ValueError("invalid original Arrow column chunks") from error
    if not isinstance(layout, list) or len(layout) != len(table.columns):
        raise ValueError("invalid original Arrow column chunks")
    columns: list[pa.ChunkedArray] = []
    for column, lengths in zip(table.columns, layout, strict=True):
        if (
            not isinstance(lengths, list)
            or any(
                not isinstance(length, int) or isinstance(length, bool) or length < 0
                for length in lengths
            )
            or sum(lengths) != len(column)
            or (len(column) > 0 and not lengths)
        ):
            raise ValueError("invalid original Arrow column chunks")
        chunks = _rechunk(column, lengths)
        columns.append(pa.chunked_array(chunks, type=column.type))
    return columns


class _Digest(Protocol):
    def update(self, data: bytes, /) -> None:
        """Incorporate the next bytes in the digest."""


class _DigestWriter:
    """Forward Arrow stream bytes directly to a hash without retaining them."""

    def __init__(self, digest: _Digest) -> None:
        self._digest = digest
        self._position = 0
        self.closed = False

    def write(self, data: bytes) -> int:
        self._digest.update(data)
        self._position += len(data)
        return len(data)

    def tell(self) -> int:
        return self._position

    def writable(self) -> bool:
        return True

    def flush(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


def _is_namespaced_extension(key: str) -> bool:
    """Return whether *key* identifies an extension outside the core namespace."""
    return key.startswith(("https://", "http://")) or (
        ":" in key and "." in key.partition(":")[0]
    )


class FieldManifest(ContractModel):
    """One logical field in an input table."""

    field_id: Identifier
    dtype: Identifier
    unit: str | None = None
    nullable: bool = True
    description: str | None = None
    semantic_type: str | None = None


class ResourceManifest(ContractModel):
    """Identity and integrity evidence for one materialized source resource."""

    resource_id: Identifier
    uri: str
    sha256: Digest
    media_type: str | None = None
    byte_size: int | None = Field(default=None, ge=0)
    retrieved_at: str | None = None


class KeyReference(ContractModel):
    """An explicit foreign-key relationship between normalized tables."""

    source_table_id: Identifier
    source_field_ids: tuple[Identifier, ...]
    target_table_id: Identifier
    target_field_ids: tuple[Identifier, ...]

    @model_validator(mode="after")
    def validate_arity(self) -> KeyReference:
        """Require a non-empty, positional mapping for composite keys."""
        if not self.source_field_ids or len(self.source_field_ids) != len(
            self.target_field_ids
        ):
            raise ValueError("key references require matching non-empty field lists")
        return self


class IngestionDiagnostic(ContractModel):
    """A stable, source-neutral validation or materialization observation."""

    code: Identifier
    severity: Literal["info", "warning", "error"]
    message: str
    table_id: Identifier | None = None
    field_id: Identifier | None = None


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

    @model_validator(mode="after")
    def validate_safe_source_uri(self) -> SourceProvenance:
        """Reject source identities that can retain credentials or signed URLs."""
        parsed = urlsplit(self.source_uri)
        if parsed.username or parsed.password or parsed.query:
            raise ValueError("source_uri must not contain credentials or query strings")
        return self

    @field_serializer("governance")
    def serialize_governance(self, value: object) -> object:
        """Restore frozen metadata to JSON containers for serialization."""
        return thaw_json(value)


class VOIBinding(ContractModel):
    """An explicit mapping from source fields to a VOI runtime role."""

    role: Literal[
        "sample",
        "strategy",
        "net_benefit",
        "cost",
        "outcome",
        "parameter",
        "design_variable",
        "target",
        "weight",
        "perspective",
        "split",
    ]
    table_id: Identifier
    field_ids: tuple[Identifier, ...]
    strategy_names: tuple[Identifier, ...] = ()
    layout: Literal["wide", "long"] = "wide"
    sample_id_field_id: Identifier | None = None
    strategy_field_id: Identifier | None = None
    value_field_id: Identifier | None = None
    unit: str | None = None
    perspective: str | None = None
    transformations: tuple[str, ...] = ()
    applicable_method_families: tuple[Identifier, ...] = ()

    @model_validator(mode="after")
    def validate_strategy_names(self) -> VOIBinding:
        """Require a non-empty mapping and aligned user-facing strategies."""
        if not self.field_ids:
            raise ValueError("a VOI binding must reference at least one field")
        if (
            self.layout == "wide"
            and self.strategy_names
            and len(self.strategy_names) != len(self.field_ids)
        ):
            raise ValueError("strategy_names must match field_ids when supplied")
        long_fields = (
            self.sample_id_field_id,
            self.strategy_field_id,
            self.value_field_id,
        )
        if self.layout == "wide":
            if any(field is not None for field in long_fields):
                raise ValueError("wide bindings cannot declare long-layout fields")
            return self
        if self.role != "net_benefit":
            raise ValueError("long layout is currently supported only for net_benefit")
        if any(field is None for field in long_fields):
            raise ValueError(
                "long net_benefit bindings require sample_id, strategy, and value fields"
            )
        if self.field_ids != (self.value_field_id,):
            raise ValueError(
                "long net_benefit field_ids must contain only value_field_id"
            )
        if not self.strategy_names:
            raise ValueError(
                "long net_benefit bindings require an explicit strategy order"
            )
        return self


class BindingProfile(ContractModel):
    """Independently versioned VOI semantics for a normalized dataset."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    bindings: tuple[VOIBinding, ...]
    precedence: Literal["profile"] = "profile"

    @model_validator(mode="after")
    def validate_unique_roles(self) -> BindingProfile:
        """Reject duplicate definitions of the same semantic role."""
        roles = tuple(binding.role for binding in self.bindings)
        if len(roles) != len(set(roles)):
            raise ValueError("binding profile roles must be unique")
        return self

    def binding_for(self, role: str, *, method_family: str) -> VOIBinding:
        """Resolve one role only when the profile explicitly allows the method."""
        matches = tuple(binding for binding in self.bindings if binding.role == role)
        if len(matches) != 1:
            raise ValueError(f"exactly one {role} binding is required")
        binding = matches[0]
        if (
            binding.applicable_method_families
            and method_family not in binding.applicable_method_families
        ):
            raise ValueError(
                f"binding role {role!r} is not applicable to {method_family!r}"
            )
        return binding

    def canonical_json(self) -> str:
        """Return canonical profile JSON for reproducible semantic identity."""
        return json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    @property
    def digest(self) -> str:
        """Return the SHA-256 digest of the canonical binding profile."""
        return sha256(self.canonical_json().encode()).hexdigest()


class DatasetManifest(ContractModel):
    """Versioned metadata sufficient to interpret a normalized dataset."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    dataset_id: Identifier
    tables: tuple[TableManifest, ...]
    provenance: SourceProvenance
    resources: tuple[ResourceManifest, ...] = ()
    key_references: tuple[KeyReference, ...] = ()
    diagnostics: tuple[IngestionDiagnostic, ...] = ()
    bindings: tuple[VOIBinding, ...] = ()
    binding_profile: BindingProfile | None = None
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
        resource_ids = tuple(resource.resource_id for resource in self.resources)
        if len(resource_ids) != len(set(resource_ids)):
            raise ValueError("resource identifiers must be unique")
        for binding in self.bindings:
            table = table_map.get(binding.table_id)
            if table is None:
                raise ValueError(
                    f"binding references unknown table {binding.table_id!r}"
                )
            available = {field.field_id for field in table.fields}
            referenced_fields = set(binding.field_ids).union(
                field
                for field in (
                    binding.sample_id_field_id,
                    binding.strategy_field_id,
                    binding.value_field_id,
                )
                if field is not None
            )
            unknown = referenced_fields.difference(available)
            if unknown:
                raise ValueError(
                    f"binding references unknown field(s): {sorted(unknown)}"
                )
            declared_units = {
                field.unit
                for field in table.fields
                if field.field_id in binding.field_ids
            }
            if (
                binding.unit is not None
                and declared_units != {None}
                and binding.unit not in declared_units
            ):
                raise ValueError("binding unit is incompatible with declared fields")
        if self.binding_profile is not None:
            if self.bindings and self.bindings != self.binding_profile.bindings:
                raise ValueError("embedded bindings conflicts with binding_profile")
            for binding in self.binding_profile.bindings:
                table = table_map.get(binding.table_id)
                referenced_fields = set(binding.field_ids).union(
                    field
                    for field in (
                        binding.sample_id_field_id,
                        binding.strategy_field_id,
                        binding.value_field_id,
                    )
                    if field is not None
                )
                if table is None or not referenced_fields.issubset(
                    {field.field_id for field in table.fields}
                ):
                    raise ValueError(
                        "binding_profile references an unknown table or field"
                    )
                declared_units = {
                    field.unit
                    for field in table.fields
                    if field.field_id in binding.field_ids
                }
                if (
                    binding.unit is not None
                    and declared_units != {None}
                    and binding.unit not in declared_units
                ):
                    raise ValueError(
                        "binding_profile unit is incompatible with declared fields"
                    )
        for reference in self.key_references:
            source = table_map.get(reference.source_table_id)
            target = table_map.get(reference.target_table_id)
            if source is None or target is None:
                raise ValueError("key reference names an unknown table")
            source_fields = {field.field_id for field in source.fields}
            target_fields = {field.field_id for field in target.fields}
            if not set(reference.source_field_ids).issubset(source_fields) or not set(
                reference.target_field_ids
            ).issubset(target_fields):
                raise ValueError("key reference names an unknown field")
            if tuple(reference.target_field_ids) != target.primary_key:
                raise ValueError(
                    "key reference target fields must match its primary_key"
                )
        for key in self.extensions:
            if not _is_namespaced_extension(key):
                raise ValueError("extensions must use stable namespaced keys")
        return self

    def canonical_json(self) -> str:
        """Return deterministic JSON used for metadata and content identity."""
        return json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )


class NormalizedInputBundle:
    """Immutable tables plus the manifest that makes them meaningful.

    The data frame representation is Apache Arrow because it has deterministic
    logical schemas and supports zero-copy handoff to Polars and other engines.
    """

    def __init__(
        self, *, manifest: DatasetManifest, tables: Mapping[str, pa.Table]
    ) -> None:
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
        return sha256(
            json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
        ).hexdigest()

    @property
    def content_digest(self) -> str:
        """Digest manifest, schemas, and canonical Arrow IPC payloads."""
        digest = sha256(self.canonical_json.encode())
        for name in sorted(self.tables):
            digest.update(name.encode())
            sink = _DigestWriter(digest)
            with (
                pa.PythonFile(sink, mode="w") as output,
                ipc.new_stream(output, self.tables[name].schema) as writer,
            ):
                writer.write_table(self.tables[name])
        return digest.hexdigest()

    def write_ipc(self, path: Path) -> None:
        """Write a deterministic single-table IPC file with its manifest metadata."""
        if len(self.tables) != 1:
            raise ValueError("IPC export currently requires exactly one table")
        table_name, table = next(iter(self.tables.items()))
        metadata = {
            **(table.schema.metadata or {}),
            _MANIFEST_KEY: self.canonical_json.encode() + _MANIFEST_FORMAT_MARKER,
            _TABLE_ID_KEY: table_name.encode(),
            _ORIGINAL_METADATA_KEY: _encode_schema_metadata(table.schema.metadata),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with ipc.new_file(
            path, table.replace_schema_metadata(metadata).schema
        ) as writer:
            writer.write_table(table.replace_schema_metadata(metadata))

    def write_parquet(self, path: Path) -> None:
        """Write a deterministic single-table Parquet file with bundle metadata."""
        if len(self.tables) != 1:
            raise ValueError("Parquet export currently requires exactly one table")
        table_name, table = next(iter(self.tables.items()))
        metadata = {
            **(table.schema.metadata or {}),
            _MANIFEST_KEY: self.canonical_json.encode() + _PARQUET_SCHEMA_FORMAT_MARKER,
            _TABLE_ID_KEY: table_name.encode(),
            _ORIGINAL_METADATA_KEY: _encode_schema_metadata(table.schema.metadata),
            _ORIGINAL_SCHEMA_KEY: table.schema.remove_metadata()
            .serialize()
            .to_pybytes(),
            _ORIGINAL_CHUNKS_KEY: json.dumps(
                [[len(chunk) for chunk in column.chunks] for column in table.columns],
                separators=(",", ":"),
            ).encode("ascii"),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            table.replace_schema_metadata(metadata), path, compression="zstd"
        )

    @classmethod
    def read_ipc(cls, path: Path) -> NormalizedInputBundle:
        """Restore a normalized bundle previously written by :meth:`write_ipc`."""
        with ipc.open_file(path) as reader:
            table = reader.read_all()
        metadata = table.schema.metadata or {}
        try:
            manifest = DatasetManifest.model_validate_json(metadata[_MANIFEST_KEY])
            table_name = metadata[_TABLE_ID_KEY].decode()
        except KeyError as error:
            raise ValueError("not a voiage normalized IPC file") from error
        return cls(
            manifest=manifest,
            tables={
                table_name: table.replace_schema_metadata(
                    _restore_schema_metadata(metadata)
                )
            },
        )

    @classmethod
    def read_parquet(cls, path: Path) -> NormalizedInputBundle:
        """Restore a normalized bundle previously written by :meth:`write_parquet`."""
        table = pq.read_table(path)
        metadata = table.schema.metadata or {}
        try:
            manifest = DatasetManifest.model_validate_json(metadata[_MANIFEST_KEY])
            table_name = metadata[_TABLE_ID_KEY].decode()
        except KeyError as error:
            raise ValueError("not a voiage normalized Parquet file") from error
        restored_metadata = _restore_schema_metadata(metadata)
        if metadata[_MANIFEST_KEY].endswith(_PARQUET_SCHEMA_FORMAT_MARKER):
            try:
                original_schema = ipc.read_schema(
                    pa.BufferReader(metadata[_ORIGINAL_SCHEMA_KEY])
                )
            except (KeyError, pa.ArrowInvalid) as error:
                raise ValueError("invalid original Arrow schema") from error
            if len(original_schema) != len(table.schema) or any(
                original.name != actual.name
                or original.nullable != actual.nullable
                or not _parquet_type_compatible(original.type, actual.type)
                for original, actual in zip(original_schema, table.schema, strict=True)
            ):
                raise ValueError("original Arrow schema does not match Parquet data")
            restored_schema = (
                original_schema.remove_metadata()
                if restored_metadata is None
                else original_schema.with_metadata(restored_metadata)
            )
            try:
                table = table.cast(original_schema.remove_metadata(), safe=True)
            except (pa.ArrowInvalid, pa.ArrowNotImplementedError) as error:
                raise ValueError(
                    "original Arrow schema does not match Parquet data"
                ) from error
            columns = _restore_column_chunks(table, metadata)
            table = pa.Table.from_arrays(columns, schema=restored_schema)
        else:
            table = table.replace_schema_metadata(restored_metadata)
        return cls(manifest=manifest, tables={table_name: table})
