"""A conservative Frictionless Data Package CSV profile adapter."""

# pyright: reportAny=false, reportUnannotatedClassAttribute=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownVariableType=false

from __future__ import annotations

import hashlib
import json
from pathlib import Path  # noqa: TC003 - public runtime annotation
from typing import cast

import pyarrow as pa

from voiage.contracts.normalized_input import (
    DatasetManifest,
    FieldManifest,
    NormalizedInputBundle,
    SourceProvenance,
    TableManifest,
)
from voiage.ingestion._tabular import materialization_receipt, read_csv
from voiage.ingestion.base import (
    IngestionError,
    ProviderCapabilities,
    SourceAccessPolicy,
)


class FrictionlessProvider:
    """Convert an offline Data Package with exactly one CSV resource."""

    provider_id = "frictionless"
    capabilities = ProviderCapabilities(
        provider_id=provider_id,
        format_versions=("1",),
        media_types=("text/csv",),
    )

    def can_handle(self, descriptor: dict[str, object]) -> bool:
        """Recognize a Data Package from its resource list."""
        return (
            isinstance(descriptor.get("resources"), list)
            and "recordSet" not in descriptor
        )

    def ingest(
        self, descriptor_path: Path, *, policy: SourceAccessPolicy
    ) -> NormalizedInputBundle:
        """Materialize the supported explicit-schema Data Package profile."""
        raw = json.loads(descriptor_path.read_text(encoding="utf-8"))
        descriptor = cast("dict[str, object]", raw)
        resources = descriptor.get("resources")
        if (
            not isinstance(resources, list)
            or len(resources) != 1
            or not isinstance(resources[0], dict)
        ):
            raise IngestionError(
                "supported Data Package profile requires exactly one resource"
            )
        resource = cast("dict[str, object]", resources[0])
        table_id, reference, schema = (
            resource.get("name"),
            resource.get("path"),
            resource.get("schema"),
        )
        if (
            not isinstance(table_id, str)
            or not isinstance(reference, str)
            or not isinstance(schema, dict)
        ):
            raise IngestionError(
                "Data Package resource requires name, path, and schema"
            )
        schema = cast("dict[str, object]", schema)
        fields = schema.get("fields")
        if not isinstance(fields, list):
            raise IngestionError("Data Package schema requires fields")
        fields = cast("list[object]", fields)
        resource_format = resource.get("format")
        if resource_format not in (None, "csv"):
            raise IngestionError("supported Data Package profile requires CSV format")
        if "checksum" in resource:
            raise IngestionError(
                "supported Data Package profile does not support integrity declarations"
            )
        declared_sha256 = _sha256(resource.get("hash"))
        declared_byte_size = _byte_size(resource.get("bytes"))
        dialect = resource.get("dialect")
        if dialect not in (None, {"delimiter": ","}):
            raise IngestionError(
                "supported Data Package profile accepts only CSV comma dialect"
            )
        table = read_csv(
            reference,
            policy,
            sha256=declared_sha256,
            byte_size=declared_byte_size,
        )
        self._validate_schema(table, fields)
        primary_key = self._primary_key(schema)
        self._validate_primary_key(table, primary_key)
        manifest_fields = tuple(
            FieldManifest(field_id=name, dtype=str(table.schema.field(name).type))
            for item in fields
            if isinstance(item, dict)
            for name in [item.get("name")]
            if isinstance(name, str) and name in table.column_names
        )
        if tuple(field.field_id for field in manifest_fields) != tuple(
            table.column_names
        ):
            raise IngestionError(
                "Data Package fields must exactly declare the CSV columns"
            )
        digest = hashlib.sha256(descriptor_path.read_bytes()).hexdigest()
        return NormalizedInputBundle(
            manifest=DatasetManifest(
                dataset_id=str(descriptor.get("name", table_id)),
                tables=(
                    TableManifest(
                        table_id=table_id,
                        fields=manifest_fields,
                        primary_key=primary_key,
                    ),
                ),
                resources=(
                    materialization_receipt(
                        reference,
                        table_id,
                        policy,
                        sha256=declared_sha256,
                        byte_size=declared_byte_size,
                    ),
                ),
                provenance=SourceProvenance(
                    provider_id=self.provider_id,
                    source_uri=descriptor_path.resolve().as_uri(),
                    descriptor_digest=digest,
                    license=_license_label(descriptor.get("licenses")),
                    citation=_citation_label(descriptor.get("citation")),
                ),
                extensions=_governance_extensions(descriptor),
            ),
            tables={table_id: table},
        )

    @staticmethod
    def _primary_key(schema: dict[str, object]) -> tuple[str, ...]:
        """Return an explicit Data Package primary key or reject ambiguity."""
        raw = schema.get("primaryKey", ())
        if isinstance(raw, str):
            return (raw,)
        if isinstance(raw, list) and all(isinstance(field, str) for field in raw):
            return tuple(cast("str", field) for field in raw)
        if raw == ():
            return ()
        raise IngestionError("Data Package primaryKey must be a string or field list")

    @staticmethod
    def _validate_primary_key(table: pa.Table, primary_key: tuple[str, ...]) -> None:
        """Reject null or duplicate declared keys without changing source rows."""
        if not primary_key:
            return
        if not set(primary_key).issubset(table.column_names):
            raise IngestionError("Data Package primaryKey references an unknown field")
        rows = tuple(
            tuple(row[field] for field in primary_key) for row in table.to_pylist()
        )
        if any(any(value is None for value in row) for row in rows):
            raise IngestionError("Data Package primaryKey contains null values")
        if len(rows) != len(set(rows)):
            raise IngestionError("Data Package primaryKey contains duplicate values")

    @staticmethod
    def _validate_schema(table: pa.Table, fields: list[object]) -> None:
        """Validate the strict supported field names, types, and basic constraints."""
        for item in fields:
            if not isinstance(item, dict) or not isinstance(item.get("name"), str):
                raise IngestionError("Data Package fields require string names")
            field = cast("dict[str, object]", item)
            name = cast("str", field["name"])
            if name not in table.column_names:
                continue
            column = table[name]
            field_type = field.get("type")
            if field_type is not None and (
                not isinstance(field_type, str)
                or field_type
                not in {
                    "string",
                    "integer",
                    "number",
                    "boolean",
                }
            ):
                raise IngestionError("unsupported Data Package field type")
            type_matches = {
                "string": pa.types.is_string(column.type),
                "integer": pa.types.is_integer(column.type),
                "number": pa.types.is_integer(column.type)
                or pa.types.is_floating(column.type),
                "boolean": pa.types.is_boolean(column.type),
            }
            if field_type is not None and not type_matches[field_type]:
                raise IngestionError("Data Package field type does not match CSV data")
            constraints = field.get("constraints", {})
            if not isinstance(constraints, dict):
                raise IngestionError("Data Package field constraints must be an object")
            values = column.to_pylist()
            if constraints.get("required") is True and any(
                value is None for value in values
            ):
                raise IngestionError("Data Package required field contains null values")
            if constraints.get("unique") is True and len(values) != len(set(values)):
                raise IngestionError(
                    "Data Package unique field contains duplicate values"
                )


def _license_label(value: object) -> str | None:
    """Expose one human-readable licence label while preserving full metadata."""
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return None
    for item in value:
        if isinstance(item, str):
            return item
        if isinstance(item, dict):
            for key in ("name", "title", "path"):
                candidate = item.get(key)
                if isinstance(candidate, str):
                    return candidate
    return None


def _citation_label(value: object) -> str | None:
    """Accept only an explicit scalar citation for the compact provenance field."""
    return value if isinstance(value, str) else None


def _sha256(value: object) -> str | None:
    """Accept only an explicit SHA-256 Data Package resource hash."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise IngestionError("Data Package resource hash must be a SHA-256 string")
    candidate = value.lower()
    if len(candidate) != 64 or any(
        char not in "0123456789abcdef" for char in candidate
    ):
        raise IngestionError("Data Package resource hash must be a SHA-256 string")
    return candidate


def _byte_size(value: object) -> int | None:
    """Accept only a non-negative Data Package byte-size declaration."""
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise IngestionError(
            "Data Package resource bytes must be a non-negative integer"
        )
    return value


def _governance_extensions(descriptor: dict[str, object]) -> dict[str, object]:
    """Preserve standard package governance metadata without semantic inference."""
    keys = (
        "licenses",
        "sources",
        "contributors",
        "profile",
        "version",
        "title",
        "description",
    )
    return {
        f"frictionlessdata.org:{key}": descriptor[key]
        for key in keys
        if key in descriptor
    }
