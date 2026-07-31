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
    KeyReference,
    NormalizedInputBundle,
    ResourceManifest,
    SourceProvenance,
    TableManifest,
)
from voiage.ingestion._tabular import (
    materialization_receipt,
    read_csv,
    read_json_table,
)
from voiage.ingestion.base import (
    IngestionError,
    ProviderCapabilities,
    SourceAccessPolicy,
)


class FrictionlessProvider:
    """Convert an offline Data Package with explicit local tabular resources."""

    provider_id = "frictionless"
    capabilities = ProviderCapabilities(
        provider_id=provider_id,
        format_versions=("1",),
        media_types=("text/csv", "text/tab-separated-values", "application/json"),
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
        if not isinstance(raw, dict):
            raise IngestionError("descriptor root must be a JSON object")
        descriptor = cast("dict[str, object]", raw)
        resources = descriptor.get("resources")
        if not isinstance(resources, list) or not resources:
            raise IngestionError("supported Data Package profile requires resources")
        tables: dict[str, pa.Table] = {}
        table_manifests: list[TableManifest] = []
        receipts: list[ResourceManifest] = []
        foreign_keys: list[KeyReference] = []
        for raw_resource in resources:
            if not isinstance(raw_resource, dict):
                raise IngestionError("Data Package resources must be objects")
            resource = cast("dict[str, object]", raw_resource)
            table_id, table, manifest, receipt, references = self._resource(
                resource, policy
            )
            if table_id in tables:
                raise IngestionError("Data Package resource names must be unique")
            tables[table_id] = table
            table_manifests.append(manifest)
            receipts.append(receipt)
            foreign_keys.extend(references)
        self._validate_foreign_keys(tables, tuple(foreign_keys))
        digest = hashlib.sha256(descriptor_path.read_bytes()).hexdigest()
        return NormalizedInputBundle(
            manifest=DatasetManifest(
                dataset_id=str(descriptor.get("name", table_manifests[0].table_id)),
                tables=tuple(table_manifests),
                key_references=tuple(foreign_keys),
                resources=tuple(receipts),
                provenance=SourceProvenance(
                    provider_id=self.provider_id,
                    source_uri=descriptor_path.resolve().as_uri(),
                    descriptor_digest=digest,
                    license=_license_label(descriptor.get("licenses")),
                    citation=_citation_label(descriptor.get("citation")),
                ),
                extensions=_governance_extensions(descriptor),
            ),
            tables=tables,
        )

    def _resource(
        self, resource: dict[str, object], policy: SourceAccessPolicy
    ) -> tuple[
        str, pa.Table, TableManifest, ResourceManifest, tuple[KeyReference, ...]
    ]:
        """Materialize one explicit local resource and its declared relationships."""
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
        self._validate_schema_declaration(schema, fields)
        if "checksum" in resource:
            raise IngestionError(
                "supported Data Package profile does not support integrity declarations"
            )
        declared_sha256 = _sha256(resource.get("hash"))
        declared_byte_size = _byte_size(resource.get("bytes"))
        resource_format = resource.get("format")
        if resource_format == "json":
            _validate_json_table_declarations(resource)
            _validate_json_table_field_schema(fields)
            table = read_json_table(
                reference,
                policy,
                sha256=declared_sha256,
                byte_size=declared_byte_size,
            )
            media_type = "application/json"
        else:
            delimiter, suffix, media_type = _delimited_text_profile(
                reference,
                resource_format,
                resource.get("dialect"),
            )
            table = read_csv(
                reference,
                policy,
                sha256=declared_sha256,
                byte_size=declared_byte_size,
                delimiter=delimiter,
                suffix=suffix,
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
        return (
            table_id,
            table,
            TableManifest(
                table_id=table_id, fields=manifest_fields, primary_key=primary_key
            ),
            materialization_receipt(
                reference,
                table_id,
                policy,
                sha256=declared_sha256,
                byte_size=declared_byte_size,
                media_type=media_type,
            ),
            self._foreign_keys(schema, table_id),
        )

    @staticmethod
    def _foreign_keys(
        schema: dict[str, object], table_id: str
    ) -> tuple[KeyReference, ...]:
        raw = schema.get("foreignKeys", [])
        if not isinstance(raw, list):
            raise IngestionError("Data Package foreignKeys must be a list")
        references: list[KeyReference] = []
        for item in raw:
            if not isinstance(item, dict) or not isinstance(
                item.get("reference"), dict
            ):
                raise IngestionError("Data Package foreignKeys require a reference")
            foreign_key = cast("dict[str, object]", item)
            target = cast("dict[str, object]", foreign_key["reference"])
            fields, target_fields = item.get("fields"), target.get("fields")
            source_fields = (
                (fields,)
                if isinstance(fields, str)
                else tuple(cast("list[str]", fields))
                if isinstance(fields, list)
                and all(isinstance(field, str) for field in fields)
                else ()
            )
            target_ids = (
                (target_fields,)
                if isinstance(target_fields, str)
                else tuple(cast("list[str]", target_fields))
                if isinstance(target_fields, list)
                and all(isinstance(field, str) for field in target_fields)
                else ()
            )
            target_table = target.get("resource", table_id)
            if not isinstance(target_table, str) or not source_fields or not target_ids:
                raise IngestionError(
                    "Data Package foreignKeys require resource and fields"
                )
            references.append(
                KeyReference(
                    source_table_id=table_id,
                    source_field_ids=source_fields,
                    target_table_id=target_table,
                    target_field_ids=target_ids,
                )
            )
        return tuple(references)

    @staticmethod
    def _validate_foreign_keys(
        tables: dict[str, pa.Table], references: tuple[KeyReference, ...]
    ) -> None:
        for reference in references:
            source, target = (
                tables.get(reference.source_table_id),
                tables.get(reference.target_table_id),
            )
            if source is None or target is None:
                raise IngestionError(
                    "Data Package foreignKey references an unknown resource"
                )
            if not set(reference.source_field_ids).issubset(
                source.column_names
            ) or not set(reference.target_field_ids).issubset(target.column_names):
                raise IngestionError(
                    "Data Package foreignKey references an unknown field"
                )
            target_rows = {
                tuple(row[field] for field in reference.target_field_ids)
                for row in target.to_pylist()
            }
            for row in source.to_pylist():
                value = tuple(row[field] for field in reference.source_field_ids)
                if (
                    any(entry is not None for entry in value)
                    and value not in target_rows
                ):
                    raise IngestionError(
                        "Data Package foreignKey references a missing value"
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
    def _validate_schema_declaration(
        schema: dict[str, object], fields: list[object]
    ) -> None:
        """Reject schema semantics that the local profile cannot preserve.

        This check runs before materializing the resource.  Otherwise a
        descriptor could make a meaningful Table Schema claim (for example,
        application-specific missing-value tokens) which the Arrow CSV reader
        would silently interpret using different semantics.
        """
        if "missingValues" in schema:
            raise IngestionError(
                "supported Data Package profile does not support schema missingValues"
            )
        for item in fields:
            if not isinstance(item, dict):
                continue
            field = cast("dict[str, object]", item)
            constraints = field.get("constraints", {})
            if not isinstance(constraints, dict):
                raise IngestionError("Data Package field constraints must be an object")
            unsupported = set(constraints).difference({"required", "unique"})
            if unsupported:
                raise IngestionError("unsupported Data Package field constraint")
            if any(not isinstance(value, bool) for value in constraints.values()):
                raise IngestionError("Data Package field constraints must be boolean")

    @staticmethod
    def _validate_schema(table: pa.Table, fields: list[object]) -> None:
        """Validate the strict supported field names, types, and basic constraints."""
        declared_names: list[str] = []
        for item in fields:
            if not isinstance(item, dict) or not isinstance(item.get("name"), str):
                raise IngestionError("Data Package fields require string names")
            field = cast("dict[str, object]", item)
            declared_names.append(cast("str", field["name"]))
        if len(declared_names) != len(set(declared_names)):
            raise IngestionError(
                "Data Package schema has ambiguous duplicate field names"
            )
        if len(table.column_names) != len(set(table.column_names)):
            raise IngestionError(
                "Data Package CSV header has ambiguous duplicate field names"
            )
        for item in fields:
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
            constraints = cast("dict[str, bool]", field.get("constraints", {}))
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


def _delimited_text_profile(
    reference: str, resource_format: object, dialect: object
) -> tuple[str, str, str]:
    """Return one explicit local CSV or TSV profile without sniffing dialects.

    The Data Package profile deliberately supports only exact descriptor/file
    pairs.  In particular, a tab-separated resource must declare both its TSV
    format and tab delimiter, avoiding an accidental comma parse of a `.tsv`
    file or a hidden dialect feature that Arrow would interpret differently.
    """
    normalized_path = reference.casefold()
    if resource_format in (None, "csv"):
        if not normalized_path.endswith(".csv") or dialect not in (
            None,
            {"delimiter": ","},
        ):
            raise IngestionError(
                "CSV resources require a .csv path and comma delimiter"
            )
        return ",", ".csv", "text/csv"
    if resource_format != "tsv":
        raise IngestionError(
            "supported Data Package profile requires CSV or TSV format"
        )
    if not normalized_path.endswith(".tsv"):
        raise IngestionError("TSV resources require a .tsv path")
    if isinstance(dialect, dict) and set(dialect) != {"delimiter"}:
        raise IngestionError("strict CSV/TSV dialects accept only delimiter")
    if dialect != {"delimiter": "\t"}:
        raise IngestionError("TSV resources require an explicit tab delimiter")
    return "\t", ".tsv", "text/tab-separated-values"


def _validate_json_table_declarations(resource: dict[str, object]) -> None:
    """Reject JSON Table declarations the narrow local profile cannot honour."""
    reference = resource.get("path")
    if not isinstance(reference, str) or not reference.casefold().endswith(".json"):
        raise IngestionError("JSON Table resources require a .json path")
    if "dialect" in resource:
        raise IngestionError("JSON Table resources do not support dialect declarations")
    if "encoding" in resource:
        raise IngestionError(
            "JSON Table resources do not support resource encoding declarations"
        )
    unsupported = {
        "compression",
        "data",
        "jsonPath",
        "mediatype",
        "pathType",
        "transform",
    }.intersection(resource)
    if unsupported:
        raise IngestionError("JSON Table resources do not support parser declarations")


def _validate_json_table_field_schema(fields: list[object]) -> None:
    """Require the small field declaration subset the JSON profile enforces."""
    if not fields:
        raise IngestionError("JSON Table schema requires at least one field")
    allowed = {"constraints", "description", "name", "title", "type"}
    for raw_field in fields:
        if not isinstance(raw_field, dict) or not isinstance(
            raw_field.get("name"), str
        ):
            raise IngestionError("JSON Table fields require string names")
        if set(raw_field).difference(allowed):
            raise IngestionError("JSON Table fields contain unsupported declarations")


def _governance_extensions(descriptor: dict[str, object]) -> dict[str, object]:
    """Preserve standard package governance metadata without semantic inference."""
    keys = (
        "citation",
        "licenses",
        "sources",
        "contributors",
        "profile",
        "version",
        "title",
        "description",
        "usage",
        "usageInfo",
        "usageRights",
    )
    extensions = {
        f"frictionlessdata.org:{key}": descriptor[key]
        for key in keys
        if key in descriptor
    }
    extensions.update(
        {
            key: value
            for key, value in descriptor.items()
            if ":" in key and key not in extensions
        }
    )
    return extensions
