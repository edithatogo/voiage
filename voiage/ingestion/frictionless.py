"""A conservative Frictionless Data Package CSV profile adapter."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path  # noqa: TC003 - public runtime annotation
from typing import cast

from voiage.contracts.normalized_input import (
    DatasetManifest,
    FieldManifest,
    NormalizedInputBundle,
    SourceProvenance,
    TableManifest,
)
from voiage.ingestion._tabular import read_csv
from voiage.ingestion.base import IngestionError, SourceAccessPolicy


class FrictionlessProvider:
    """Convert an offline Data Package with exactly one CSV resource."""

    provider_id = "frictionless"

    def can_handle(self, descriptor: dict[str, object]) -> bool:
        """Recognize a Data Package from its resource list."""
        return isinstance(descriptor.get("resources"), list) and "recordSet" not in descriptor

    def ingest(self, descriptor_path: Path, *, policy: SourceAccessPolicy) -> NormalizedInputBundle:
        """Materialize the supported explicit-schema Data Package profile."""
        raw = json.loads(descriptor_path.read_text(encoding="utf-8"))
        descriptor = cast("dict[str, object]", raw)
        resources = descriptor.get("resources")
        if not isinstance(resources, list) or len(resources) != 1 or not isinstance(resources[0], dict):
            raise IngestionError("supported Data Package profile requires exactly one resource")
        resource = cast("dict[str, object]", resources[0])
        table_id, reference, schema = resource.get("name"), resource.get("path"), resource.get("schema")
        if not isinstance(table_id, str) or not isinstance(reference, str) or not isinstance(schema, dict):
            raise IngestionError("Data Package resource requires name, path, and schema")
        fields = schema.get("fields")
        if not isinstance(fields, list):
            raise IngestionError("Data Package schema requires fields")
        table = read_csv(reference, policy)
        manifest_fields = tuple(
            FieldManifest(field_id=name, dtype=str(table.schema.field(name).type))
            for item in fields
            if isinstance(item, dict)
            for name in [item.get("name")]
            if isinstance(name, str) and name in table.column_names
        )
        if tuple(field.field_id for field in manifest_fields) != tuple(table.column_names):
            raise IngestionError("Data Package fields must exactly declare the CSV columns")
        digest = hashlib.sha256(descriptor_path.read_bytes()).hexdigest()
        return NormalizedInputBundle(
            manifest=DatasetManifest(
                dataset_id=str(descriptor.get("name", table_id)),
                tables=(TableManifest(table_id=table_id, fields=manifest_fields),),
                provenance=SourceProvenance(
                    provider_id=self.provider_id,
                    source_uri=descriptor_path.resolve().as_uri(),
                    descriptor_digest=digest,
                    license=cast("str | None", descriptor.get("licenses")),
                ),
            ),
            tables={table_id: table},
        )
