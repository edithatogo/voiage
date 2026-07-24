"""A conservative Croissant 1.1 CSV profile adapter."""

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
from voiage.ingestion.base import (
    IngestionError,
    ProviderCapabilities,
    SourceAccessPolicy,
)


class CroissantProvider:
    """Convert an offline Croissant descriptor with one CSV RecordSet."""

    provider_id = "croissant"
    capabilities = ProviderCapabilities(
        provider_id=provider_id,
        format_versions=("1.1",),
        media_types=("text/csv",),
    )

    def can_handle(self, descriptor: dict[str, object]) -> bool:
        """Recognize the Croissant context rather than a filename convention."""
        context = descriptor.get("@context")
        return isinstance(context, str) and "mlcommons.org/croissant" in context

    def ingest(
        self, descriptor_path: Path, *, policy: SourceAccessPolicy
    ) -> NormalizedInputBundle:
        """Materialize the supported, unambiguous one-resource Croissant profile."""
        raw = json.loads(descriptor_path.read_text(encoding="utf-8"))
        descriptor = cast("dict[str, object]", raw)
        context = descriptor.get("@context")
        if not (isinstance(context, str) and "mlcommons.org/croissant/1.1" in context):
            raise IngestionError("supported Croissant profile requires version 1.1")
        record_sets = descriptor.get("recordSet")
        distributions = descriptor.get("distribution")
        if not isinstance(record_sets, list) or len(record_sets) != 1:
            raise IngestionError(
                "supported Croissant profile requires exactly one recordSet"
            )
        if not isinstance(distributions, list) or len(distributions) != 1:
            raise IngestionError(
                "supported Croissant profile requires exactly one distribution"
            )
        record_set = cast("dict[str, object]", record_sets[0])
        distribution = cast("dict[str, object]", distributions[0])
        table_id = record_set.get("name")
        reference = distribution.get("contentUrl")
        fields = record_set.get("field")
        if (
            not isinstance(table_id, str)
            or not isinstance(reference, str)
            or not isinstance(fields, list)
        ):
            raise IngestionError(
                "Croissant recordSet requires name, field, and distribution contentUrl"
            )
        table = read_csv(reference, policy)
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
                "Croissant fields must exactly declare the CSV columns"
            )
        digest = hashlib.sha256(descriptor_path.read_bytes()).hexdigest()
        return NormalizedInputBundle(
            manifest=DatasetManifest(
                dataset_id=str(descriptor.get("name", table_id)),
                tables=(TableManifest(table_id=table_id, fields=manifest_fields),),
                provenance=SourceProvenance(
                    provider_id=self.provider_id,
                    source_uri=descriptor_path.resolve().as_uri(),
                    descriptor_digest=digest,
                    license=cast("str | None", descriptor.get("license")),
                    citation=cast("str | None", descriptor.get("citation")),
                ),
            ),
            tables={table_id: table},
        )
