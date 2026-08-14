"""A disposable external package exercising only the public provider SDK."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from pyarrow import csv

from voiage.contracts import (
    DatasetManifest,
    FieldManifest,
    NormalizedInputBundle,
    SourceProvenance,
    TableManifest,
)
from voiage.ingestion import IngestionError, ProviderCapabilities, SourceAccessPolicy

if TYPE_CHECKING:
    from pathlib import Path


class WheelConsumerProvider:
    """Recognize a small local descriptor profile owned by this external package."""

    provider_id = "example-wheel-provider"
    capabilities = ProviderCapabilities(
        provider_id=provider_id,
        format_versions=("1",),
        media_types=("text/csv",),
    )

    def can_handle(self, descriptor: dict[str, object]) -> bool:
        """Recognize a descriptor without reading its declared resource."""
        return descriptor.get("voiage_wheel_example") == "1"

    def ingest(
        self, descriptor_path: Path, *, policy: SourceAccessPolicy
    ) -> NormalizedInputBundle:
        """Read a small local CSV through the caller-owned source policy."""
        descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
        resource = descriptor.get("resource")
        if not isinstance(resource, str):
            raise IngestionError("wheel descriptor requires a string resource")
        source = policy.resolve(resource)
        if source.suffix.lower() != ".csv":
            raise IngestionError("wheel provider supports CSV resources only")
        try:
            table = csv.read_csv(source)
        except Exception as error:
            raise IngestionError("wheel CSV resource cannot be parsed") from error
        return NormalizedInputBundle(
            manifest=DatasetManifest(
                dataset_id=descriptor_path.stem,
                tables=(
                    TableManifest(
                        table_id="data",
                        fields=tuple(
                            FieldManifest(field_id=field.name, dtype=str(field.type))
                            for field in table.schema
                        ),
                    ),
                ),
                provenance=SourceProvenance(
                    provider_id=self.provider_id,
                    source_uri=descriptor_path.resolve().as_uri(),
                    descriptor_digest="0" * 64,
                ),
            ),
            tables={"data": table},
        )


provider = WheelConsumerProvider()
