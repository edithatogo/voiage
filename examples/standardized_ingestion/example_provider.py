"""Minimal third-party provider example using only VOIAGE public contracts."""

from __future__ import annotations

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


class ExampleCSVProvider:
    """Recognize a deliberately small descriptor profile owned by an application."""

    provider_id = "example-csv"
    capabilities = ProviderCapabilities(
        provider_id=provider_id,
        format_versions=("1",),
        media_types=("text/csv",),
    )

    def can_handle(self, descriptor: dict[str, object]) -> bool:
        """Recognize a descriptor without I/O or optional-parser imports."""
        return descriptor.get("voiage_example") == "1"

    def ingest(
        self, descriptor_path: Path, *, policy: SourceAccessPolicy
    ) -> NormalizedInputBundle:
        """Materialize the declared local CSV through the caller's policy."""
        import json

        descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
        resource = descriptor.get("resource")
        if not isinstance(resource, str):
            raise IngestionError("example descriptor requires a string resource")
        path = policy.resolve(resource)
        if path.suffix.lower() != ".csv":
            raise IngestionError("example provider supports CSV resources only")
        try:
            table = csv.read_csv(path)
        except Exception as error:
            raise IngestionError("example CSV resource cannot be parsed") from error
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


provider = ExampleCSVProvider()
