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
from voiage.ingestion._tabular import materialization_receipt, read_csv
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
        if not isinstance(raw, dict):
            raise IngestionError("descriptor root must be a JSON object")
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
        if not isinstance(record_sets[0], dict):
            raise IngestionError(
                "supported Croissant profile requires a recordSet object"
            )
        if not isinstance(distributions[0], dict):
            raise IngestionError(
                "supported Croissant profile requires a distribution object"
            )
        record_set = cast("dict[str, object]", record_sets[0])
        distribution = cast("dict[str, object]", distributions[0])
        self._reject_unsupported_semantics(descriptor, distribution, record_set)
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
        if reference.lower().endswith((".zip", ".tar", ".gz", ".tgz", ".bz2", ".xz")):
            raise IngestionError(
                "supported Croissant profile does not support archives"
            )
        if distribution.get("transform") not in (None, []):
            raise IngestionError(
                "supported Croissant profile does not support transformations"
            )
        declared_sha256 = distribution.get("sha256")
        if declared_sha256 is not None and not isinstance(declared_sha256, str):
            raise IngestionError("declared Croissant SHA-256 must be a string")
        try:
            table = read_csv(reference, policy, sha256=declared_sha256)
        except IngestionError as error:
            if declared_sha256 is not None and "checksum" in str(error):
                raise IngestionError(
                    "declared Croissant SHA-256 does not match local content"
                ) from error
            raise
        manifest_fields = tuple(
            FieldManifest(
                field_id=name,
                dtype=str(table.schema.field(name).type),
                semantic_type=_declared_data_type(cast("dict[str, object]", item)),
            )
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
                resources=(
                    materialization_receipt(
                        reference, table_id, policy, sha256=declared_sha256
                    ),
                ),
                provenance=SourceProvenance(
                    provider_id=self.provider_id,
                    source_uri=descriptor_path.resolve().as_uri(),
                    descriptor_digest=digest,
                    license=_scalar_metadata(descriptor.get("license")),
                    citation=_scalar_metadata(descriptor.get("citation")),
                ),
                extensions=_governance_extensions(descriptor),
            ),
            tables={table_id: table},
        )

    @staticmethod
    def _reject_unsupported_semantics(
        descriptor: dict[str, object],
        distribution: dict[str, object],
        record_set: dict[str, object],
    ) -> None:
        """Reject semantics that the conservative profile cannot preserve."""
        conforms_to = descriptor.get("conformsTo")
        if (
            conforms_to is not None
            and conforms_to != "http://mlcommons.org/croissant/1.1"
        ):
            raise IngestionError(
                "supported Croissant profile requires Croissant 1.1 conformsTo"
            )
        encoding_format = distribution.get("encodingFormat")
        if encoding_format is not None and encoding_format != "text/csv":
            raise IngestionError("supported Croissant profile requires CSV media type")
        if any(key in distribution for key in ("contentChecksum", "checksum")):
            raise IngestionError(
                "supported Croissant profile does not support integrity declarations"
            )
        if any(key in record_set for key in ("key", "primaryKey")):
            raise IngestionError("supported Croissant profile does not support keys")
        if "split" in record_set:
            raise IngestionError("supported Croissant profile does not support splits")
        fields = record_set.get("field")
        if not isinstance(fields, list):
            return
        for field in fields:
            if not isinstance(field, dict):
                continue
            if "references" in field:
                raise IngestionError(
                    "supported Croissant profile does not support field references"
                )
            if "subField" in field:
                raise IngestionError(
                    "supported Croissant profile does not support nested fields"
                )
            if "source" in field:
                raise IngestionError(
                    "supported Croissant profile does not support field sources"
                )


def _scalar_metadata(value: object) -> str | None:
    """Expose scalar metadata in provenance without coercing structured values."""
    return value if isinstance(value, str) else None


def _declared_data_type(field: dict[str, object]) -> str | None:
    """Retain a declared Croissant data type without giving it VOI meaning."""
    value = field.get("dataType")
    if value is None or isinstance(value, str):
        return value
    raise IngestionError("Croissant field dataType must be a string")


def _governance_extensions(descriptor: dict[str, object]) -> dict[str, object]:
    """Retain Croissant governance metadata without inferring VOI semantics."""
    keys = (
        "@id",
        "citation",
        "creator",
        "datePublished",
        "description",
        "isAccessibleForFree",
        "keywords",
        "license",
        "odrl",
        "provenance",
        "rai",
        "sameAs",
        "usageInfo",
    )
    return (
        {
            "mlcommons.org:croissant-governance": {
                key: descriptor[key] for key in keys if key in descriptor
            }
        }
        if any(key in descriptor for key in keys)
        else {}
    )
