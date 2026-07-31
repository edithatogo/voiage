"""A conservative Croissant 1.1 CSV profile adapter."""

from __future__ import annotations

from dataclasses import dataclass
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
    SourceSelection,
)


class CroissantProvider:
    """Convert an offline Croissant descriptor with one CSV RecordSet."""

    provider_id = "croissant"
    capabilities = ProviderCapabilities(
        provider_id=provider_id,
        format_versions=("1.1",),
        media_types=("text/csv",),
        source_selection_keys=("record_set", "distribution"),
    )

    def can_handle(self, descriptor: dict[str, object]) -> bool:
        """Recognize the Croissant context rather than a filename convention."""
        return any(
            "mlcommons.org/croissant" in context
            for context in _context_entries(descriptor.get("@context"))
        )

    def ingest(
        self,
        descriptor_path: Path,
        *,
        policy: SourceAccessPolicy,
        selection: CroissantSelection | SourceSelection | None = None,
    ) -> NormalizedInputBundle:
        """Materialize one explicitly selected local Croissant CSV pair.

        Legacy one-record-set/one-distribution descriptors need no selector.
        A descriptor with more than one local pair must instead identify both a
        ``recordSet`` name and a distribution ``@id``.  The selected record
        set must explicitly link to that distribution through the narrowly
        supported local-profile ``distribution`` member; VOIAGE never guesses
        a relationship from matching field names.
        """
        raw = json.loads(descriptor_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise IngestionError("descriptor root must be a JSON object")
        descriptor = cast("dict[str, object]", raw)
        if not _has_croissant_1_1_context(descriptor.get("@context")):
            raise IngestionError("supported Croissant profile requires version 1.1")
        if not _has_only_string_context_entries(descriptor.get("@context")):
            raise IngestionError(
                "supported Croissant profile requires string JSON-LD context entries"
            )
        record_sets = descriptor.get("recordSet")
        distributions = descriptor.get("distribution")
        if isinstance(record_sets, list):
            for candidate in record_sets:
                if isinstance(candidate, dict):
                    self._reject_unsupported_record_set(
                        cast("dict[str, object]", candidate)
                    )
        if isinstance(distributions, list):
            for candidate in distributions:
                if isinstance(candidate, dict):
                    self._reject_unsupported_distribution(
                        cast("dict[str, object]", candidate)
                    )
        record_set, distribution = _select_local_pair(
            record_sets,
            distributions,
            selection=_croissant_selection(selection),
        )
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
        # Reject a URI, archive, or traversal reference before the tabular
        # helper can invoke a materializer supplied by an application.
        policy.source_uri(reference)
        declared_sha256 = _sha256(distribution.get("sha256"))
        declared_byte_size = _content_size(distribution.get("contentSize"))
        try:
            table = read_csv(
                reference,
                policy,
                sha256=declared_sha256,
                byte_size=declared_byte_size,
            )
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
                    license=_scalar_metadata(descriptor.get("license")),
                    citation=_scalar_metadata(descriptor.get("citation")),
                ),
                extensions=_extensions(descriptor, record_set, distribution),
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
        CroissantProvider._reject_unsupported_distribution(distribution)
        CroissantProvider._reject_unsupported_record_set(record_set)

    @staticmethod
    def _reject_unsupported_distribution(distribution: dict[str, object]) -> None:
        """Reject every unsupported distribution before source access."""
        distribution_type = distribution.get("@type")
        if distribution_type in {"FileObject", "FileSet"}:
            raise IngestionError(
                "supported Croissant profile does not support FileObject or FileSet distributions"
            )
        encoding_format = distribution.get("encodingFormat")
        if encoding_format is not None and encoding_format != "text/csv":
            raise IngestionError("supported Croissant profile requires CSV media type")
        if any(key in distribution for key in ("contentChecksum", "checksum")):
            raise IngestionError(
                "supported Croissant profile does not support integrity declarations"
            )

    @staticmethod
    def _reject_unsupported_record_set(record_set: dict[str, object]) -> None:
        """Reject record-set semantics outside the local CSV profile."""
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


@dataclass(frozen=True)
class CroissantSelection:
    """Explicit local-profile selectors for a multi-pair Croissant descriptor.

    This is an ingestion-source choice, not Arrow projection or row filtering.
    Both identifiers are required whenever the descriptor contains multiple
    record sets or distributions.
    """

    record_set: str | None = None
    distribution: str | None = None


def _croissant_selection(
    selection: CroissantSelection | SourceSelection | None,
) -> CroissantSelection | None:
    """Adapt the provider-neutral request without accepting unknown selectors."""
    if selection is None or isinstance(selection, CroissantSelection):
        return selection
    if selection.provider_id != CroissantProvider.provider_id:
        raise IngestionError("source selection does not match the descriptor provider")
    if {key for key, _ in selection.values} != {"record_set", "distribution"}:
        raise IngestionError(
            "Croissant source selection requires record_set and distribution"
        )
    return CroissantSelection(
        record_set=selection.value_for("record_set"),
        distribution=selection.value_for("distribution"),
    )


def _select_local_pair(
    record_sets: object,
    distributions: object,
    *,
    selection: CroissantSelection | None,
) -> tuple[dict[str, object], dict[str, object]]:
    """Select one proven record-set/distribution relationship without inference."""
    if not isinstance(record_sets, list) or not record_sets:
        raise IngestionError("supported Croissant profile requires recordSet entries")
    if not isinstance(distributions, list) or not distributions:
        raise IngestionError(
            "supported Croissant profile requires distribution entries"
        )
    if not all(isinstance(item, dict) for item in record_sets):
        raise IngestionError("supported Croissant profile requires recordSet objects")
    if not all(isinstance(item, dict) for item in distributions):
        raise IngestionError(
            "supported Croissant profile requires distribution objects"
        )

    multi_pair = len(record_sets) > 1 or len(distributions) > 1
    requested_record_set = selection.record_set if selection is not None else None
    requested_distribution = selection.distribution if selection is not None else None
    if multi_pair and requested_record_set is None:
        raise IngestionError(
            "Croissant recordSet selection is required for multiple local entries"
        )
    if multi_pair and requested_distribution is None:
        raise IngestionError(
            "Croissant distribution selection is required for multiple local entries"
        )

    typed_record_sets = tuple(cast("dict[str, object]", item) for item in record_sets)
    typed_distributions = tuple(
        cast("dict[str, object]", item) for item in distributions
    )
    record_set = _select_by_identifier(
        typed_record_sets,
        identifier=requested_record_set,
        member="name",
        label="recordSet",
    )
    distribution = _select_by_identifier(
        typed_distributions,
        identifier=requested_distribution,
        member="@id",
        label="distribution",
    )
    if multi_pair:
        selected_distribution_id = distribution.get("@id")
        if not isinstance(selected_distribution_id, str):
            raise IngestionError(
                "selected Croissant distribution requires a string @id"
            )
        if record_set.get("distribution") != selected_distribution_id:
            raise IngestionError(
                "selected Croissant recordSet must explicitly reference the selected distribution"
            )
    return record_set, distribution


def _select_by_identifier(
    entries: tuple[dict[str, object], ...],
    *,
    identifier: str | None,
    member: str,
    label: str,
) -> dict[str, object]:
    """Return a unique named entry or produce a stable selector diagnostic."""
    if identifier is None:
        if len(entries) == 1:
            return entries[0]
        raise IngestionError(f"Croissant {label} selection is required")
    matching = tuple(entry for entry in entries if entry.get(member) == identifier)
    if not matching:
        raise IngestionError(f"selected Croissant {label} is not declared")
    if len(matching) != 1:
        raise IngestionError(f"Croissant {label} identifiers must be unique")
    return matching[0]


def _scalar_metadata(value: object) -> str | None:
    """Expose scalar metadata in provenance without coercing structured values."""
    return value if isinstance(value, str) else None


def _sha256(value: object) -> str | None:
    """Accept only an explicit SHA-256 FileObject declaration."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise IngestionError("Croissant sha256 must be a SHA-256 string")
    candidate = value.lower()
    if len(candidate) != 64 or any(
        char not in "0123456789abcdef" for char in candidate
    ):
        raise IngestionError("Croissant sha256 must be a SHA-256 string")
    return candidate


def _content_size(value: object) -> int | None:
    """Accept the profile's non-negative integer Croissant contentSize only.

    The generic Croissant field is narrowed to a byte count. Unit-bearing,
    textual, and structured values are outside this local CSV profile,
    so they fail before source materialization rather than being ignored.
    """
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise IngestionError(
            "supported Croissant profile requires contentSize to be a non-negative integer"
        )
    return value


def _declared_data_type(field: dict[str, object]) -> str | None:
    """Retain a declared Croissant data type without giving it VOI meaning."""
    value = field.get("dataType")
    if value is None or isinstance(value, str):
        return value
    raise IngestionError("Croissant field dataType must be a string")


def _context_entries(value: object) -> tuple[str, ...]:
    """Return string entries from a JSON-LD context without coercion."""
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list):
        return tuple(entry for entry in value if isinstance(entry, str))
    return ()


def _has_croissant_1_1_context(value: object) -> bool:
    """Recognize the exact supported context among JSON-LD context entries."""
    return any(
        context.rstrip("/")
        in {
            "mlcommons.org/croissant/1.1",
            "http://mlcommons.org/croissant/1.1",
            "https://mlcommons.org/croissant/1.1",
        }
        for context in _context_entries(value)
    )


def _has_only_string_context_entries(value: object) -> bool:
    """Keep the offline profile free of unexpanded JSON-LD context objects."""
    return isinstance(value, str) or (
        isinstance(value, list) and all(isinstance(entry, str) for entry in value)
    )


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


def _extensions(
    descriptor: dict[str, object],
    record_set: dict[str, object],
    distribution: dict[str, object],
) -> dict[str, object]:
    """Retain governance and the explicit source-pair identity separately."""
    extensions = _governance_extensions(descriptor)
    record_set_id = record_set.get("name")
    distribution_id = distribution.get("@id")
    if isinstance(record_set_id, str) and isinstance(distribution_id, str):
        extensions["mlcommons.org:croissant-selection"] = {
            "distribution": distribution_id,
            "recordSet": record_set_id,
        }
    return extensions
