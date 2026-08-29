"""Small CLI surface for inspecting and normalizing supported descriptors."""

# Typer builds command declarations from defaults.
# ruff: noqa: B008

from __future__ import annotations

import json
from pathlib import Path  # noqa: TC003 - public CLI annotation
from typing import cast

import typer

from voiage.contracts.normalized_input import (
    BindingProfile,
    DatasetManifest,
    NormalizedInputBundle,
    VOIBinding,
)
from voiage.contracts.preparation import prepare_analysis_inputs
from voiage.ingestion.base import IngestionError, SourceAccessPolicy, SourceSelection
from voiage.ingestion.registry import ProviderRegistry, default_registry
from voiage.methods.basic import evpi

app = typer.Typer(help="Validate and normalize standardized dataset descriptors.")

_EXIT_INGESTION = 3
_EXIT_BINDING = 4
_EXIT_OUTPUT = 5


def _assert_provider(bundle: NormalizedInputBundle, expected: str | None) -> None:
    """Require a caller-selected provider without guessing the source format."""
    if expected is not None and bundle.manifest.provenance.provider_id != expected:
        raise IngestionError("descriptor does not match explicitly selected provider")


def _binding_profile(path: Path) -> BindingProfile:
    """Load one caller-selected binding profile without descriptor inference."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("binding profile is not valid UTF-8 JSON") from error
    if isinstance(payload, dict) and isinstance(payload.get("bindings"), list):
        bindings: list[object] = []
        for candidate in payload["bindings"]:
            if isinstance(candidate, dict):
                binding = dict(candidate)
                for field in (
                    "applicable_method_families",
                    "field_ids",
                    "strategy_names",
                    "transformations",
                ):
                    if isinstance(binding.get(field), list):
                        binding[field] = tuple(binding[field])
                bindings.append(binding)
            else:
                bindings.append(candidate)
        payload = {**payload, "bindings": tuple(bindings)}
    return BindingProfile.model_validate(payload)


def _prepared_summary(bundle: NormalizedInputBundle) -> dict[str, object]:
    """Expose explicit binding resolution and data quality without inference."""
    prepared = prepare_analysis_inputs(bundle)
    quality = prepared.quality_report
    return {
        "binding": prepared.binding.model_dump(mode="json"),
        "binding_profile_digest": prepared.binding_profile_digest,
        "data_quality": {
            "coercions": list(quality.coercions),
            "duplicate_row_count": quality.duplicate_row_count,
            "exclusions": list(quality.exclusions),
            "join_coverage": dict(quality.join_coverage),
            "null_counts": dict(quality.null_counts),
            "population_transforms": list(quality.population_transforms),
            "primary_key_duplicate_count": quality.primary_key_duplicate_count,
            "primary_key_fields": list(quality.primary_key_fields),
            "primary_key_null_count": quality.primary_key_null_count,
            "row_count": quality.row_count,
            "selected_field_ids": list(quality.selected_field_ids),
            "selected_partitions": list(quality.selected_partitions),
            "table_id": quality.table_id,
            "unique_value_counts": dict(quality.unique_value_counts),
        },
        "input_digest": prepared.input_digest,
    }


def _source_policy(
    descriptor: Path,
    *,
    source_root: Path | None,
    offline: bool,
    cache_dir: Path | None,
    max_resource_bytes: int,
    max_resource_rows: int,
) -> SourceAccessPolicy:
    """Build an explicit, local-only policy for one CLI invocation."""
    return SourceAccessPolicy(
        source_root or descriptor.parent,
        offline=offline,
        cache_dir=cache_dir,
        max_resource_bytes=max_resource_bytes,
        max_resource_rows=max_resource_rows,
    )


def _source_selection(
    *, record_set: str | None, distribution: str | None
) -> SourceSelection | None:
    """Build the explicit Croissant local-pair request used by the CLI."""
    if record_set is None and distribution is None:
        return None
    if record_set is None or distribution is None:
        raise IngestionError(
            "Croissant source selection requires both --record-set and --distribution"
        )
    return SourceSelection(
        provider_id="croissant",
        values=(("record_set", record_set), ("distribution", distribution)),
    )


def _ingest_bundle(
    descriptor: Path,
    *,
    policy: SourceAccessPolicy,
    expected_provider: str | None,
    selection: SourceSelection | None,
) -> tuple[NormalizedInputBundle, ProviderRegistry]:
    """Materialize one explicit source exactly once for a CLI command."""
    registry = default_registry()
    bundle = registry.ingest(descriptor, policy=policy, selection=selection)
    _assert_provider(bundle, expected_provider)
    return bundle, registry


def _bundle_summary(
    bundle: NormalizedInputBundle,
    registry: ProviderRegistry,
    *,
    binding: VOIBinding | None = None,
) -> dict[str, object]:
    """Return stable, non-secret metadata for a descriptor."""
    capabilities = registry.capabilities_for(bundle.manifest.provenance.provider_id)
    summary: dict[str, object] = {
        "capabilities": {
            "format_versions": capabilities.format_versions,
            "media_types": capabilities.media_types,
            "provider_id": capabilities.provider_id,
            "supported_transforms": capabilities.supported_transforms,
            "supports_filtering": capabilities.supports_filtering,
            "supports_projection": capabilities.supports_projection,
            "supports_random_access": capabilities.supports_random_access,
            "supports_streaming": capabilities.supports_streaming,
        },
        "content_digest": bundle.content_digest,
        "dataset_id": bundle.manifest.dataset_id,
        "diagnostics": [
            diagnostic.model_dump(mode="json")
            for diagnostic in bundle.manifest.diagnostics
        ],
        "governance": bundle.manifest.model_dump(mode="json")["extensions"],
        "provider": bundle.manifest.provenance.provider_id,
        "provenance": bundle.manifest.provenance.model_dump(mode="json"),
        "resources": [
            {
                "byte_size": resource.byte_size,
                "media_type": resource.media_type,
                "resource_id": resource.resource_id,
                "sha256": resource.sha256,
                "uri": resource.uri,
            }
            for resource in bundle.manifest.resources
        ],
        "schema_fingerprint": bundle.schema_fingerprint,
        "tables": {name: table.num_rows for name, table in bundle.tables.items()},
    }
    if binding is not None:
        manifest = DatasetManifest(
            **bundle.manifest.model_dump(mode="python", exclude={"bindings"}),
            bindings=(binding,),
        )
        summary["binding_resolution"] = _prepared_summary(
            NormalizedInputBundle(manifest=manifest, tables=bundle.tables)
        )
    else:
        summary["binding_resolution"] = None
    return summary


def _inspection_summary(
    descriptor: Path, *, expected_provider: str | None = None
) -> dict[str, object]:
    """Return descriptor-only diagnostics without resolving any resources.

    Binding resolution, provenance receipts, and data-quality reports require
    materializing resources. Those details deliberately belong to the
    materializing validation, normalization, and calculation commands rather
    than the metadata-only ``inspect`` command.
    """
    inspection = default_registry().inspect(descriptor)
    if expected_provider is not None and inspection["provider_id"] != expected_provider:
        raise IngestionError("descriptor does not match explicitly selected provider")
    capabilities = cast("dict[str, object]", inspection["capabilities"])
    raw = json.loads(descriptor.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise IngestionError("descriptor root must be an object")
    return {
        "binding_resolution": None,
        "capabilities": {
            "provider_id": inspection["provider_id"],
            **capabilities,
        },
        "descriptor": inspection["descriptor"],
        "provider": inspection["provider_id"],
        "schema": _descriptor_schema_summary(
            raw, provider_id=cast("str", inspection["provider_id"])
        ),
    }


def _descriptor_schema_summary(
    descriptor: dict[str, object], *, provider_id: str
) -> dict[str, object]:
    """Project declared tables without opening any declared data resource.

    This is intentionally descriptive rather than a second provider validator:
    it makes only table/field/key declarations and already-known unsupported
    metadata visible to ``inspect``. Materializing commands remain responsible
    for source access, exact schema validation, receipts, and binding quality.
    """
    if provider_id == "frictionless":
        return _frictionless_descriptor_schema_summary(descriptor)
    if provider_id == "croissant":
        return _croissant_descriptor_schema_summary(descriptor)
    return {"tables": [], "unsupported_features": []}


def _frictionless_descriptor_schema_summary(
    descriptor: dict[str, object],
) -> dict[str, object]:
    """Describe Data Package table declarations without materialization."""
    tables: list[dict[str, object]] = []
    unsupported: list[dict[str, str]] = []
    resources = descriptor.get("resources")
    if not isinstance(resources, list):
        return {"tables": tables, "unsupported_features": unsupported}
    supported_formats = {None, "csv", "tsv", "json", "parquet", "arrow", "feather"}
    for index, raw_resource in enumerate(resources):
        if not isinstance(raw_resource, dict):
            unsupported.append(
                {"code": "resource-not-object", "path": f"resources[{index}]"}
            )
            continue
        resource = cast("dict[str, object]", raw_resource)
        schema = resource.get("schema")
        schema = schema if isinstance(schema, dict) else {}
        fields = schema.get("fields")
        field_ids = (
            [
                cast("str", field.get("name"))
                for field in fields
                if isinstance(field, dict) and isinstance(field.get("name"), str)
            ]
            if isinstance(fields, list)
            else []
        )
        primary_key = schema.get("primaryKey", [])
        primary_key = [primary_key] if isinstance(primary_key, str) else primary_key
        tables.append(
            {
                "table_id": resource.get("name")
                if isinstance(resource.get("name"), str)
                else None,
                "field_ids": field_ids,
                "primary_key": primary_key
                if isinstance(primary_key, list)
                and all(isinstance(item, str) for item in primary_key)
                else [],
                "foreign_keys": schema.get("foreignKeys", [])
                if isinstance(schema.get("foreignKeys", []), list)
                else [],
            }
        )
        unsupported.extend(
            {"code": f"resource-{key}", "path": f"resources[{index}].{key}"}
            for key in ("dialect", "transform")
            if key in resource
        )
        if resource.get("format") not in supported_formats:
            unsupported.append(
                {"code": "resource-format", "path": f"resources[{index}].format"}
            )
        if "missingValues" in schema:
            unsupported.append(
                {
                    "code": "schema-missing-values",
                    "path": f"resources[{index}].schema.missingValues",
                }
            )
    return {"tables": tables, "unsupported_features": unsupported}


def _croissant_descriptor_schema_summary(
    descriptor: dict[str, object],
) -> dict[str, object]:
    """Describe Croissant record sets without resolving distributions."""
    tables: list[dict[str, object]] = []
    unsupported: list[dict[str, str]] = []
    record_sets = descriptor.get("recordSet")
    if not isinstance(record_sets, list):
        return {"tables": tables, "unsupported_features": unsupported}
    for index, raw_record_set in enumerate(record_sets):
        if not isinstance(raw_record_set, dict):
            unsupported.append(
                {"code": "record-set-not-object", "path": f"recordSet[{index}]"}
            )
            continue
        record_set = cast("dict[str, object]", raw_record_set)
        fields = record_set.get("field")
        field_ids = (
            [
                cast("str", field.get("name"))
                for field in fields
                if isinstance(field, dict) and isinstance(field.get("name"), str)
            ]
            if isinstance(fields, list)
            else []
        )
        tables.append(
            {
                "table_id": record_set.get("name")
                if isinstance(record_set.get("name"), str)
                else None,
                "field_ids": field_ids,
                "primary_key": [],
                "foreign_keys": [],
            }
        )
        unsupported.extend(
            {"code": f"record-set-{key}", "path": f"recordSet[{index}].{key}"}
            for key in ("key", "primaryKey", "split")
            if key in record_set
        )
        if isinstance(fields, list):
            for field_index, field in enumerate(fields):
                if isinstance(field, dict):
                    unsupported.extend(
                        {
                            "code": f"field-{key}",
                            "path": f"recordSet[{index}].field[{field_index}].{key}",
                        }
                        for key in ("references", "subField", "source")
                        if key in field
                    )
    return {"tables": tables, "unsupported_features": unsupported}


def _calculation_manifest(
    bundle: NormalizedInputBundle,
    *,
    table: str | None,
    field: list[str],
    strategy: list[str],
    binding_profile: Path | None,
) -> DatasetManifest:
    """Resolve exactly one explicit calculation binding contract."""
    if binding_profile is not None:
        if table is not None or field or strategy:
            raise ValueError(
                "--binding-profile cannot be combined with inline binding options"
            )
        return DatasetManifest(
            **bundle.manifest.model_dump(
                mode="python", exclude={"bindings", "binding_profile"}
            ),
            binding_profile=_binding_profile(binding_profile),
        )
    if table is None or not field:
        raise ValueError(
            "calculation requires --binding-profile or both --table and --field"
        )
    binding = VOIBinding(
        role="net_benefit",
        table_id=table,
        field_ids=tuple(field),
        strategy_names=tuple(strategy),
    )
    return DatasetManifest(
        **bundle.manifest.model_dump(mode="python", exclude={"bindings"}),
        bindings=(binding,),
    )


@app.command("validate")
def validate(
    descriptor: Path = typer.Argument(..., exists=True, readable=True),
    source_root: Path | None = typer.Option(
        None,
        "--source-root",
        file_okay=False,
        exists=True,
        readable=True,
        help="Explicit local root allowed for declared resources.",
    ),
    offline: bool = typer.Option(False, "--offline", help="Require an offline replay."),
    cache_dir: Path | None = typer.Option(
        None, "--cache-dir", help="Verified materialization cache directory."
    ),
    max_resource_bytes: int = typer.Option(
        512 * 1024 * 1024,
        "--max-resource-bytes",
        min=1,
        help="Maximum accepted local resource size.",
    ),
    max_resource_rows: int = typer.Option(
        10_000_000,
        "--max-resource-rows",
        min=1,
        help="Maximum accepted parsed rows in each local resource.",
    ),
    provider: str | None = typer.Option(
        None,
        "--provider",
        help="Require this registered provider ID for the descriptor.",
    ),
    record_set: str | None = typer.Option(
        None, "--record-set", help="Explicit Croissant recordSet name."
    ),
    distribution: str | None = typer.Option(
        None, "--distribution", help="Explicit Croissant distribution @id."
    ),
) -> None:
    """Validate a supported descriptor and its declared local resources."""
    try:
        bundle, registry = _ingest_bundle(
            descriptor,
            policy=_source_policy(
                descriptor,
                source_root=source_root,
                offline=offline,
                cache_dir=cache_dir,
                max_resource_bytes=max_resource_bytes,
                max_resource_rows=max_resource_rows,
            ),
            expected_provider=provider,
            selection=_source_selection(
                record_set=record_set, distribution=distribution
            ),
        )
        typer.echo(
            json.dumps(
                {
                    "valid": True,
                    **_bundle_summary(bundle, registry),
                },
                sort_keys=True,
            )
        )
    except IngestionError as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(_EXIT_INGESTION) from error


@app.command()
def inspect(
    descriptor: Path = typer.Argument(..., exists=True, readable=True),
    provider: str | None = typer.Option(
        None,
        "--provider",
        help="Require this registered provider ID for the descriptor.",
    ),
) -> None:
    """Inspect descriptor identity and provider capabilities without loading data."""
    try:
        typer.echo(
            json.dumps(
                _inspection_summary(descriptor, expected_provider=provider),
                sort_keys=True,
            )
        )
    except IngestionError as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(_EXIT_INGESTION) from error


@app.command()
def normalize(
    descriptor: Path = typer.Argument(..., exists=True, readable=True),
    output: Path = typer.Option(..., "--output", "-o"),
    source_root: Path | None = typer.Option(
        None,
        "--source-root",
        file_okay=False,
        exists=True,
        readable=True,
        help="Explicit local root allowed for declared resources.",
    ),
    offline: bool = typer.Option(False, "--offline", help="Require an offline replay."),
    cache_dir: Path | None = typer.Option(
        None, "--cache-dir", help="Verified materialization cache directory."
    ),
    max_resource_bytes: int = typer.Option(
        512 * 1024 * 1024, "--max-resource-bytes", min=1
    ),
    max_resource_rows: int = typer.Option(10_000_000, "--max-resource-rows", min=1),
    provider: str | None = typer.Option(
        None,
        "--provider",
        help="Require this registered provider ID for the descriptor.",
    ),
    record_set: str | None = typer.Option(
        None, "--record-set", help="Explicit Croissant recordSet name."
    ),
    distribution: str | None = typer.Option(
        None, "--distribution", help="Explicit Croissant distribution @id."
    ),
) -> None:
    """Normalize a descriptor into a deterministic Arrow IPC file."""
    try:
        policy = _source_policy(
            descriptor,
            source_root=source_root,
            offline=offline,
            cache_dir=cache_dir,
            max_resource_bytes=max_resource_bytes,
            max_resource_rows=max_resource_rows,
        )
        bundle, registry = _ingest_bundle(
            descriptor,
            policy=policy,
            expected_provider=provider,
            selection=_source_selection(
                record_set=record_set, distribution=distribution
            ),
        )
        bundle.write_ipc(output)
        typer.echo(
            json.dumps(
                _bundle_summary(bundle, registry),
                sort_keys=True,
            )
        )
    except IngestionError as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(_EXIT_INGESTION) from error
    except OSError as error:
        typer.echo("Error: normalized output could not be written", err=True)
        raise typer.Exit(_EXIT_OUTPUT) from error


@app.command("calculate-from-dataset")
def calculate_from_dataset(
    descriptor: Path = typer.Argument(..., exists=True, readable=True),
    table: str | None = typer.Option(
        None, "--table", help="Explicit normalized table ID for an inline binding."
    ),
    field: list[str] = typer.Option(
        [], "--field", help="Net-benefit field; repeat per strategy."
    ),
    strategy: list[str] = typer.Option(
        [], "--strategy", help="Optional strategy name; repeat in field order."
    ),
    binding_profile: Path | None = typer.Option(
        None,
        "--binding-profile",
        exists=True,
        readable=True,
        help="Explicit JSON BindingProfile; cannot be combined with inline binding flags.",
    ),
    source_root: Path | None = typer.Option(
        None,
        "--source-root",
        file_okay=False,
        exists=True,
        readable=True,
        help="Explicit local root allowed for declared resources.",
    ),
    offline: bool = typer.Option(False, "--offline", help="Require an offline replay."),
    cache_dir: Path | None = typer.Option(
        None, "--cache-dir", help="Verified materialization cache directory."
    ),
    max_resource_bytes: int = typer.Option(
        512 * 1024 * 1024, "--max-resource-bytes", min=1
    ),
    max_resource_rows: int = typer.Option(10_000_000, "--max-resource-rows", min=1),
    provider: str | None = typer.Option(
        None,
        "--provider",
        help="Require this registered provider ID for the descriptor.",
    ),
    record_set: str | None = typer.Option(
        None, "--record-set", help="Explicit Croissant recordSet name."
    ),
    distribution: str | None = typer.Option(
        None, "--distribution", help="Explicit Croissant distribution @id."
    ),
) -> None:
    """Calculate EVPI from explicitly selected normalized net-benefit fields."""
    try:
        bundle, _ = _ingest_bundle(
            descriptor,
            policy=_source_policy(
                descriptor,
                source_root=source_root,
                offline=offline,
                cache_dir=cache_dir,
                max_resource_bytes=max_resource_bytes,
                max_resource_rows=max_resource_rows,
            ),
            expected_provider=provider,
            selection=_source_selection(
                record_set=record_set, distribution=distribution
            ),
        )
        manifest = _calculation_manifest(
            bundle,
            table=table,
            field=field,
            strategy=strategy,
            binding_profile=binding_profile,
        )
        prepared = prepare_analysis_inputs(
            NormalizedInputBundle(manifest=manifest, tables=bundle.tables)
        )
        typer.echo(
            json.dumps(
                {
                    "binding_profile_digest": prepared.binding_profile_digest,
                    "evpi": evpi(prepared.net_benefits),
                    "input_digest": prepared.input_digest,
                },
                sort_keys=True,
            )
        )
    except IngestionError as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(_EXIT_INGESTION) from error
    except ValueError as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(_EXIT_BINDING) from error
