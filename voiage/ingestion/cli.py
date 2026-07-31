"""Small CLI surface for inspecting and normalizing supported descriptors."""

# Typer builds command declarations from defaults.
# ruff: noqa: B008

from __future__ import annotations

import json
from pathlib import Path  # noqa: TC003 - public CLI annotation
from typing import cast

import typer

from voiage.contracts.normalized_input import (
    DatasetManifest,
    NormalizedInputBundle,
    VOIBinding,
)
from voiage.contracts.preparation import prepare_analysis_inputs
from voiage.ingestion.base import IngestionError, SourceAccessPolicy
from voiage.ingestion.registry import default_registry
from voiage.methods.basic import evpi

app = typer.Typer(help="Validate and normalize standardized dataset descriptors.")


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
) -> SourceAccessPolicy:
    """Build an explicit, local-only policy for one CLI invocation."""
    return SourceAccessPolicy(
        source_root or descriptor.parent,
        offline=offline,
        cache_dir=cache_dir,
        max_resource_bytes=max_resource_bytes,
    )


def _bundle_summary(
    descriptor: Path,
    *,
    binding: VOIBinding | None = None,
    policy: SourceAccessPolicy | None = None,
) -> dict[str, object]:
    """Return stable, non-secret metadata for a descriptor."""
    registry = default_registry()
    bundle = registry.ingest(descriptor, policy=policy)
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


def _inspection_summary(descriptor: Path) -> dict[str, object]:
    """Return descriptor-only diagnostics without resolving any resources.

    Binding resolution, provenance receipts, and data-quality reports require
    materializing resources. Those details deliberately belong to the
    materializing validation, normalization, and calculation commands rather
    than the metadata-only ``inspect`` command.
    """
    inspection = default_registry().inspect(descriptor)
    capabilities = cast("dict[str, object]", inspection["capabilities"])
    return {
        "binding_resolution": None,
        "capabilities": {
            "provider_id": inspection["provider_id"],
            **capabilities,
        },
        "descriptor": inspection["descriptor"],
        "provider": inspection["provider_id"],
    }


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
) -> None:
    """Validate a supported descriptor and its declared local resources."""
    try:
        typer.echo(
            json.dumps(
                {
                    "valid": True,
                    **_bundle_summary(
                        descriptor,
                        policy=_source_policy(
                            descriptor,
                            source_root=source_root,
                            offline=offline,
                            cache_dir=cache_dir,
                            max_resource_bytes=max_resource_bytes,
                        ),
                    ),
                },
                sort_keys=True,
            )
        )
    except IngestionError as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(2) from error


@app.command()
def inspect(
    descriptor: Path = typer.Argument(..., exists=True, readable=True),
) -> None:
    """Inspect descriptor identity and provider capabilities without loading data."""
    try:
        typer.echo(json.dumps(_inspection_summary(descriptor), sort_keys=True))
    except IngestionError as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(2) from error


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
) -> None:
    """Normalize a descriptor into a deterministic Arrow IPC file."""
    try:
        policy = _source_policy(
            descriptor,
            source_root=source_root,
            offline=offline,
            cache_dir=cache_dir,
            max_resource_bytes=max_resource_bytes,
        )
        bundle = default_registry().ingest(descriptor, policy=policy)
        bundle.write_ipc(output)
        typer.echo(
            json.dumps(_bundle_summary(descriptor, policy=policy), sort_keys=True)
        )
    except IngestionError as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(2) from error


@app.command("calculate-from-dataset")
def calculate_from_dataset(
    descriptor: Path = typer.Argument(..., exists=True, readable=True),
    table: str = typer.Option(..., "--table", help="Explicit normalized table ID."),
    field: list[str] = typer.Option(
        ..., "--field", help="Net-benefit field; repeat per strategy."
    ),
    strategy: list[str] = typer.Option(
        [], "--strategy", help="Optional strategy name; repeat in field order."
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
) -> None:
    """Calculate EVPI from explicitly selected normalized net-benefit fields."""
    try:
        bundle = default_registry().ingest(
            descriptor,
            policy=_source_policy(
                descriptor,
                source_root=source_root,
                offline=offline,
                cache_dir=cache_dir,
                max_resource_bytes=max_resource_bytes,
            ),
        )
        binding = VOIBinding(
            role="net_benefit",
            table_id=table,
            field_ids=tuple(field),
            strategy_names=tuple(strategy),
        )
        manifest = DatasetManifest(
            **bundle.manifest.model_dump(mode="python", exclude={"bindings"}),
            bindings=(binding,),
        )
        prepared = prepare_analysis_inputs(
            NormalizedInputBundle(manifest=manifest, tables=bundle.tables)
        )
        typer.echo(
            json.dumps(
                {
                    "evpi": evpi(prepared.net_benefits),
                    "input_digest": prepared.input_digest,
                },
                sort_keys=True,
            )
        )
    except (IngestionError, ValueError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(2) from error
