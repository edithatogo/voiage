"""Small CLI surface for inspecting and normalizing supported descriptors."""

# Typer builds command declarations from defaults.
# ruff: noqa: B008

from __future__ import annotations

import json
from pathlib import Path  # noqa: TC003 - public CLI annotation

import typer

from voiage.contracts.normalized_input import (
    DatasetManifest,
    NormalizedInputBundle,
    VOIBinding,
)
from voiage.contracts.preparation import prepare_analysis_inputs
from voiage.ingestion.base import IngestionError
from voiage.ingestion.registry import default_registry
from voiage.methods.basic import evpi

app = typer.Typer(help="Validate and normalize standardized dataset descriptors.")


def _bundle_summary(descriptor: Path) -> dict[str, object]:
    """Return stable, non-secret metadata for a descriptor."""
    bundle = default_registry().ingest(descriptor)
    return {
        "content_digest": bundle.content_digest,
        "dataset_id": bundle.manifest.dataset_id,
        "provider": bundle.manifest.provenance.provider_id,
        "schema_fingerprint": bundle.schema_fingerprint,
        "tables": {name: table.num_rows for name, table in bundle.tables.items()},
    }


@app.command()
def inspect(descriptor: Path = typer.Argument(..., exists=True, readable=True)) -> None:
    """Inspect a descriptor and emit its normalized identity as JSON."""
    try:
        typer.echo(json.dumps(_bundle_summary(descriptor), sort_keys=True))
    except IngestionError as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(2) from error


@app.command()
def normalize(
    descriptor: Path = typer.Argument(..., exists=True, readable=True),
    output: Path = typer.Option(..., "--output", "-o"),
) -> None:
    """Normalize a descriptor into a deterministic Arrow IPC file."""
    try:
        bundle = default_registry().ingest(descriptor)
        bundle.write_ipc(output)
        typer.echo(json.dumps(_bundle_summary(descriptor), sort_keys=True))
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
) -> None:
    """Calculate EVPI from explicitly selected normalized net-benefit fields."""
    try:
        bundle = default_registry().ingest(descriptor)
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
