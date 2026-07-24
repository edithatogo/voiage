"""Behavioural tests for the format-neutral normalized input contract."""

from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator
import pyarrow as pa
from pydantic import ValidationError
import pytest

from voiage.contracts import (
    BindingProfile,
    DatasetManifest,
    FieldManifest,
    IngestionDiagnostic,
    KeyReference,
    NormalizedInputBundle,
    ResourceManifest,
    SourceProvenance,
    TableManifest,
    VOIBinding,
    prepare_analysis_inputs,
)


def _bundle() -> NormalizedInputBundle:
    return NormalizedInputBundle(
        manifest=DatasetManifest(
            dataset_id="decision-fixture",
            tables=(
                TableManifest(
                    table_id="net_benefit",
                    fields=(
                        FieldManifest(field_id="strategy_a", dtype="float64"),
                        FieldManifest(field_id="strategy_b", dtype="float64"),
                    ),
                ),
            ),
            provenance=SourceProvenance(
                provider_id="direct",
                source_uri="file:///redacted/input.parquet",
                descriptor_digest="a" * 64,
            ),
            bindings=(
                VOIBinding(
                    role="net_benefit",
                    table_id="net_benefit",
                    field_ids=("strategy_a", "strategy_b"),
                    strategy_names=("A", "B"),
                ),
            ),
        ),
        tables={
            "net_benefit": pa.table(
                {"strategy_a": [1.0, 3.0], "strategy_b": [2.0, 1.0]}
            )
        },
    )


def test_bundle_is_strict_immutable_and_canonical() -> None:
    bundle = _bundle()

    assert bundle.content_digest == _bundle().content_digest
    assert json.loads(bundle.canonical_json)["dataset_id"] == "decision-fixture"
    with pytest.raises(TypeError):
        bundle.tables["other"] = pa.table({})  # type: ignore[index]


def test_bundle_rejects_stale_binding_reference() -> None:
    with pytest.raises(ValidationError, match="unknown field"):
        DatasetManifest(
            dataset_id="bad",
            tables=(
                TableManifest(
                    table_id="net_benefit",
                    fields=(FieldManifest(field_id="strategy_a", dtype="float64"),),
                ),
            ),
            provenance=SourceProvenance(
                provider_id="direct",
                source_uri="file:///input",
                descriptor_digest="a" * 64,
            ),
            bindings=(
                VOIBinding(
                    role="net_benefit",
                    table_id="net_benefit",
                    field_ids=("missing",),
                ),
            ),
        )


def test_bundle_arrow_round_trip(tmp_path) -> None:
    bundle = _bundle()
    path = tmp_path / "input.arrow"

    bundle.write_ipc(path)
    restored = NormalizedInputBundle.read_ipc(path)

    assert restored.content_digest == bundle.content_digest
    assert restored.table("net_benefit").equals(bundle.table("net_benefit"))


def test_preparation_preserves_explicit_binding_and_digest() -> None:
    prepared = prepare_analysis_inputs(_bundle())

    assert prepared.net_benefits.strategy_names == ["A", "B"]
    assert prepared.net_benefits.numpy_values.tolist() == [[1.0, 2.0], [3.0, 1.0]]
    assert prepared.input_digest == _bundle().content_digest


def test_manifest_matches_published_json_schema() -> None:
    schema_path = Path("specs/core-api/schemas/v2/normalized-input-bundle.schema.json")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    Draft202012Validator(schema).validate(_bundle().manifest.model_dump(mode="json"))


@pytest.mark.parametrize(
    "factory",
    [
        lambda: TableManifest(
            table_id="x", fields=(FieldManifest(field_id="a", dtype="float64"),) * 2
        ),
        lambda: TableManifest(
            table_id="x",
            fields=(FieldManifest(field_id="a", dtype="float64"),),
            primary_key=("missing",),
        ),
    ],
)
def test_table_manifest_rejects_invalid_field_declarations(factory) -> None:
    with pytest.raises(ValidationError):
        factory()


def test_binding_and_bundle_reject_invalid_shapes(tmp_path) -> None:
    with pytest.raises(ValidationError):
        VOIBinding(role="net_benefit", table_id="x", field_ids=())
    with pytest.raises(ValidationError):
        VOIBinding(
            role="net_benefit",
            table_id="x",
            field_ids=("a",),
            strategy_names=("A", "B"),
        )
    manifest = _bundle().manifest
    with pytest.raises(ValueError, match="names"):
        NormalizedInputBundle(manifest=manifest, tables={})
    invalid = tmp_path / "not-normalized.arrow"
    with pa.ipc.new_file(invalid, pa.schema([("a", pa.int64())])) as writer:
        writer.write_table(pa.table({"a": [1]}))
    with pytest.raises(ValueError, match="not a voiage"):
        NormalizedInputBundle.read_ipc(invalid)


def test_manifest_and_bundle_reject_duplicate_or_dangling_references() -> None:
    table = TableManifest(
        table_id="net_benefit",
        fields=(FieldManifest(field_id="a", dtype="float64"),),
    )
    provenance = _bundle().manifest.provenance
    with pytest.raises(ValidationError, match="table identifiers"):
        DatasetManifest(dataset_id="x", tables=(table, table), provenance=provenance)
    with pytest.raises(ValidationError, match="unknown table"):
        DatasetManifest(
            dataset_id="x",
            tables=(table,),
            provenance=provenance,
            bindings=(
                VOIBinding(role="net_benefit", table_id="other", field_ids=("a",)),
            ),
        )
    with pytest.raises(ValueError, match="columns"):
        NormalizedInputBundle(
            manifest=DatasetManifest(
                dataset_id="x", tables=(table,), provenance=provenance
            ),
            tables={"net_benefit": pa.table({"b": [1.0]})},
        )


def test_ipc_export_rejects_multiple_tables(tmp_path) -> None:
    manifest = _bundle().manifest.model_copy(
        update={
            "tables": (
                *_bundle().manifest.tables,
                TableManifest(
                    table_id="other",
                    fields=(FieldManifest(field_id="x", dtype="float64"),),
                ),
            )
        }
    )
    bundle = NormalizedInputBundle(
        manifest=manifest,
        tables={**_bundle().tables, "other": pa.table({"x": [1.0]})},
    )
    with pytest.raises(ValueError, match="exactly one"):
        bundle.write_ipc(tmp_path / "bundle.arrow")


def test_manifest_preserves_explicit_resource_and_relationship_contracts() -> None:
    manifest = DatasetManifest(
        dataset_id="linked",
        tables=(
            TableManifest(
                table_id="samples",
                fields=(FieldManifest(field_id="strategy_id", dtype="string"),),
                primary_key=("strategy_id",),
            ),
            TableManifest(
                table_id="outcomes",
                fields=(FieldManifest(field_id="strategy_id", dtype="string"),),
            ),
        ),
        resources=(
            ResourceManifest(
                resource_id="samples-csv",
                uri="samples.csv",
                sha256="b" * 64,
                media_type="text/csv",
            ),
        ),
        key_references=(
            KeyReference(
                source_table_id="outcomes",
                source_field_ids=("strategy_id",),
                target_table_id="samples",
                target_field_ids=("strategy_id",),
            ),
        ),
        diagnostics=(
            IngestionDiagnostic(code="validated", severity="info", message="ok"),
        ),
        provenance=SourceProvenance(
            provider_id="direct",
            source_uri="file:///descriptor.json",
            descriptor_digest="c" * 64,
        ),
    )

    payload = json.loads(manifest.canonical_json())

    assert payload["resources"][0]["resource_id"] == "samples-csv"
    assert payload["key_references"][0]["target_table_id"] == "samples"
    assert payload["diagnostics"][0]["severity"] == "info"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"extensions": {"plain": "value"}}, "namespaced"),
        ({"extensions": {"vendor:extension": "value"}}, "namespaced"),
        (
            {
                "resources": (
                    ResourceManifest(resource_id="x", uri="x", sha256="d" * 64),
                )
                * 2
            },
            "resource identifiers",
        ),
    ],
)
def test_manifest_rejects_ambiguous_extensions_and_resources(kwargs, message) -> None:
    with pytest.raises(ValidationError, match=message):
        DatasetManifest(
            dataset_id="bad",
            tables=(
                TableManifest(
                    table_id="t", fields=(FieldManifest(field_id="x", dtype="float64"),)
                ),
            ),
            provenance=SourceProvenance(
                provider_id="direct",
                source_uri="file:///descriptor.json",
                descriptor_digest="d" * 64,
            ),
            **kwargs,
        )


def test_provenance_redacts_credential_bearing_source_uris() -> None:
    with pytest.raises(ValidationError, match="credentials or query strings"):
        SourceProvenance(
            provider_id="direct",
            source_uri="https://user:secret@example.test/data?token=secret",
            descriptor_digest="d" * 64,
        )


def test_binding_profile_is_versioned_deterministic_and_reference_checked() -> None:
    profile = BindingProfile(
        bindings=(
            VOIBinding(
                role="net_benefit",
                table_id="net_benefit",
                field_ids=("strategy_a", "strategy_b"),
                strategy_names=("A", "B"),
            ),
        )
    )
    manifest = _bundle().manifest.model_copy(update={"binding_profile": profile})

    assert (
        profile.digest
        == BindingProfile.model_validate_json(profile.canonical_json()).digest
    )
    assert manifest.bindings == profile.bindings
    with pytest.raises(ValueError, match="conflicts"):
        _bundle().manifest.model_copy(
            update={
                "binding_profile": BindingProfile(
                    bindings=(
                        VOIBinding(
                            role="net_benefit",
                            table_id="net_benefit",
                            field_ids=("strategy_b", "strategy_a"),
                        ),
                    )
                )
            }
        ).validate_references()
