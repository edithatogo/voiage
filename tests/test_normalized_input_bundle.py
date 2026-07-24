"""Behavioural tests for the format-neutral normalized input contract."""

from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator
import pyarrow as pa
from pydantic import ValidationError
import pytest

from voiage.contracts import (
    DatasetManifest,
    FieldManifest,
    NormalizedInputBundle,
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
