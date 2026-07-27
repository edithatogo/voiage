"""Cross-format conformance for the canonical normalized decision fixture."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st
import numpy as np
import polars as pl
import pyarrow as pa
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
from voiage.ingestion import SourceAccessPolicy, default_registry, from_dataframe
from voiage.methods.basic import evpi

_FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "standardized_ingestion"


def _binding() -> VOIBinding:
    return VOIBinding(
        role="net_benefit",
        table_id="samples",
        field_ids=("strategy_a", "strategy_b"),
        strategy_names=("A", "B"),
    )


def _bound(bundle: NormalizedInputBundle) -> NormalizedInputBundle:
    """Attach the canonical binding without changing a source table or metadata."""
    return NormalizedInputBundle(
        manifest=bundle.manifest.model_copy(update={"bindings": (_binding(),)}),
        tables=bundle.tables,
    )


def _direct_bundle(table: pa.Table) -> NormalizedInputBundle:
    return NormalizedInputBundle(
        manifest=DatasetManifest(
            dataset_id="canonical-decision-fixture",
            tables=(
                TableManifest(
                    table_id="samples",
                    fields=tuple(
                        FieldManifest(field_id=field.name, dtype=str(field.type))
                        for field in table.schema
                    ),
                ),
            ),
            provenance=SourceProvenance(
                provider_id="direct-fixture",
                source_uri="urn:voiage:canonical-decision-fixture",
                descriptor_digest="f" * 64,
            ),
            bindings=(_binding(),),
        ),
        tables={"samples": table},
    )


def test_canonical_decision_fixture_manifest_pins_source_artifacts() -> None:
    """Fixture bytes and its explicit VOI binding remain deterministic evidence."""
    manifest = json.loads(
        (_FIXTURE_ROOT / "canonical-decision.manifest.json").read_text(encoding="utf-8")
    )

    assert manifest["schema_version"] == "1.0.0"
    assert manifest["dataset_id"] == "canonical-decision-fixture"
    assert manifest["binding"] == {
        "field_ids": ["strategy_a", "strategy_b"],
        "role": "net_benefit",
        "strategy_names": ["A", "B"],
        "table_id": "samples",
    }
    assert {
        path.name: sha256(path.read_bytes()).hexdigest()
        for path in _FIXTURE_ROOT.glob("canonical-decision.*")
        if path.name != "canonical-decision.manifest.json"
    } == manifest["files"]


def test_canonical_decision_fixture_has_cross_format_evpi_parity(tmp_path) -> None:
    """Every supported representation reaches one unchanged preparation path."""
    values = {"strategy_a": [10.0, 30.0, 20.0], "strategy_b": [20.0, 10.0, 25.0]}

    direct = _direct_bundle(pa.table(values))
    ipc_path = tmp_path / "fixture.arrow"
    parquet_path = tmp_path / "fixture.parquet"
    direct.write_ipc(ipc_path)
    direct.write_parquet(parquet_path)
    policy = SourceAccessPolicy(_FIXTURE_ROOT)
    bundles = (
        direct,
        _bound(
            default_registry().ingest(
                _FIXTURE_ROOT / "canonical-decision.croissant.json", policy=policy
            )
        ),
        _bound(
            default_registry().ingest(
                _FIXTURE_ROOT / "canonical-decision.datapackage.json", policy=policy
            )
        ),
        NormalizedInputBundle.read_ipc(ipc_path),
        NormalizedInputBundle.read_parquet(parquet_path),
        from_dataframe(
            pl.DataFrame(values),
            dataset_id="canonical-decision-fixture",
            table_id="samples",
            bindings=(_binding(),),
        ),
    )

    prepared = tuple(prepare_analysis_inputs(bundle) for bundle in bundles)
    expected_values = np.asarray([[10.0, 20.0], [30.0, 10.0], [20.0, 25.0]])
    expected_evpi = evpi(expected_values)

    for item in prepared:
        assert item.net_benefits.numpy_values.tolist() == expected_values.tolist()
        assert item.net_benefits.strategy_names == ["A", "B"]
        assert evpi(item.net_benefits.numpy_values) == pytest.approx(expected_evpi)
    assert {bundle.schema_fingerprint for bundle in bundles} == {
        direct.schema_fingerprint
    }


@settings(max_examples=30, deadline=None)
@given(
    rows=st.lists(
        st.tuples(
            st.floats(
                min_value=-1_000_000,
                max_value=1_000_000,
                allow_nan=False,
                allow_infinity=False,
            ),
            st.floats(
                min_value=-1_000_000,
                max_value=1_000_000,
                allow_nan=False,
                allow_infinity=False,
            ),
        ),
        min_size=1,
        max_size=20,
    ),
    column_order=st.permutations(("strategy_a", "strategy_b")),
)
def test_dataframe_and_direct_input_preserve_explicit_binding_under_column_order(
    rows: list[tuple[float, float]], column_order: list[str]
) -> None:
    """Explicit bindings, not producer column order, define decision semantics."""
    values = {
        "strategy_a": [row[0] for row in rows],
        "strategy_b": [row[1] for row in rows],
    }
    ordered_values = {field: values[field] for field in column_order}
    direct = _direct_bundle(pa.table(ordered_values))
    dataframe = from_dataframe(
        pl.DataFrame(ordered_values),
        dataset_id="property-fixture",
        table_id="samples",
        bindings=(_binding(),),
    )

    direct_prepared = prepare_analysis_inputs(direct)
    dataframe_prepared = prepare_analysis_inputs(dataframe)
    expected = np.asarray(
        list(zip(values["strategy_a"], values["strategy_b"], strict=True))
    )

    assert direct_prepared.net_benefits.numpy_values.tolist() == expected.tolist()
    assert dataframe_prepared.net_benefits.numpy_values.tolist() == expected.tolist()
    assert evpi(direct_prepared.net_benefits.numpy_values) == pytest.approx(
        evpi(dataframe_prepared.net_benefits.numpy_values)
    )
