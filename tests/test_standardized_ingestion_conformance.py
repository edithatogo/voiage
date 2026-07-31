"""Cross-format conformance for the canonical normalized decision fixture."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

from hypothesis import given, settings
from hypothesis import strategies as st
import numpy as np
import polars as pl
import pyarrow as pa
import pytest
import xarray as xr

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


def test_cross_format_preparation_has_identical_numpy_and_xarray_views() -> None:
    """Normalized preparation owns both existing compute-facing representations."""
    policy = SourceAccessPolicy(_FIXTURE_ROOT)
    bundles = (
        _direct_bundle(
            pa.table(
                {
                    "strategy_a": np.asarray([10.0, 30.0, 20.0]),
                    "strategy_b": np.asarray([20.0, 10.0, 25.0]),
                }
            )
        ),
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
    )
    prepared = tuple(prepare_analysis_inputs(bundle) for bundle in bundles)
    expected = prepared[0].net_benefits.values

    assert isinstance(expected, xr.DataArray)
    assert expected.dims == ("n_samples", "n_strategies")
    assert expected.strategy.values.tolist() == ["A", "B"]
    for item in prepared:
        assert item.net_benefits.values.equals(expected)
        assert np.array_equal(item.net_benefits.numpy_values, expected.values)


def test_canonical_source_formats_preserve_binding_quality_and_receipt_parity() -> None:
    """Provider metadata may differ, but decision inputs and receipts must agree."""
    policy = SourceAccessPolicy(_FIXTURE_ROOT)
    croissant = _bound(
        default_registry().ingest(
            _FIXTURE_ROOT / "canonical-decision.croissant.json", policy=policy
        )
    )
    frictionless = _bound(
        default_registry().ingest(
            _FIXTURE_ROOT / "canonical-decision.datapackage.json", policy=policy
        )
    )

    croissant_prepared = prepare_analysis_inputs(croissant)
    frictionless_prepared = prepare_analysis_inputs(frictionless)

    assert croissant_prepared.binding.model_dump(mode="json") == (
        frictionless_prepared.binding.model_dump(mode="json")
    )
    assert croissant_prepared.binding_profile_digest == (
        frictionless_prepared.binding_profile_digest
    )
    assert croissant_prepared.quality_report == frictionless_prepared.quality_report
    assert [
        receipt.model_dump(mode="json") for receipt in croissant.manifest.resources
    ] == [
        receipt.model_dump(mode="json") for receipt in frictionless.manifest.resources
    ]


def test_arrow_round_trips_are_equivalent_in_a_fresh_python_process(tmp_path) -> None:
    """IPC and Parquet must preserve the normalized table outside this process."""
    direct = _direct_bundle(
        pa.table(
            {
                "strategy_a": [10.0, 30.0, 20.0],
                "strategy_b": [20.0, 10.0, 25.0],
            }
        )
    )
    ipc_path = tmp_path / "canonical.arrow"
    parquet_path = tmp_path / "canonical.parquet"
    direct.write_ipc(ipc_path)
    direct.write_parquet(parquet_path)
    script = "\n".join(
        (
            "import json",
            "import polars as pl",
            "import sys",
            "from voiage.contracts import NormalizedInputBundle",
            "reader = getattr(NormalizedInputBundle, sys.argv[1])",
            "bundle = reader(sys.argv[2])",
            "table = bundle.table('samples')",
            "frame = pl.from_arrow(table)",
            "print(json.dumps({'schema': str(table.schema), 'rows': frame.to_dicts()}))",
        )
    )

    outputs = [
        subprocess.run(
            [sys.executable, "-c", script, reader, str(path)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        for reader, path in (("read_ipc", ipc_path), ("read_parquet", parquet_path))
    ]

    assert json.loads(outputs[0]) == json.loads(outputs[1])


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


@settings(max_examples=20, deadline=None)
@given(
    rows=st.lists(
        st.tuples(
            st.integers(min_value=-10_000, max_value=10_000),
            st.integers(min_value=-10_000, max_value=10_000),
        ),
        min_size=1,
        max_size=8,
    )
)
def test_provider_mapping_property_preserves_rows_and_explicit_binding(
    rows: list[tuple[int, int]],
) -> None:
    """Both descriptor formats preserve generated CSV rows without inference."""
    with TemporaryDirectory() as directory:
        root = Path(directory)
        csv_path = root / "samples.csv"
        csv_path.write_text(
            "strategy_a,strategy_b\n"
            + "".join(
                f"{strategy_a},{strategy_b}\n" for strategy_a, strategy_b in rows
            ),
            encoding="utf-8",
        )
        croissant_path = root / "croissant.json"
        croissant_path.write_text(
            json.dumps(
                {
                    "@context": "https://mlcommons.org/croissant/1.1",
                    "name": "property-decision-fixture",
                    "distribution": [{"contentUrl": csv_path.name}],
                    "recordSet": [
                        {
                            "name": "samples",
                            "field": [
                                {"name": "strategy_a"},
                                {"name": "strategy_b"},
                            ],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        frictionless_path = root / "datapackage.json"
        frictionless_path.write_text(
            json.dumps(
                {
                    "name": "property-decision-fixture",
                    "resources": [
                        {
                            "name": "samples",
                            "path": csv_path.name,
                            "schema": {
                                "fields": [
                                    {"name": "strategy_a", "type": "integer"},
                                    {"name": "strategy_b", "type": "integer"},
                                ]
                            },
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        policy = SourceAccessPolicy(root)
        croissant = _bound(default_registry().ingest(croissant_path, policy=policy))
        frictionless = _bound(
            default_registry().ingest(frictionless_path, policy=policy)
        )
        expected = [[strategy_a, strategy_b] for strategy_a, strategy_b in rows]

        assert (
            croissant.table("samples").to_pylist()
            == frictionless.table("samples").to_pylist()
        )
        for bundle in (croissant, frictionless):
            prepared = prepare_analysis_inputs(bundle)
            assert prepared.net_benefits.numpy_values.tolist() == expected
            assert prepared.net_benefits.strategy_names == ["A", "B"]
