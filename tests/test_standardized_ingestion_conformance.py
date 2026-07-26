"""Cross-format conformance for the canonical normalized decision fixture."""

from __future__ import annotations

import json

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


def test_canonical_decision_fixture_has_cross_format_evpi_parity(tmp_path) -> None:
    """Every supported representation reaches one unchanged preparation path."""
    values = {"strategy_a": [10.0, 30.0, 20.0], "strategy_b": [20.0, 10.0, 25.0]}
    (tmp_path / "samples.csv").write_text(
        "strategy_a,strategy_b\n10.0,20.0\n30.0,10.0\n20.0,25.0\n",
        encoding="utf-8",
    )
    croissant_path = tmp_path / "croissant.json"
    croissant_path.write_text(
        json.dumps(
            {
                "@context": "https://mlcommons.org/croissant/1.1",
                "name": "canonical-decision-fixture",
                "distribution": [{"contentUrl": "samples.csv"}],
                "recordSet": [
                    {
                        "name": "samples",
                        "field": [{"name": "strategy_a"}, {"name": "strategy_b"}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    package_path = tmp_path / "datapackage.json"
    package_path.write_text(
        json.dumps(
            {
                "name": "canonical-decision-fixture",
                "resources": [
                    {
                        "name": "samples",
                        "path": "samples.csv",
                        "schema": {
                            "fields": [
                                {"name": "strategy_a", "type": "number"},
                                {"name": "strategy_b", "type": "number"},
                            ]
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    direct = _direct_bundle(pa.table(values))
    ipc_path = tmp_path / "fixture.arrow"
    parquet_path = tmp_path / "fixture.parquet"
    direct.write_ipc(ipc_path)
    direct.write_parquet(parquet_path)
    policy = SourceAccessPolicy(tmp_path)
    bundles = (
        direct,
        _bound(default_registry().ingest(croissant_path, policy=policy)),
        _bound(default_registry().ingest(package_path, policy=policy)),
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
