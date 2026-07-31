"""Consumer contracts for the format-neutral DataFrame ingestion SDK."""

from __future__ import annotations

from datetime import UTC, datetime

import pyarrow as pa
import pytest

from voiage.contracts import VOIBinding, prepare_analysis_inputs
from voiage.ingestion import INGESTION_PROVIDER_SDK_VERSION, from_dataframe


class _RecordingFrame:
    """A minimal protocol producer that records the requested copy policy."""

    def __init__(self, table: pa.Table) -> None:
        self.table = table
        self.copy_policies: list[bool] = []

    def __dataframe__(self, *, allow_copy: bool = True) -> object:
        self.copy_policies.append(allow_copy)
        return self.table.__dataframe__(allow_copy=allow_copy)


def _binding() -> VOIBinding:
    return VOIBinding(
        role="net_benefit",
        table_id="samples",
        field_ids=("strategy_a", "strategy_b"),
        strategy_names=("A", "B"),
    )


def test_dataframe_sdk_forwards_copy_policy_and_preserves_nullable_schema() -> None:
    """A generic producer crosses the SDK boundary without semantic inference."""
    producer = _RecordingFrame(
        pa.table(
            {
                "strategy_a": pa.array([10.0, None, 20.0]),
                "strategy_b": pa.array([20.0, 10.0, 25.0]),
                "observed_at": pa.array(
                    [datetime(2026, 1, 1, tzinfo=UTC)] * 3,
                    type=pa.timestamp("s", tz="UTC"),
                ),
            }
        )
    )

    bundle = from_dataframe(
        producer,
        dataset_id="business-scenarios",
        table_id="samples",
        bindings=(_binding(),),
        allow_copy=False,
    )

    assert producer.copy_policies == [False]
    assert bundle.manifest.provenance.provider_id == "dataframe-interchange"
    assert bundle.table("samples").schema == producer.table.schema
    assert bundle.manifest.tables[0].fields[2].dtype == "timestamp[s, tz=UTC]"


def test_dataframe_sdk_records_copy_and_schema_conversion_diagnostics() -> None:
    """Consumers can audit the non-semantic interchange decisions made."""
    table = pa.table(
        {
            "tier": pa.array(["standard", None]).dictionary_encode(),
            "observed_at": pa.array(
                [datetime(2026, 1, 1, tzinfo=UTC), None],
                type=pa.timestamp("s", tz="UTC"),
            ),
        }
    )

    bundle = from_dataframe(table, dataset_id="auditable-sdk", allow_copy=False)
    extension = bundle.manifest.model_dump(mode="json")["extensions"][
        "voiage.dev:dataframe-interchange"
    ]

    assert extension == {
        "adapter_version": "1",
        "copy_outcome": "zero_copy",
        "copy_policy": "disallow_copy",
        "field_decisions": [
            {
                "categorical": True,
                "dtype": "dictionary<values=string, indices=int32, ordered=0>",
                "field_id": "tier",
                "nullable": True,
                "timezone": None,
            },
            {
                "categorical": False,
                "dtype": "timestamp[s, tz=UTC]",
                "field_id": "observed_at",
                "nullable": True,
                "timezone": "UTC",
            },
        ],
        "index_policy": "excluded_by_dataframe_interchange_protocol",
    }
    assert bundle.manifest.diagnostics[0].code == "dataframe_interchange.copy.zero_copy"


def test_dataframe_sdk_does_not_claim_an_unobservable_copy_outcome() -> None:
    """Arrow's public interchange API does not report copies when allowed."""
    bundle = from_dataframe(pa.table({"value": [1]}), dataset_id="auditable-sdk")
    extension = bundle.manifest.model_dump(mode="json")["extensions"][
        "voiage.dev:dataframe-interchange"
    ]

    assert extension["copy_policy"] == "allow_copy"
    assert extension["copy_outcome"] == "not_observable"
    assert bundle.manifest.diagnostics[0].code == "dataframe_interchange.copy.unknown"


def test_dataframe_sdk_bundle_uses_the_standard_preparation_contract() -> None:
    """DataFrame callers receive the same explicit-binding VOI input as providers."""
    bundle = from_dataframe(
        pa.table({"strategy_a": [10.0, 30.0], "strategy_b": [20.0, 10.0]}),
        dataset_id="business-scenarios",
        table_id="samples",
        bindings=(_binding(),),
    )

    prepared = prepare_analysis_inputs(bundle)

    assert prepared.net_benefits.numpy_values.tolist() == [[10.0, 20.0], [30.0, 10.0]]
    assert prepared.net_benefits.strategy_names == ["A", "B"]


def test_dataframe_sdk_rejects_a_binding_outside_the_declared_table() -> None:
    """SDK callers cannot attach a binding that does not resolve in the bundle."""
    invalid = VOIBinding(
        role="net_benefit",
        table_id="missing",
        field_ids=("strategy_a",),
    )

    with pytest.raises(ValueError, match="unknown table"):
        from_dataframe(
            pa.table({"strategy_a": [10.0]}),
            dataset_id="business-scenarios",
            bindings=(invalid,),
        )


def test_dataframe_sdk_v1_preserves_index_exclusion_and_category_values() -> None:
    """The interchange contract retains columns, not producer-specific indexes."""
    pandas = pytest.importorskip("pandas")
    index = pandas.Index(["one", "two"], name="scenario")
    frame = pandas.DataFrame(
        {
            "tier": pandas.Series(["standard", None], dtype="category", index=index),
            "net_benefit": pandas.Series([10, None], dtype="Int64", index=index),
        },
        index=index,
    )

    bundle = from_dataframe(frame, dataset_id="consumer-v1")

    assert INGESTION_PROVIDER_SDK_VERSION == "1"
    assert bundle.table("data").column_names == ["tier", "net_benefit"]
    assert bundle.table("data").to_pylist() == [
        {"tier": "standard", "net_benefit": 10},
        {"tier": None, "net_benefit": None},
    ]
