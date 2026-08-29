"""Consumer contracts for the format-neutral DataFrame ingestion SDK."""

from __future__ import annotations

from datetime import UTC, datetime

import pyarrow as pa
import pytest

from voiage.contracts import VOIBinding, prepare_analysis_inputs
from voiage.ingestion import INGESTION_PROVIDER_SDK_VERSION, from_dataframe
from voiage.ingestion import dataframe as dataframe_module


class _RecordingFrame:
    """A minimal protocol producer that records the requested copy policy."""

    def __init__(self, table: pa.Table) -> None:
        self.table = table
        self.copy_policies: list[bool] = []

    def __dataframe__(self, *, allow_copy: bool = True) -> object:
        self.copy_policies.append(allow_copy)
        return self.table.__dataframe__(allow_copy=allow_copy)


class _CapsuleOnlyFrame:
    """Expose Arrow capsules without a producer-native column-name attribute."""

    def __init__(self, table: pa.Table) -> None:
        self.table = table

    def __arrow_c_stream__(self, requested_schema: object | None = None) -> object:
        return self.table.__arrow_c_stream__(requested_schema)


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
        "adapter_version": "2",
        "conversion_protocol": "arrow_py_capsule",
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
    assert extension["conversion_protocol"] == "arrow_py_capsule"
    assert bundle.manifest.diagnostics[0].code == "dataframe_interchange.copy.unknown"


def test_arrow_capsule_without_declared_names_uses_capsule_protocol() -> None:
    table, protocol = dataframe_module._to_arrow_table(
        _CapsuleOnlyFrame(pa.table({"value": [1]})),
        allow_copy=True,
    )

    assert table.column_names == ["value"]
    assert protocol == "arrow_py_capsule"


def test_invalid_arrow_capsule_falls_back_to_dataframe_protocol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = pa.table({"value": [1]})
    producer = _CapsuleOnlyFrame(expected)

    def _reject_capsule(*_args: object, **_kwargs: object) -> pa.Table:
        raise TypeError("bad capsule")

    monkeypatch.setattr(
        dataframe_module.pa,
        "table",
        _reject_capsule,
    )
    monkeypatch.setattr(
        dataframe_module,
        "arrow_from_dataframe",
        lambda *_args, **_kwargs: expected,
    )

    table, protocol = dataframe_module._to_arrow_table(producer, allow_copy=True)

    assert table is expected
    assert protocol == "dataframe_interchange_fallback"


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


def test_dataframe_sdk_reports_pandas_nullable_category_and_timezone_decisions() -> (
    None
):
    """Pandas conversion records the Arrow decisions without retaining its index."""
    pandas = pytest.importorskip("pandas")
    index = pandas.Index(["first", "second"], name="scenario")
    frame = pandas.DataFrame(
        {
            "tier": pandas.Series(["standard", None], dtype="category", index=index),
            "cost": pandas.Series([10, None], dtype="Int64", index=index),
            "observed_at": pandas.Series(
                pandas.to_datetime(["2026-01-01T00:00:00Z", None]), index=index
            ),
        },
        index=index,
    )

    bundle = from_dataframe(frame, dataset_id="pandas-consumer")
    extension = bundle.manifest.model_dump(mode="json")["extensions"][
        "voiage.dev:dataframe-interchange"
    ]
    decisions = {item["field_id"]: item for item in extension["field_decisions"]}

    assert bundle.table("data").column_names == ["tier", "cost", "observed_at"]
    assert bundle.table("data").to_pylist()[1] == {
        "tier": None,
        "cost": None,
        "observed_at": None,
    }
    assert extension["copy_policy"] == "allow_copy"
    assert extension["copy_outcome"] == "not_observable"
    assert extension["index_policy"] == "excluded_by_dataframe_interchange_protocol"
    assert extension["conversion_protocol"] == "dataframe_interchange_fallback"
    assert decisions["tier"]["categorical"] is True
    assert decisions["tier"]["nullable"] is True
    assert decisions["cost"]["nullable"] is True
    assert decisions["observed_at"]["timezone"] == "UTC"
    assert decisions["observed_at"]["dtype"].startswith("timestamp[")


def test_dataframe_sdk_reports_polars_nullable_category_and_timezone_decisions() -> (
    None
):
    """Polars conversion reaches the same Arrow-backed diagnostic contract."""
    polars = pytest.importorskip("polars")
    frame = polars.DataFrame(
        {
            "tier": polars.Series(["standard", None], dtype=polars.Categorical),
            "cost": polars.Series([10, None], dtype=polars.Int64),
            "observed_at": polars.Series(
                [datetime(2026, 1, 1, tzinfo=UTC), None],
                dtype=polars.Datetime(time_zone="UTC"),
            ),
        }
    )

    bundle = from_dataframe(frame, dataset_id="polars-consumer")
    extension = bundle.manifest.model_dump(mode="json")["extensions"][
        "voiage.dev:dataframe-interchange"
    ]
    decisions = {item["field_id"]: item for item in extension["field_decisions"]}

    assert bundle.table("data").to_pylist()[1] == {
        "tier": None,
        "cost": None,
        "observed_at": None,
    }
    assert extension["copy_policy"] == "allow_copy"
    assert extension["copy_outcome"] == "not_observable"
    assert decisions["tier"]["categorical"] is True
    assert decisions["tier"]["nullable"] is True
    assert decisions["cost"]["nullable"] is True
    assert decisions["observed_at"]["timezone"] == "UTC"
    assert decisions["observed_at"]["dtype"].startswith("timestamp[")


def test_dataframe_sdk_preserves_a_non_contiguous_pandas_slice_on_the_standard_path() -> (
    None
):
    """A sliced consumer frame retains rows and explicit binding semantics."""
    pandas = pytest.importorskip("pandas")
    frame = pandas.DataFrame(
        {
            "strategy_a": [10.0, 99.0, 30.0, 99.0, 20.0],
            "strategy_b": [20.0, 99.0, 10.0, 99.0, 25.0],
            "segment": pandas.Series(
                ["a", "skip", "b", "skip", None], dtype="category"
            ),
        }
    )
    frame.index = pandas.Index([10, 11, 12, 13, 14], name="source_index")
    frame = frame.iloc[::2]

    bundle = from_dataframe(
        frame,
        dataset_id="sliced-consumer",
        table_id="samples",
        bindings=(_binding(),),
    )
    prepared = prepare_analysis_inputs(bundle)
    decisions = bundle.manifest.model_dump(mode="json")["extensions"][
        "voiage.dev:dataframe-interchange"
    ]["field_decisions"]

    assert bundle.table("samples").to_pylist() == [
        {"strategy_a": 10.0, "strategy_b": 20.0, "segment": "a"},
        {"strategy_a": 30.0, "strategy_b": 10.0, "segment": "b"},
        {"strategy_a": 20.0, "strategy_b": 25.0, "segment": None},
    ]
    assert prepared.net_benefits.numpy_values.tolist() == [
        [10.0, 20.0],
        [30.0, 10.0],
        [20.0, 25.0],
    ]
    assert "source_index" not in bundle.table("samples").column_names
    assert {item["field_id"] for item in decisions} == {
        "strategy_a",
        "strategy_b",
        "segment",
    }


def test_dataframe_sdk_retains_zero_row_schema_and_diagnostics() -> None:
    """An empty-but-typed producer is an auditable normalized input."""
    table = pa.table(
        {
            "strategy_a": pa.array([], type=pa.float64()),
            "strategy_b": pa.array([], type=pa.float64()),
        }
    )

    bundle = from_dataframe(
        table,
        dataset_id="zero-row",
        table_id="samples",
        bindings=(_binding(),),
        allow_copy=False,
    )

    assert bundle.table("samples").num_rows == 0
    assert bundle.schema_fingerprint
    assert bundle.manifest.model_dump(mode="json")["extensions"][
        "voiage.dev:dataframe-interchange"
    ]["field_decisions"] == [
        {
            "field_id": "strategy_a",
            "dtype": "double",
            "nullable": True,
            "categorical": False,
            "timezone": None,
        },
        {
            "field_id": "strategy_b",
            "dtype": "double",
            "nullable": True,
            "categorical": False,
            "timezone": None,
        },
    ]
    assert bundle.manifest.diagnostics[0].code == "dataframe_interchange.copy.zero_copy"


def test_dataframe_sdk_retains_a_no_column_producer_without_inventing_fields() -> None:
    """The adapter reports an empty schema rather than producer-specific data."""
    bundle = from_dataframe(pa.table({}), dataset_id="no-columns")
    extension = bundle.manifest.model_dump(mode="json")["extensions"][
        "voiage.dev:dataframe-interchange"
    ]

    assert bundle.table("data").num_rows == 0
    assert bundle.table("data").column_names == []
    assert bundle.manifest.tables[0].fields == ()
    assert extension["field_decisions"] == []
    assert extension["index_policy"] == "excluded_by_dataframe_interchange_protocol"
