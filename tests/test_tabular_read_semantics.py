"""Defined inference and missing-value behavior at the migrated I/O boundary."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

from voiage import _tabular_io
from voiage.ecosystem_integration import DataFormatConnector
from voiage.exceptions import InputError
from voiage.methods.basic import evpi
from voiage.schema import ValueArray

FIXTURE = (
    Path(__file__).parent / "fixtures/tabular_io/read_semantics/all-null-mixed.csv"
)


def test_all_null_and_mixed_csv_schema_is_explicit() -> None:
    frame = _tabular_io.read_csv(FIXTURE)
    assert frame.schema == {
        "id": pl.Int64,
        "all_null": pl.Float64,
        "mixed": pl.String,
        "number": pl.Float64,
    }
    assert frame["all_null"].to_list() == [None, None, None]
    assert frame["mixed"].to_list() == ["1", "text", "3"]
    assert frame["number"].to_list() == [1.0, None, 3.0]


def test_public_null_payload_retains_legacy_values() -> None:
    legacy = pd.read_csv(FIXTURE)
    payload = DataFormatConnector().import_health_data(str(FIXTURE), "generic")
    assert payload["columns"] == list(legacy.columns)
    assert payload["shape"] == legacy.shape
    assert payload["missing_values"] == legacy.isna().sum().to_dict()
    for actual, expected in zip(
        payload["data"], legacy.to_dict("records"), strict=True
    ):
        for name, value in expected.items():
            if pd.isna(value):
                assert math.isnan(actual[name])
            else:
                assert actual[name] == value
                assert type(actual[name]) is type(value)


def test_late_text_prevents_numeric_cast(tmp_path: Path) -> None:
    path = tmp_path / "late.csv"
    path.write_text("mixed,number\n" + "1,1\n" * 101 + "text,1.5\n")
    frame = _tabular_io.read_csv(path)
    assert frame.schema == {"mixed": pl.String, "number": pl.Float64}
    assert frame["mixed"][0] == "1"
    assert frame["mixed"][-1] == "text"
    assert frame["number"][-1] == 1.5


def test_csv_nulls_propagate_without_imputation() -> None:
    frame = _tabular_io.read_csv(FIXTURE)
    derived = frame.select(
        (pl.col("all_null") + 1).alias("missing"),
        (pl.col("number") * 2).alias("number"),
    )
    assert derived["missing"].to_list() == [None, None, None]
    assert derived["number"].to_list() == [2.0, None, 6.0]
    assert derived["missing"].sum() == 0  # Native nullable reduction; no filled rows.
    assert derived["missing"].null_count() == 3


def test_parquet_preserves_schema_null_bitmap_and_nan(tmp_path: Path) -> None:
    path = tmp_path / "missing.parquet"
    original = pl.DataFrame(
        {
            "number": [1.0, None, float("nan")],
            "all_null": pl.Series([None] * 3, dtype=pl.Float64),
            "mixed": ["1", "text", "3"],
        }
    )
    original.write_parquet(path)
    frame = _tabular_io.read_parquet(path)
    assert frame.schema == original.schema
    assert frame["number"].null_count() == 1
    assert frame["number"].is_nan().sum() == 1
    derived = frame.select(pl.col("number") + 1)["number"]
    assert derived[0] == 2.0
    assert derived[1] is None
    assert math.isnan(derived[2])
    assert math.isnan(derived.sum())
    payload = DataFormatConnector().import_health_data(str(path))
    assert payload["missing_values"] == {"number": 2, "all_null": 3, "mixed": 0}
    assert math.isnan(payload["data"][1]["number"])
    assert math.isnan(payload["data"][2]["number"])


def test_csv_export_and_reload_preserve_missingness(tmp_path: Path) -> None:
    original = pl.DataFrame(
        {
            "number": [1.0, None, float("nan")],
            "all_null": pl.Series([None] * 3, dtype=pl.Float64),
            "mixed": ["1", "text", "3"],
        }
    )
    path = tmp_path / "missing.csv"
    _tabular_io.write_csv(original, path)
    legacy = pd.DataFrame(
        {
            "number": [1.0, float("nan"), float("nan")],
            "all_null": [float("nan")] * 3,
            "mixed": ["1", "text", "3"],
        }
    )
    assert path.read_bytes() == legacy.to_csv(index=False).encode("utf-8")
    reloaded = _tabular_io.read_csv(path)
    assert reloaded.schema == original.schema
    assert reloaded["number"].null_count() == 2
    assert reloaded["number"].is_nan().sum() == 0
    assert reloaded["mixed"].to_list() == original["mixed"].to_list()


def test_missing_inputs_do_not_change_numerical_container_policy() -> None:
    frame = _tabular_io.read_csv(FIXTURE).select("number")
    with pytest.raises(InputError, match="null"):
        ValueArray.from_polars(frame)
    values = frame.to_numpy()
    assert np.isnan(values[1, 0])
    with pytest.raises(InputError, match="finite") as error:
        evpi(values)
    assert error.value.diagnostic_code == "non_finite_value"


def test_header_only_csv_keeps_textual_empty_schema(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv"
    path.write_text("all_null,mixed\n")
    frame = _tabular_io.read_csv(path)
    assert frame.shape == (0, 2)
    assert frame.schema == {"all_null": pl.String, "mixed": pl.String}
