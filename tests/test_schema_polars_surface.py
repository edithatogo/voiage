"""Tests for the v3 Polars-native container surface (AC2).

These tests assert the new contract surface defined by ``specs/v3/stable-api.json``:
``ValueArray.frame`` / ``ParameterSet.frame`` (wide Polars frames), ``from_polars`` /
``to_polars``, ``ValueArray.values`` returning ``numpy.ndarray``, 3D multi-perspective
``pl.Array(pl.Float64, n_perspectives)`` columns with ``perspective_names``, and
construction-time shape/dtype/non-null validation.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from voiage.exceptions import InputError
from voiage.schema import ParameterSet, ValueArray


def test_value_array_frame_is_wide_polars_frame_with_strategy_columns() -> None:
    values = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    value_array = ValueArray.from_numpy(values, strategy_names=["A", "B"])

    assert isinstance(value_array.frame, pl.DataFrame)
    assert value_array.frame.columns == ["A", "B"]
    assert value_array.frame.width == 2
    assert value_array.frame.height == 2
    assert all(dtype == pl.Float64 for dtype in value_array.frame.dtypes)
    assert value_array.strategy_names == ["A", "B"]
    assert value_array.n_samples == 2
    assert value_array.n_strategies == 2


def test_value_array_values_returns_numpy_ndarray() -> None:
    values = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    value_array = ValueArray.from_numpy(values, strategy_names=["A", "B"])

    assert isinstance(value_array.values, np.ndarray)
    assert value_array.values.shape == (2, 2)
    np.testing.assert_array_equal(value_array.values, values)
    np.testing.assert_array_equal(value_array.numpy_values, values)


def test_value_array_from_polars_to_polars_round_trip() -> None:
    frame = pl.DataFrame({"A": [1.0, 3.0], "B": [2.0, 4.0]})
    value_array = ValueArray.from_polars(frame)

    assert value_array.strategy_names == ["A", "B"]
    assert value_array.n_samples == 2
    np.testing.assert_array_equal(value_array.numpy_values, frame.to_numpy())
    assert value_array.to_polars().equals(frame)


def test_value_array_from_polars_rejects_non_float_dtype() -> None:
    frame = pl.DataFrame({"A": [1, 2], "B": [3, 4]})
    with pytest.raises(InputError, match="[Ff]loat"):
        ValueArray.from_polars(frame)


def test_value_array_from_polars_rejects_null_values() -> None:
    frame = pl.DataFrame({"A": [1.0, None], "B": [3.0, 4.0]})
    with pytest.raises(InputError, match="null"):
        ValueArray.from_polars(frame)


def test_value_array_from_polars_rejects_empty_frame() -> None:
    with pytest.raises(InputError, match="strategy"):
        ValueArray.from_polars(pl.DataFrame())


def test_value_array_from_polars_rejects_duplicate_strategy_names() -> None:
    # Polars rejects duplicate columns at DataFrame construction, so the
    # duplicate guarantee is exercised through the naming constructor.
    with pytest.raises(InputError, match="[Dd]uplicate|unique"):
        ValueArray.from_numpy(
            np.array([[1.0, 2.0], [3.0, 4.0]]), strategy_names=["A", "A"]
        )


def test_value_array_from_numpy_perspectives_uses_array_columns() -> None:
    values = np.array(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float64
    )
    value_array = ValueArray.from_numpy_perspectives(
        values,
        strategy_names=["A", "B"],
        perspective_names=["payer", "societal"],
    )

    assert isinstance(value_array.frame, pl.DataFrame)
    assert value_array.frame.columns == ["A", "B"]
    assert value_array.frame.width == 2
    expected_dtype = pl.Array(pl.Float64, 2)
    assert all(dtype == expected_dtype for dtype in value_array.frame.dtypes)
    assert value_array.n_strategies == 2
    assert value_array.n_samples == 2
    assert value_array.perspective_names == ["payer", "societal"]


def test_value_array_from_numpy_perspectives_values_is_3d_ndarray() -> None:
    values = np.array(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float64
    )
    value_array = ValueArray.from_numpy_perspectives(
        values,
        strategy_names=["A", "B"],
        perspective_names=["payer", "societal"],
    )

    assert isinstance(value_array.values, np.ndarray)
    assert value_array.values.shape == (2, 2, 2)
    np.testing.assert_array_equal(value_array.values, values)
    np.testing.assert_array_equal(value_array.numpy_values, values)


def test_value_array_perspective_names_none_for_2d() -> None:
    value_array = ValueArray.from_numpy(np.array([[1.0, 2.0]]))
    assert value_array.perspective_names is None


def test_value_array_from_polars_perspective_defaults_and_round_trip() -> None:
    frame = pl.DataFrame(
        {
            "A": pl.Series("A", [[1.0, 2.0], [3.0, 4.0]], dtype=pl.Array(pl.Float64, 2)),
            "B": pl.Series("B", [[5.0, 6.0], [7.0, 8.0]], dtype=pl.Array(pl.Float64, 2)),
        }
    )
    value_array = ValueArray.from_polars(frame, perspective_names=["p1", "p2"])

    assert value_array.perspective_names == ["p1", "p2"]
    np.testing.assert_array_equal(
        value_array.numpy_values,
        np.array([[[1.0, 2.0], [5.0, 6.0]], [[3.0, 4.0], [7.0, 8.0]]]),
    )
    assert value_array.to_polars().equals(frame)


def test_value_array_copy_returns_independent_container() -> None:
    value_array = ValueArray.from_numpy(
        np.array([[1.0, 2.0], [3.0, 4.0]]), strategy_names=["A", "B"]
    )
    copied = value_array.copy()

    assert copied is not value_array
    assert copied == value_array
    copied_frame = copied.to_polars()
    copied_frame = copied_frame.with_columns(pl.col("A").alias("A").cast(pl.Float64))
    assert value_array.to_polars().equals(
        pl.DataFrame({"A": [1.0, 3.0], "B": [2.0, 4.0]})
    )


def test_value_array_slice_by_strategies_preserves_declared_order() -> None:
    value_array = ValueArray.from_numpy(
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        strategy_names=["A", "B", "C"],
    )
    sliced = value_array.slice_by_strategies(["C", "A"])

    assert isinstance(sliced, ValueArray)
    assert sliced.strategy_names == ["C", "A"]
    np.testing.assert_array_equal(sliced.numpy_values, np.array([[3.0, 1.0], [6.0, 4.0]]))


def test_parameter_set_frame_is_wide_polars_frame_per_parameter() -> None:
    parameter_set = ParameterSet.from_numpy_or_dict(
        {"alpha": np.array([0.1, 0.2]), "beta": np.array([1.0, 2.0])}
    )

    assert isinstance(parameter_set.frame, pl.DataFrame)
    assert parameter_set.frame.columns == ["alpha", "beta"]
    assert parameter_set.frame.width == 2
    assert parameter_set.frame.height == 2
    assert all(dtype == pl.Float64 for dtype in parameter_set.frame.dtypes)
    assert parameter_set.parameter_names == ["alpha", "beta"]
    assert parameter_set.n_samples == 2


def test_parameter_set_parameters_from_frame_columns() -> None:
    parameter_set = ParameterSet.from_numpy_or_dict(
        {"alpha": np.array([0.1, 0.2]), "beta": np.array([1.0, 2.0])}
    )

    assert isinstance(parameter_set.parameters, dict)
    np.testing.assert_array_equal(parameter_set.parameters["alpha"], [0.1, 0.2])
    np.testing.assert_array_equal(parameter_set.parameters["beta"], [1.0, 2.0])


def test_parameter_set_from_polars_to_polars_round_trip() -> None:
    frame = pl.DataFrame({"alpha": [0.1, 0.2], "beta": [1.0, 2.0]})
    parameter_set = ParameterSet.from_polars(frame)

    assert parameter_set.parameter_names == ["alpha", "beta"]
    assert parameter_set.n_samples == 2
    assert parameter_set.to_polars().equals(frame)


def test_parameter_set_from_polars_rejects_non_float_and_null() -> None:
    with pytest.raises(InputError, match="[Ff]loat"):
        ParameterSet.from_polars(pl.DataFrame({"alpha": [1, 2]}))
    with pytest.raises(InputError, match="null"):
        ParameterSet.from_polars(pl.DataFrame({"alpha": [0.1, None]}))


def test_parameter_set_from_polars_rejects_empty_frame() -> None:
    with pytest.raises(InputError, match="parameter"):
        ParameterSet.from_polars(pl.DataFrame())


def test_parameter_set_from_polars_rejects_extra_variable_dtype() -> None:
    # A non-numeric extra variable fails closed rather than being retained.
    with pytest.raises(InputError, match="Float64|extra"):
        ParameterSet.from_polars(pl.DataFrame({"alpha": [0.1, 0.2], "label": ["a", "b"]}))


def test_value_array_from_numpy_keeps_existing_validation() -> None:
    with pytest.raises(InputError, match="2D"):
        ValueArray.from_numpy(np.array([1.0, 2.0]))
    with pytest.raises(InputError, match="strategy_names"):
        ValueArray.from_numpy(np.array([[1.0, 2.0]]), strategy_names=["only-one"])


def test_value_array_equality_covers_values_and_names() -> None:
    a = ValueArray.from_numpy(np.array([[1.0, 2.0]]), strategy_names=["A", "B"])
    b = ValueArray.from_numpy(np.array([[1.0, 2.0]]), strategy_names=["A", "B"])
    c = ValueArray.from_numpy(np.array([[1.0, 3.0]]), strategy_names=["A", "B"])
    d = ValueArray.from_numpy(np.array([[1.0, 2.0]]), strategy_names=["A", "C"])

    assert a == b
    assert a != c
    assert a != d


def test_parameter_set_equality_covers_values_and_names() -> None:
    a = ParameterSet.from_numpy_or_dict({"x": np.array([1.0, 2.0])})
    b = ParameterSet.from_numpy_or_dict({"x": np.array([1.0, 2.0])})
    c = ParameterSet.from_numpy_or_dict({"x": np.array([1.0, 3.0])})
    d = ParameterSet.from_numpy_or_dict({"y": np.array([1.0, 2.0])})

    assert a == b
    assert a != c
    assert a != d