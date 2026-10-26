"""Validation parity for the Polars containers (AC3).

Enumerates every structural error the xarray container raised and asserts the
new container raises the same Voiage exception type with an equivalent message.
The decision to reject unrecognised extra variables (fail closed) is asserted
here and recorded in the track evidence.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from voiage.exceptions import DimensionMismatchError, InputError
from voiage.schema import ParameterSet, ValueArray


def test_value_array_frame_type_is_input_error() -> None:
    # was: "ValueArray 'dataset' must be a xarray.Dataset." (InputError)
    with pytest.raises(InputError, match="polars.DataFrame"):
        ValueArray(frame="not-a-frame")  # type: ignore[arg-type]


def test_value_array_empty_frame_is_input_error() -> None:
    # was: missing n_samples / n_strategies dimension (InputError)
    with pytest.raises(InputError, match="at least one strategy"):
        ValueArray(frame=pl.DataFrame())


def test_value_array_wrong_dtype_and_extra_variable_fail_closed() -> None:
    # originally the xarray container silently retained extra data variables;
    # the Polars container rejects them explicitly (recorded decision).
    with pytest.raises(InputError, match="Float64|extra"):
        ValueArray(frame=pl.DataFrame({"A": [1, 2], "B": [3, 4]}))  # Int64
    with pytest.raises(InputError, match="Float64|extra"):
        ValueArray.from_polars(pl.DataFrame({"A": [1.0, 2.0], "label": ["a", "b"]}))


def test_value_array_mixed_dtype_fails_closed() -> None:
    frame = pl.DataFrame(
        {
            "A": pl.Series("A", [[1.0, 2.0]], dtype=pl.Array(pl.Float64, 2)),
            "B": [3.0],
        }
    )
    with pytest.raises(InputError, match="Float64|extra"):
        ValueArray.from_polars(frame)


def test_value_array_null_values_are_input_error() -> None:
    with pytest.raises(InputError, match="null"):
        ValueArray.from_polars(pl.DataFrame({"A": [1.0, None], "B": [3.0, 4.0]}))


def test_value_array_duplicate_strategy_names_are_input_error() -> None:
    with pytest.raises(InputError, match="unique"):
        ValueArray.from_numpy(np.array([[1.0, 2.0]]), strategy_names=["A", "A"])


def test_value_array_length_mismatch_is_input_error() -> None:
    # was: "strategy_names must have {n} elements" (InputError)
    with pytest.raises(InputError, match="strategy_names"):
        ValueArray.from_numpy(np.array([[1.0, 2.0]]), strategy_names=["A"])
    with pytest.raises(InputError, match="perspective_names"):
        ValueArray.from_numpy_perspectives(
            np.array([[[1.0, 2.0]]]),
            strategy_names=["A"],
            perspective_names=["only-one"],
        )


def test_value_array_ndim_errors_keep_exception_types() -> None:
    # was: DimensionMismatchError "values must be a 2D array"
    with pytest.raises(DimensionMismatchError, match="2D"):
        ValueArray.from_numpy(np.array([1.0, 2.0]))
    # was: InputError "values must be a 3D array"
    with pytest.raises(InputError, match="3D"):
        ValueArray.from_numpy_perspectives(np.array([[1.0, 2.0]]))


def test_parameter_set_frame_type_is_input_error() -> None:
    with pytest.raises(InputError, match="polars.DataFrame"):
        ParameterSet(frame="not-a-frame")  # type: ignore[arg-type]


def test_parameter_set_empty_frame_is_input_error() -> None:
    # was: "must have at least one parameter variable." (InputError)
    with pytest.raises(InputError, match="at least one parameter"):
        ParameterSet(frame=pl.DataFrame())


def test_parameter_set_wrong_dtype_and_extra_variable_fail_closed() -> None:
    with pytest.raises(InputError, match="Float64|extra"):
        ParameterSet(frame=pl.DataFrame({"alpha": [1, 2]}))
    with pytest.raises(InputError, match="Float64|extra"):
        ParameterSet.from_polars(
            pl.DataFrame({"alpha": [0.1, 0.2], "label": ["a", "b"]})
        )


def test_parameter_set_null_values_are_input_error() -> None:
    with pytest.raises(InputError, match="null"):
        ParameterSet.from_polars(pl.DataFrame({"alpha": [0.1, None]}))


def test_parameter_set_length_and_type_errors_keep_types() -> None:
    with pytest.raises(InputError, match="same length"):
        ParameterSet.from_numpy_or_dict(
            {"alpha": np.array([0.1, 0.2]), "beta": np.array([1.0])}
        )
    with pytest.raises(InputError, match="numpy array or dictionary"):
        ParameterSet.from_numpy_or_dict("bad-input")
    with pytest.raises(InputError, match="cannot be empty"):
        ParameterSet.from_numpy_or_dict({})


def test_parameter_set_ndim_error_keeps_type() -> None:
    with pytest.raises(InputError, match="2D"):
        ParameterSet.from_numpy_or_dict(np.array([0.1, 0.2]))


def test_value_array_perspective_names_rejected_for_2d() -> None:
    with pytest.raises(InputError, match="perspective_names"):
        ValueArray.from_polars(
            pl.DataFrame({"A": [1.0, 2.0]}), perspective_names=["p1"]
        )
