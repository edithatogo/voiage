"""Numerical parity for the container swap (AC4).

The container change must be a representation change, not a numerical one:
every reference case must round-trip to a bit-identical NumPy array, column
order must be preserved, and the ``specs/core-api`` example fixtures must
produce identical output. This module pins those guarantees so a regression
cannot silently widen a tolerance or reorder columns.
"""

from __future__ import annotations

import numpy as np
import pytest

from voiage.schema import ParameterSet, ValueArray


def test_value_array_from_numpy_is_bit_identical() -> None:
    rng = np.random.default_rng(20261026)
    values = rng.standard_normal((128, 5))
    value_array = ValueArray.from_numpy(values, strategy_names=list("ABCDE"))

    assert value_array.strategy_names == list("ABCDE")
    assert np.array_equal(value_array.numpy_values, values)
    assert value_array.numpy_values.dtype == np.float64


def test_value_array_from_numpy_perspectives_is_bit_identical() -> None:
    rng = np.random.default_rng(7)
    values = rng.standard_normal((40, 4, 3))
    value_array = ValueArray.from_numpy_perspectives(
        values,
        strategy_names=["A", "B", "C", "D"],
        perspective_names=["payer", "societal", "nhs"],
    )

    assert value_array.strategy_names == ["A", "B", "C", "D"]
    assert value_array.perspective_names == ["payer", "societal", "nhs"]
    assert np.array_equal(value_array.numpy_values, values)


def test_parameter_set_from_numpy_or_dict_is_bit_identical() -> None:
    rng = np.random.default_rng(11)
    params = rng.standard_normal((128, 3))
    parameter_set = ParameterSet.from_numpy_or_dict(params)

    assert parameter_set.parameter_names == ["param_0", "param_1", "param_2"]
    for index, name in enumerate(parameter_set.parameter_names):
        assert np.array_equal(parameter_set.parameters[name], params[:, index])


def test_parameter_set_dict_preserves_declared_order() -> None:
    params = {
        "first": np.array([1.0, 2.0, 3.0]),
        "second": np.array([4.0, 5.0, 6.0]),
        "third": np.array([7.0, 8.0, 9.0]),
    }
    parameter_set = ParameterSet.from_numpy_or_dict(params)

    assert parameter_set.parameter_names == ["first", "second", "third"]
    for name, values in params.items():
        assert np.array_equal(parameter_set.parameters[name], values)


def test_specs_core_api_value_array_example_is_bit_identical() -> None:
    net_benefit = np.array([[1500.0, 1625.0], [1490.0, 1650.0], [1510.0, 1635.0]])
    value_array = ValueArray.from_numpy(
        net_benefit, strategy_names=["Usual care", "Targeted screening"]
    )

    assert value_array.strategy_names == ["Usual care", "Targeted screening"]
    assert value_array.n_samples == 3
    assert np.array_equal(
        value_array.numpy_values,
        np.array([[1500.0, 1625.0], [1490.0, 1650.0], [1510.0, 1635.0]]),
    )


def test_specs_core_api_parameter_set_example_is_bit_identical() -> None:
    parameters = {
        "prob_screen_positive": np.array([0.08, 0.1, 0.09]),
        "incremental_cost": np.array([120.0, 140.0, 130.0]),
        "incremental_effect": np.array([0.02, 0.03, 0.025]),
    }
    parameter_set = ParameterSet.from_numpy_or_dict(parameters)

    assert parameter_set.parameter_names == [
        "prob_screen_positive",
        "incremental_cost",
        "incremental_effect",
    ]
    for name, values in parameters.items():
        assert np.array_equal(parameter_set.parameters[name], values)


def test_slice_by_strategies_preserves_requested_column_order_and_values() -> None:
    rng = np.random.default_rng(3)
    values = rng.standard_normal((10, 4))
    value_array = ValueArray.from_numpy(values, strategy_names=["A", "B", "C", "D"])

    sliced = value_array.slice_by_strategies(["D", "B", "A"])

    assert sliced.strategy_names == ["D", "B", "A"]
    assert np.array_equal(
        sliced.numpy_values,
        np.column_stack([values[:, 3], values[:, 1], values[:, 0]]),
    )


def test_parameter_subset_preserves_requested_order_and_values() -> None:
    rng = np.random.default_rng(5)
    params = {name: rng.standard_normal(12) for name in ["w", "x", "y", "z"]}
    parameter_set = ParameterSet.from_numpy_or_dict(params)

    subset = parameter_set.subset_by_parameters(["z", "x"])

    assert subset.parameter_names == ["z", "x"]
    assert np.array_equal(subset.parameters["z"], params["z"])
    assert np.array_equal(subset.parameters["x"], params["x"])


@pytest.mark.parametrize("shape", [(1, 1), (1, 50), (50, 1), (300, 7)])
def test_value_array_edge_shapes_are_bit_identical(shape: tuple[int, int]) -> None:
    rng = np.random.default_rng(sum(shape))
    values = rng.standard_normal(shape)
    value_array = ValueArray.from_numpy(values)
    assert np.array_equal(value_array.numpy_values, values)
