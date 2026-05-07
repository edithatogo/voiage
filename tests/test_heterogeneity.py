"""Tests for Value of Heterogeneity calculations."""

from typing import cast

import numpy as np
import pytest
import xarray as xr

from voiage.exceptions import InputError
from voiage.methods.heterogeneity import (
    HeterogeneityResult,
    identify_optimal_subgroups,
    value_of_heterogeneity,
)
from voiage.schema import ValueArray


@pytest.fixture
def heterogeneity_value_array() -> ValueArray:
    values = np.array(
        [
            [10.0, 0.0],
            [8.0, 2.0],
            [0.0, 12.0],
            [1.0, 10.0],
        ]
    )
    return ValueArray.from_numpy(values, ["A", "B"])


def test_value_of_heterogeneity_identifies_subgroup_optimal_decisions(
    heterogeneity_value_array: ValueArray,
) -> None:
    result = value_of_heterogeneity(
        heterogeneity_value_array,
        subgroups=["low", "low", "high", "high"],
    )

    assert isinstance(result, HeterogeneityResult)
    assert result.value == pytest.approx(4.0)
    assert result.subgroup_labels == ["high", "low"]
    assert result.subgroup_weights.tolist() == pytest.approx([0.5, 0.5])
    assert result.subgroup_optimal_strategy_names == ["B", "A"]
    assert result.overall_optimal_strategy_name == "B"
    assert result.reporting["reporting_standard"] == "CHEERS-VOI"
    assert result.reporting["analysis_type"] == "value_of_heterogeneity"
    assert identify_optimal_subgroups(result) == {"high": "B", "low": "A"}


def test_value_of_heterogeneity_supports_numeric_binning(
    heterogeneity_value_array: ValueArray,
) -> None:
    result = value_of_heterogeneity(
        heterogeneity_value_array,
        subgroups=np.array([0.1, 0.2, 0.9, 1.0]),
        n_bins=2,
    )

    assert result.subgroup_labels == ["bin_1", "bin_2"]
    assert result.value == pytest.approx(4.0)


def test_value_of_heterogeneity_rejects_invalid_inputs(
    heterogeneity_value_array: ValueArray,
) -> None:
    three_dimensional_values = ValueArray(
        dataset=xr.Dataset(
            {
                "net_benefit": (
                    ("n_samples", "n_strategies", "n_wtp"),
                    np.ones((4, 2, 2)),
                )
            },
            coords={
                "n_samples": np.arange(4),
                "n_strategies": np.arange(2),
                "strategy": ("n_strategies", ["A", "B"]),
            },
        )
    )

    with pytest.raises(InputError, match="ValueArray"):
        value_of_heterogeneity(cast("ValueArray", "not a value array"), ["a"])

    with pytest.raises(InputError, match="2D"):
        value_of_heterogeneity(
            three_dimensional_values,
            ["a", "a", "b", "b"],
        )

    with pytest.raises(InputError, match="length"):
        value_of_heterogeneity(heterogeneity_value_array, ["a"])

    with pytest.raises(InputError, match="strategy_names"):
        value_of_heterogeneity(
            heterogeneity_value_array,
            ["a", "a", "b", "b"],
            strategy_names=["A"],
        )

    with pytest.raises(InputError, match="n_bins"):
        value_of_heterogeneity(heterogeneity_value_array, [1, 2, 3, 4], n_bins=1)

    with pytest.raises(InputError, match="numeric"):
        value_of_heterogeneity(
            heterogeneity_value_array, ["a", "a", "b", "b"], n_bins=2
        )
