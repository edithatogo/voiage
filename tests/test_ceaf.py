"""Tests for cost-effectiveness acceptability frontier calculations."""

from typing import cast

import numpy as np
import pytest
import xarray as xr

from voiage.exceptions import InputError
from voiage.methods.ceaf import CEAFResult, calculate_ceaf
from voiage.schema import ValueArray


@pytest.fixture
def ceaf_value_array() -> ValueArray:
    """Create net benefits with known CEAF probabilities."""
    nb_values = np.array(
        [
            [[5.0, 5.0, 1.0], [0.0, 2.0, 0.0]],
            [[5.0, 1.0, 1.0], [0.0, 2.0, 2.0]],
            [[5.0, 1.0, 1.0], [0.0, 2.0, 2.0]],
            [[5.0, 1.0, 1.0], [0.0, 2.0, 2.0]],
        ],
    )
    return ValueArray(
        dataset=xr.Dataset(
            {"net_benefit": (("n_samples", "n_strategies", "n_wtp"), nb_values)},
            coords={
                "n_samples": np.arange(4),
                "n_strategies": np.arange(2),
                "n_wtp": np.arange(3),
                "strategy": ("n_strategies", ["A", "B"]),
            },
        )
    )


def test_calculate_ceaf_identifies_expected_optimal_frontier(
    ceaf_value_array: ValueArray,
) -> None:
    result = calculate_ceaf(ceaf_value_array, [0.0, 50.0, 100.0])

    assert isinstance(result, CEAFResult)
    assert result.optimal_strategy_indices.tolist() == [0, 0, 1]
    assert result.optimal_strategy_names == ["A", "A", "B"]
    assert result.acceptability_probabilities.tolist() == pytest.approx(
        [1.0, 0.25, 0.75]
    )
    assert result.expected_net_benefit.tolist() == pytest.approx([5.0, 2.0, 1.5])
    assert np.all(result.probability_lower >= 0)
    assert np.all(result.probability_upper <= 1)
    assert result.reporting["reporting_standard"] == "CHEERS-VOI"
    assert result.reporting["analysis_type"] == "calculate_ceaf"


def test_calculate_ceaf_converges_to_one_with_resolved_uncertainty() -> None:
    nb_values = np.array(
        [
            [[10.0, 5.0], [0.0, 1.0]],
            [[10.0, 5.0], [0.0, 1.0]],
            [[10.0, 5.0], [0.0, 1.0]],
        ]
    )
    value_array = ValueArray.from_numpy(np.ones((3, 2)), ["A", "B"])
    value_array = ValueArray(
        dataset=xr.Dataset(
            {"net_benefit": (("n_samples", "n_strategies", "n_wtp"), nb_values)},
            coords={
                "n_samples": np.arange(3),
                "n_strategies": np.arange(2),
                "n_wtp": np.arange(2),
                "strategy": ("n_strategies", value_array.strategy_names),
            },
        )
    )

    result = calculate_ceaf(value_array, [0.0, 1.0])

    assert result.optimal_strategy_names == ["A", "A"]
    assert result.acceptability_probabilities.tolist() == [1.0, 1.0]


def test_calculate_ceaf_rejects_invalid_inputs(ceaf_value_array: ValueArray) -> None:
    with pytest.raises(InputError, match="ValueArray"):
        calculate_ceaf(cast("ValueArray", "not a value array"), [0.0])

    with pytest.raises(InputError, match="3D"):
        calculate_ceaf(ValueArray.from_numpy(np.ones((3, 2))), [0.0])

    with pytest.raises(InputError, match="1D"):
        calculate_ceaf(ceaf_value_array, np.ones((1, 3)))

    with pytest.raises(InputError, match="must match the third dimension"):
        calculate_ceaf(ceaf_value_array, [0.0, 1.0])

    with pytest.raises(InputError, match="strategy_names"):
        calculate_ceaf(ceaf_value_array, [0.0, 1.0, 2.0], strategy_names=["A"])

    with pytest.raises(InputError, match="confidence_level"):
        calculate_ceaf(ceaf_value_array, [0.0, 1.0, 2.0], confidence_level=1.0)
