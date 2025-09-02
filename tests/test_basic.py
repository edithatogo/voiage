"""Unit tests for the basic VOI methods (EVPI, EVPPI) in voiage.methods.basic."""

import numpy as np
import pytest
import xarray as xr

from voiage.exceptions import (
    InputError,
)
from voiage.methods.basic import evpi, evppi
from voiage.schema import ParameterSet, ValueArray

# A simple, known scenario for testing
# Strategies: A, B
# Samples: 5
NB_A = np.array([100, 150, 120, 80, 90], dtype=np.float64)
NB_B = np.array([110, 140, 130, 100, 95], dtype=np.float64)
NB_ARRAY = np.stack([NB_A, NB_B], axis=1)

# EV with PI = mean(110, 150, 130, 100, 95) = 117
# EV with CI = max(mean(NB_A), mean(NB_B)) = max(108, 115) = 115
# EVPI = 117 - 115 = 2
EXPECTED_EVPI = 2.0


@pytest.fixture()
def sample_value_array() -> ValueArray:
    """Provide a ValueArray instance."""
    dataset = xr.Dataset(
        {"net_benefit": (("n_samples", "n_strategies"), NB_ARRAY)},
        coords={
            "n_samples": np.arange(NB_ARRAY.shape[0]),
            "n_strategies": np.arange(NB_ARRAY.shape[1]),
            "strategy": ("n_strategies", ["Strategy A", "Strategy B"]),
        },
    )
    return ValueArray(dataset=dataset)


@pytest.fixture()
def sample_parameter_set() -> ParameterSet:
    """Provide a ParameterSet instance."""
    params = {
        "param1": np.array([1, 2, 3, 4, 5]),
        "param2": np.array([10, 20, 30, 40, 50]),
    }
    dataset = xr.Dataset(
        {k: (("n_samples",), v) for k, v in params.items()},
        coords={"n_samples": np.arange(5)},
    )
    return ParameterSet(dataset=dataset)


def test_evpi_calculation(sample_value_array: ValueArray):
    """Test basic EVPI calculation."""
    calculated_evpi = evpi(sample_value_array)
    np.testing.assert_allclose(calculated_evpi, EXPECTED_EVPI, rtol=1e-5)


def test_evpi_population(sample_value_array: ValueArray):
    """Test population-level EVPI calculation."""
    pop_evpi = evpi(
        sample_value_array, population=1000, time_horizon=10, discount_rate=0.03
    )
    annuity = (1 - (1 + 0.03) ** -10) / 0.03
    expected_pop_evpi = EXPECTED_EVPI * 1000 * annuity
    np.testing.assert_allclose(pop_evpi, expected_pop_evpi, rtol=1e-5)


def test_evppi_calculation(
    sample_value_array: ValueArray, sample_parameter_set: ParameterSet
):
    """Test basic EVPPI calculation (smoke test)."""
    calculated_evppi = evppi(sample_value_array, sample_parameter_set, ["param1"])
    assert isinstance(calculated_evppi, float)
    assert calculated_evppi >= 0
    assert calculated_evppi <= evpi(sample_value_array) + 1e-9


def test_evppi_invalid_inputs(
    sample_value_array: ValueArray, sample_parameter_set: ParameterSet
):
    """Test EVPPI with various invalid inputs."""
    with pytest.raises(
        InputError, match="All `parameters_of_interest` must be in the ParameterSet"
    ):
        evppi(sample_value_array, sample_parameter_set, ["non_existent_param"])

    with pytest.raises(InputError, match="cannot exceed total samples"):
        evppi(
            sample_value_array,
            sample_parameter_set,
            ["param1"],
            n_regression_samples=10,
        )

    with pytest.raises(TypeError):
        evppi("not a value array", sample_parameter_set, ["param1"])

    with pytest.raises(TypeError):
        evppi(sample_value_array, "not a parameter set", ["param1"])
