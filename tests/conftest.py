# tests/conftest.py

"""
Configuration for pytest.

This file can be used to define fixtures, hooks, and plugins that are
shared across multiple test files.
"""

import numpy as np
import pytest
import xarray as xr

from voiage.schema import ParameterSet, ValueArray


@pytest.fixture(scope="session")
def sample_value_array() -> ValueArray:
    """Provide a ValueArray instance."""
    data = np.array(
        [[100, 105], [110, 100], [90, 110], [120, 100], [95, 115]],
        dtype=np.float64,
    )
    dataset = xr.Dataset(
        {"value": (("n_samples", "n_options"), data)},
        coords={
            "n_samples": np.arange(data.shape[0]),
            "n_options": np.arange(data.shape[1]),
            "option": ("n_options", ["Strategy A", "Strategy B"]),
        },
    )
    return ValueArray(dataset=dataset)


@pytest.fixture(scope="session")
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
