# tests/test_plotting.py

import numpy as np
import pytest
import xarray as xr

from voiage.plot.ceac import plot_ceac
from voiage.plot.voi_curves import (
    plot_evpi_vs_wtp,
    plot_evppi_surface,
    plot_evsi_vs_sample_size,
)
from voiage.schema import ValueArray


@pytest.fixture()
def sample_value_array():
    data = {
        "net_benefit": (
            ("n_samples", "n_strategies", "n_wtp"),
            np.random.rand(100, 2, 10),
        ),
    }
    return ValueArray(
        dataset=xr.Dataset(data, coords={"strategy": ["Strategy 1", "Strategy 2"]})
    )


def test_plot_ceac(sample_value_array):
    plot_ceac(sample_value_array, wtp_thresholds=np.linspace(0, 100000, 10))


def test_plot_evpi_vs_wtp():
    plot_evpi_vs_wtp(
        evpi_values=np.random.rand(10), wtp_thresholds=np.linspace(0, 100000, 10)
    )


def test_plot_evsi_vs_sample_size():
    plot_evsi_vs_sample_size(evsi_values=np.random.rand(10), sample_sizes=np.arange(10))


def test_plot_evppi_surface():
    evppi_values = np.random.rand(5, 10)
    plot_evppi_surface(
        evppi_values,
        param_names=["param1", "param2", "param3", "param4", "param5"],
        wtp_thresholds=np.linspace(0, 100000, 10),
    )
