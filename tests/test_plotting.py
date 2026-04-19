# tests/test_plotting.py

import numpy as np
import pytest
import xarray as xr

from voiage.exceptions import InputError
from voiage.plot import ceac as ceac_module
from voiage.plot import voi_curves as voi_curves_module
from voiage.plot.ceac import plot_ceac
from voiage.plot.voi_curves import (
    plot_evpi_vs_wtp,
    plot_evppi_surface,
    plot_evsi_vs_sample_size,
)
from voiage.schema import ValueArray


@pytest.fixture
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


def test_plot_ceac(sample_value_array) -> None:
    plot_ceac(sample_value_array, wtp_thresholds=np.linspace(0, 100000, 10))


def test_plot_ceac_accepts_per_strategy_plot_kwargs(sample_value_array) -> None:
    ax = plot_ceac(
        sample_value_array,
        wtp_thresholds=np.linspace(0, 100000, 10),
        plot_kwargs_per_strategy=[
            {"linestyle": "--", "label": "Custom 1"},
            {"linestyle": ":", "color": "black", "label": "Custom 2"},
        ],
    )

    assert [line.get_linestyle() for line in ax.lines] == ["--", ":"]
    assert [line.get_label() for line in ax.lines] == ["Custom 1", "Custom 2"]


def test_plot_ceac_rejects_non_3d_inputs() -> None:
    with pytest.raises(InputError, match="must have a 'n_strategies' dimension"):
        plot_ceac(
            ValueArray(dataset=xr.Dataset({"net_benefit": (("n_samples", "n_wtp"), np.ones((5, 4)))})),
            wtp_thresholds=np.linspace(0, 100000, 4),
        )


def test_plot_ceac_rejects_wtp_mismatch(sample_value_array) -> None:
    with pytest.raises(InputError, match="must match the third dimension"):
        plot_ceac(sample_value_array, wtp_thresholds=np.linspace(0, 100000, 9))


def test_plot_ceac_rejects_strategy_name_mismatch(sample_value_array) -> None:
    with pytest.raises(InputError, match="must match the second dimension"):
        plot_ceac(
            sample_value_array,
            wtp_thresholds=np.linspace(0, 100000, 10),
            strategy_names=["Only one"],
        )


def test_plot_ceac_rejects_non_list_plot_kwargs(sample_value_array) -> None:
    with pytest.raises(InputError, match="plot_kwargs_per_strategy must be a list"):
        plot_ceac(
            sample_value_array,
            wtp_thresholds=np.linspace(0, 100000, 10),
            plot_kwargs_per_strategy={"linestyle": "--"},
        )


def test_plot_evpi_vs_wtp() -> None:
    plot_evpi_vs_wtp(
        evpi_values=np.random.rand(10), wtp_thresholds=np.linspace(0, 100000, 10)
    )


def test_plot_evpi_vs_wtp_accepts_empty_inputs() -> None:
    ax = plot_evpi_vs_wtp(evpi_values=np.array([]), wtp_thresholds=np.array([]))

    assert len(ax.lines) == 1
    assert ax.lines[0].get_xdata().size == 0
    assert ax.lines[0].get_ydata().size == 0


def test_plot_evpi_vs_wtp_rejects_non_1d_inputs() -> None:
    with pytest.raises(InputError, match="1-dimensional"):
        plot_evpi_vs_wtp(
            evpi_values=np.ones((2, 2)),
            wtp_thresholds=np.linspace(0, 100000, 4),
        )


def test_plot_evpi_vs_wtp_rejects_length_mismatch() -> None:
    with pytest.raises(InputError, match="must match length"):
        plot_evpi_vs_wtp(
            evpi_values=np.linspace(0, 1, 3),
            wtp_thresholds=np.linspace(0, 100000, 4),
        )


def test_plot_evsi_vs_sample_size() -> None:
    plot_evsi_vs_sample_size(evsi_values=np.random.rand(10), sample_sizes=np.arange(10))


def test_plot_evsi_vs_sample_size_with_optional_series() -> None:
    ax = plot_evsi_vs_sample_size(
        evsi_values=np.linspace(10, 100, 5),
        sample_sizes=np.arange(5),
        enbs_values=np.linspace(5, 50, 5),
        research_costs=np.linspace(1, 10, 5),
    )

    assert len(ax.lines) == 3
    assert [line.get_label() for line in ax.lines] == ["EVSI", "ENBS", "Research Cost"]


def test_plot_evsi_vs_sample_size_rejects_non_1d_inputs() -> None:
    with pytest.raises(InputError, match="1-dimensional"):
        plot_evsi_vs_sample_size(
            evsi_values=np.ones((2, 2)),
            sample_sizes=np.arange(4),
        )


def test_plot_evsi_vs_sample_size_rejects_length_mismatch() -> None:
    with pytest.raises(InputError, match="must match length"):
        plot_evsi_vs_sample_size(
            evsi_values=np.linspace(10, 20, 3),
            sample_sizes=np.arange(4),
        )


def test_plot_evsi_vs_sample_size_rejects_optional_series_length_mismatch() -> None:
    with pytest.raises(InputError, match="Length of enbs_values mismatch"):
        plot_evsi_vs_sample_size(
            evsi_values=np.linspace(10, 40, 4),
            sample_sizes=np.arange(4),
            enbs_values=np.linspace(1, 2, 3),
        )

    with pytest.raises(InputError, match="Length of research_costs mismatch"):
        plot_evsi_vs_sample_size(
            evsi_values=np.linspace(10, 40, 4),
            sample_sizes=np.arange(4),
            research_costs=np.linspace(1, 2, 3),
        )


def test_plot_evppi_surface() -> None:
    evppi_values = np.random.rand(5, 10)
    plot_evppi_surface(
        evppi_values,
        param_names=["param1", "param2", "param3", "param4", "param5"],
        wtp_thresholds=np.linspace(0, 100000, 10),
    )


def test_plot_evppi_surface_rejects_non_2d_inputs() -> None:
    with pytest.raises(InputError, match="must be a 2D array"):
        plot_evppi_surface(
            evppi_values=np.linspace(0, 1, 5),
            param_names=["param1"],
            wtp_thresholds=np.linspace(0, 100000, 5),
        )


def test_plot_evppi_surface_rejects_parameter_name_mismatch() -> None:
    with pytest.raises(InputError, match="Length of param_names"):
        plot_evppi_surface(
            evppi_values=np.ones((2, 3)),
            param_names=["param1"],
            wtp_thresholds=np.linspace(0, 100000, 3),
        )


def test_plot_evppi_surface_rejects_wtp_mismatch() -> None:
    with pytest.raises(InputError, match="Length of wtp_thresholds"):
        plot_evppi_surface(
            evppi_values=np.ones((2, 3)),
            param_names=["param1", "param2"],
            wtp_thresholds=np.linspace(0, 100000, 2),
        )


@pytest.mark.parametrize(
    ("module", "func", "kwargs"),
    [
        (
            ceac_module,
            plot_ceac,
            {
                "value_array": ValueArray(
                    dataset=xr.Dataset(
                        {
                            "net_benefit": (
                                ("n_samples", "n_strategies", "n_wtp"),
                                np.ones((4, 2, 3)),
                            )
                        }
                    )
                ),
                "wtp_thresholds": np.linspace(0, 100000, 3),
            },
        ),
        (
            voi_curves_module,
            plot_evpi_vs_wtp,
            {
                "evpi_values": np.linspace(0, 1, 3),
                "wtp_thresholds": np.linspace(0, 100000, 3),
            },
        ),
        (
            voi_curves_module,
            plot_evsi_vs_sample_size,
            {
                "evsi_values": np.linspace(0, 1, 3),
                "sample_sizes": np.arange(3),
            },
        ),
        (
            voi_curves_module,
            plot_evppi_surface,
            {
                "evppi_values": np.ones((2, 3)),
                "param_names": ["p1", "p2"],
                "wtp_thresholds": np.linspace(0, 100000, 3),
            },
        ),
    ],
)
def test_plotting_functions_raise_when_matplotlib_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    module,
    func,
    kwargs,
) -> None:
    monkeypatch.setattr(module, "MATPLOTLIB_AVAILABLE", False)

    with pytest.raises(Exception, match="Matplotlib is required"):
        func(**kwargs)
