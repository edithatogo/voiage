"""Regression coverage for curated subpackage exports."""

from __future__ import annotations

from voiage.core import (
    calculate_net_benefit,
    check_input_array,
    read_parameter_set_csv,
    read_value_array_csv,
    write_parameter_set_csv,
    write_value_array_csv,
)
from voiage.core.io import (
    read_parameter_set_csv as read_parameter_set_csv_impl,
    read_value_array_csv as read_value_array_csv_impl,
    write_parameter_set_csv as write_parameter_set_csv_impl,
    write_value_array_csv as write_value_array_csv_impl,
)
from voiage.core.utils import (
    calculate_net_benefit as calculate_net_benefit_impl,
    check_input_array as check_input_array_impl,
)
from voiage.methods import enbs, evpi, evppi, evsi, portfolio_voi, sequential_voi, voi_calibration
from voiage.methods.basic import evpi as evpi_impl, evppi as evppi_impl
from voiage.methods.calibration import voi_calibration as voi_calibration_impl
from voiage.methods.portfolio import portfolio_voi as portfolio_voi_impl
from voiage.methods.sample_information import enbs as enbs_impl, evsi as evsi_impl
from voiage.methods.sequential import sequential_voi as sequential_voi_impl
from voiage.plot import plot_ceac, plot_evpi_vs_wtp, plot_evppi_surface, plot_evsi_vs_sample_size
from voiage.plot.ceac import plot_ceac as plot_ceac_impl
from voiage.plot.voi_curves import (
    plot_evpi_vs_wtp as plot_evpi_vs_wtp_impl,
    plot_evppi_surface as plot_evppi_surface_impl,
    plot_evsi_vs_sample_size as plot_evsi_vs_sample_size_impl,
)


def test_core_package_exports_point_to_leaf_implementations() -> None:
    """Core package exports should remain stable curated aliases."""
    assert calculate_net_benefit is calculate_net_benefit_impl
    assert check_input_array is check_input_array_impl
    assert read_parameter_set_csv is read_parameter_set_csv_impl
    assert read_value_array_csv is read_value_array_csv_impl
    assert write_parameter_set_csv is write_parameter_set_csv_impl
    assert write_value_array_csv is write_value_array_csv_impl


def test_methods_package_exports_point_to_leaf_implementations() -> None:
    """Method package exports should remain stable curated aliases."""
    assert enbs is enbs_impl
    assert evpi is evpi_impl
    assert evppi is evppi_impl
    assert evsi is evsi_impl
    assert portfolio_voi is portfolio_voi_impl
    assert sequential_voi is sequential_voi_impl
    assert voi_calibration is voi_calibration_impl


def test_plot_package_exports_point_to_leaf_implementations() -> None:
    """Plot package exports should remain stable curated aliases."""
    assert plot_ceac is plot_ceac_impl
    assert plot_evpi_vs_wtp is plot_evpi_vs_wtp_impl
    assert plot_evppi_surface is plot_evppi_surface_impl
    assert plot_evsi_vs_sample_size is plot_evsi_vs_sample_size_impl
