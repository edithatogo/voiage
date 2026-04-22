"""Regression coverage for curated subpackage exports."""

from __future__ import annotations

import voiage
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
)
from voiage.core.io import (
    read_value_array_csv as read_value_array_csv_impl,
)
from voiage.core.io import (
    write_parameter_set_csv as write_parameter_set_csv_impl,
)
from voiage.core.io import (
    write_value_array_csv as write_value_array_csv_impl,
)
from voiage.core.utils import (
    calculate_net_benefit as calculate_net_benefit_impl,
)
from voiage.core.utils import (
    check_input_array as check_input_array_impl,
)
from voiage.methods import (
    enbs,
    evpi,
    evppi,
    evsi,
    portfolio_voi,
    sequential_voi,
    voi_calibration,
)
from voiage.methods.basic import evpi as evpi_impl
from voiage.methods.basic import evppi as evppi_impl
from voiage.methods.calibration import voi_calibration as voi_calibration_impl
from voiage.methods.portfolio import portfolio_voi as portfolio_voi_impl
from voiage.methods.sample_information import enbs as enbs_impl
from voiage.methods.sample_information import evsi as evsi_impl
from voiage.methods.sequential import sequential_voi as sequential_voi_impl
from voiage.plot import (
    plot_ceac,
    plot_evpi_vs_wtp,
    plot_evppi_surface,
    plot_evsi_vs_sample_size,
)
from voiage.plot.ceac import plot_ceac as plot_ceac_impl
from voiage.plot.voi_curves import (
    plot_evpi_vs_wtp as plot_evpi_vs_wtp_impl,
)
from voiage.plot.voi_curves import (
    plot_evppi_surface as plot_evppi_surface_impl,
)
from voiage.plot.voi_curves import (
    plot_evsi_vs_sample_size as plot_evsi_vs_sample_size_impl,
)
from voiage import analysis as analysis_module
from voiage import backends as backends_module
from voiage import cli as cli_module
from voiage import config as config_module
from voiage import core as core_module
from voiage import exceptions as exceptions_module
from voiage import factory as factory_module
from voiage import fluent as fluent_module
from voiage import health_economics as health_economics_module
from voiage import hta_integration as hta_integration_module
from voiage import methods as methods_module
from voiage import multi_domain as multi_domain_module
from voiage import plot as plot_module
from voiage import schema as schema_module


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


def test_top_level_package_exports_modules() -> None:
    """Top-level package exports should remain stable curated modules."""
    assert voiage.__all__ == [
        "analysis",
        "backends",
        "cli",
        "config",
        "core",
        "exceptions",
        "factory",
        "fluent",
        "health_economics",
        "hta_integration",
        "methods",
        "multi_domain",
        "plot",
        "schema",
    ]


def test_top_level_package_exports_point_to_modules() -> None:
    """Top-level package exports should remain stable module aliases."""
    assert voiage.analysis is analysis_module
    assert voiage.backends is backends_module
    assert voiage.cli is cli_module
    assert voiage.config is config_module
    assert voiage.core is core_module
    assert voiage.exceptions is exceptions_module
    assert voiage.factory is factory_module
    assert voiage.fluent is fluent_module
    assert voiage.health_economics is health_economics_module
    assert voiage.hta_integration is hta_integration_module
    assert voiage.methods is methods_module
    assert voiage.multi_domain is multi_domain_module
    assert voiage.plot is plot_module
    assert voiage.schema is schema_module
