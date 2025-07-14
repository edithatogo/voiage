# voiage/__init__.py

"""
voiage: A Python Library for Value of Information Analysis.

This library provides tools for calculating various Value of Information metrics,
aiding in decision-making under uncertainty, particularly in health economics
and other fields requiring structured decision analysis.
"""

__version__ = "0.1.0"  # Target for v0.1

# --- Core Data Structures ---
# Make key data structures easily accessible from `voiage.core` or directly `voiage`
# --- Plotting Utilities (Optional Top-Level Import) ---
# Users might prefer to import these explicitly, e.g., `from voiage.plot import plot_ceac`
# If they are very commonly used, they could be exposed here too.
# from .plot.ceac import plot_ceac, plot_ce_plane
# from .plot.voi_curves import plot_evpi_vs_wtp, plot_evsi_vs_sample_size
# --- Logging Setup ---
# Configure a default null logger to prevent "No handler found" warnings if the
# library is used by an application that doesn't configure logging.
# The application using voiage can then set up its own logging handlers.
import logging

# --- Configuration and Exceptions ---
from . import (
    config,  # Allow access like `voiage.config.DEFAULT_DTYPE`
    exceptions,  # Allow access like `voiage.exceptions.InputError`
)
from .schema import (
    DynamicSpec,
    ValueArray,
    PortfolioSpec,
    PortfolioStudy,
    ParameterSet,
    DecisionOption,
    TrialDesign,
)
from .analysis import DecisionAnalysis
from .methods.adaptive import adaptive_evsi

# --- Core VOI Methods ---
# Expose primary VOI calculation functions at the top level of the package
# for ease of use, e.g., `from voiage import evpi`.
from .methods.basic import evpi as evpi_func, evppi as evppi_func
from .analysis import DecisionAnalysis
from .schema import ParameterSet, ValueArray
import numpy as np
from typing import Any, Dict, Optional, Union


from .backends import get_backend, set_backend
from .methods import jax_basic

def evpi(
    nb_array: Union[np.ndarray, ValueArray],
    population: Optional[float] = None,
    time_horizon: Optional[float] = None,
    discount_rate: Optional[float] = None,
) -> float:
    if get_backend() == "jax":
        if isinstance(nb_array, ValueArray):
            nb_array = nb_array.values
        return jax_basic.evpi(
            nb_array,
            population=population,
            time_horizon=time_horizon,
            discount_rate=discount_rate,
        )
    else:
        if isinstance(nb_array, np.ndarray):
            nb_array = ValueArray(values=nb_array)
        analysis = DecisionAnalysis(parameters=None, values=nb_array)
        return analysis.evpi(
            population=population,
            time_horizon=time_horizon,
            discount_rate=discount_rate,
        )


def evppi(
    nb_array: Union[np.ndarray, ValueArray],
    parameter_samples: Union[np.ndarray, ParameterSet, Dict[str, np.ndarray]],
    population: Optional[float] = None,
    time_horizon: Optional[float] = None,
    discount_rate: Optional[float] = None,
    n_regression_samples: Optional[int] = None,
    regression_model: Optional[Any] = None,
) -> float:
    if isinstance(nb_array, np.ndarray):
        nb_array = ValueArray(values=nb_array)
    if isinstance(parameter_samples, (np.ndarray, dict)):
        parameter_samples = ParameterSet(parameters=parameter_samples)
    analysis = DecisionAnalysis(parameters=parameter_samples, values=nb_array)
    return analysis.evppi(
        parameter_samples=parameter_samples,
        population=population,
        time_horizon=time_horizon,
        discount_rate=discount_rate,
        n_regression_samples=n_regression_samples,
        regression_model=regression_model,
    )
from .methods.calibration import voi_calibration
from .methods.network_nma import evsi_nma
from .methods.observational import voi_observational
from .methods.portfolio import portfolio_voi
from .methods.sample_information import enbs, evsi
from .methods.sequential import sequential_voi

# Placeholder imports for advanced methods (they will raise NotImplementedError if called)
from .methods.structural import structural_evpi, structural_evppi

logging.getLogger(__name__).addHandler(logging.NullHandler())


# --- __all__ for `from voiage import *` ---
# Define what gets imported by `from voiage import *`.
# It's generally good practice to be explicit.
__all__ = [
    "DecisionAnalysis",
    "DynamicSpec",
    "ValueArray",
    "ParameterSet",
    "PortfolioSpec",
    "PortfolioStudy",
    "DecisionOption",
    "TrialDesign",
    "__version__",
    "adaptive_evsi",
    "cli",
    "config",
    "core",
    "enbs",
    "evpi",
    "evppi",
    "evsi",
    "evsi_nma",
    "exceptions",
    "methods",
    "plot",
    "portfolio_voi",
    "sequential_voi",
    "structural_evpi",
    "structural_evppi",
    "voi_calibration",
    "voi_observational",
]

# Print a message if run directly (e.g., python -m voiage)
# This part is optional and usually for packages with a runnable __main__.py
# if __name__ == "__main__": # pragma: no cover
#     print(f"voiage package version: {__version__}")
#     print("This is a library. Import it in your Python scripts to use its functionalities.")
#     # You could add a simple CLI dispatch here if desired, but cli.py is preferred for that.
