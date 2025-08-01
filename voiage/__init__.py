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
from .analysis import DecisionAnalysis
from .core.data_structures import (
    DecisionOption,
    DynamicSpec,
    PortfolioSpec,
    PortfolioStudy,
    PSASample,
    TrialDesign,
    ValueArray,
)
from .methods.adaptive import adaptive_evsi

# --- Core VOI Methods ---
# Expose primary VOI calculation functions at the top level of the package
# for ease of use, e.g., `from voiage import evpi`.
from .methods.basic import evpi, evppi
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
    "DynamicSpec",
    "ValueArray",
    "PSASample",
    "PortfolioSpec",
    "PortfolioStudy",
    "DecisionOption",
    "TrialDesign",
    "DecisionAnalysis",
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
