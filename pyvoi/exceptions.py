# pyvoi/exceptions.py

"""
Custom exception classes for the pyVOI library.

Defining custom exceptions allows users of the library to catch specific errors
that originate from pyVOI and handle them appropriately, rather than relying on
generic Python exceptions.
"""


class PyVOIError(Exception):
    """Base class for all exceptions raised by the pyVOI library."""

    pass


# --- Configuration Errors ---
class ConfigurationError(PyVOIError):
    """Raised when there is an issue with pyVOI configuration."""

    pass


# --- Input Validation Errors ---
class InputError(PyVOIError, ValueError):
    """Raised when input data is invalid or inconsistent."""

    pass


class DimensionMismatchError(InputError):
    """Raised when array dimensions are incompatible for an operation."""

    pass


class MissingParameterError(InputError):
    """Raised when a required parameter is missing."""

    pass


# --- Calculation Errors ---
class CalculationError(PyVOIError):
    """Raised during a VOI calculation if an error occurs."""

    pass


class NonConvergenceError(CalculationError):
    """Raised if a numerical method (e.g., optimization, MCMC) fails to converge."""

    pass


class InvalidMethodError(CalculationError):
    """Raised if an invalid method or algorithm is specified for a calculation."""

    pass


# --- EVSI Specific Errors ---
class EVSIMethodError(CalculationError):
    """Base class for errors specific to EVSI calculations."""

    pass


class RegressionMetamodelError(EVSIMethodError):
    """Raised for errors during the regression metamodeling step in EVSI."""

    pass


# --- Data Structure Errors ---
class DataStructureError(PyVOIError):
    """Raised for issues related to pyVOI's internal data structures."""

    pass


# --- Plotting Errors ---
class PlottingError(PyVOIError):
    """Raised if an error occurs during plot generation."""

    pass


# --- Backend Errors ---
class BackendError(PyVOIError):
    """Raised for errors related to the computation backend (e.g., JAX, NumPy)."""

    pass


# Example usage:
# if not isinstance(param, expected_type):
#     raise InputError(f"Parameter '{param_name}' must be of type {expected_type}.")
#
# try:
#     # some calculation
# except SomeUnderlyingLibError as e:
#     raise CalculationError("Underlying library failed during calculation.") from e


class PyVoiNotImplementedError(PyVOIError):
    """Raised when a feature or method is not yet implemented."""

    pass


class OptionalDependencyError(PyVOIError, ImportError):
    """Raised when an optional dependency is not installed but is required for a feature."""

    pass
