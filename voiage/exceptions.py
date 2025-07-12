# voiage/exceptions.py

"""
Custom exception classes for the voiage library.

Defining custom exceptions allows users of the library to catch specific errors
that originate from voiage and handle them appropriately, rather than relying on
generic Python exceptions.
"""


class voiageError(Exception):
    """Base class for all exceptions raised by the voiage library."""

    pass


# --- Configuration Errors ---
class ConfigurationError(voiageError):
    """Raised when there is an issue with voiage configuration."""

    pass


# --- Input Validation Errors ---
class InputError(voiageError, ValueError):
    """Raised when input data is invalid or inconsistent."""

    pass


class DimensionMismatchError(InputError):
    """Raised when array dimensions are incompatible for an operation."""

    pass


class MissingParameterError(InputError):
    """Raised when a required parameter is missing."""

    pass


# --- Calculation Errors ---
class CalculationError(voiageError):
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
class DataStructureError(voiageError):
    """Raised for issues related to voiage's internal data structures."""

    pass


# --- Plotting Errors ---
class PlottingError(voiageError):
    """Raised if an error occurs during plot generation."""

    pass


# --- Backend Errors ---
class BackendError(voiageError):
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


class VoiageNotImplementedError(voiageError, NotImplementedError):
    """Raised when a feature or method is not yet implemented."""

    pass


class OptionalDependencyError(voiageError, ImportError):
    """Raised when an optional dependency is not installed but is required for a feature."""

    pass
