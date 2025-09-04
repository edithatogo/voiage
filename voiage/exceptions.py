# voiage/exceptions.py

"""
Custom exception classes for the voiage library.

Defining custom exceptions allows users of the library to catch specific errors
that originate from voiage and handle them appropriately, rather than relying on
generic Python exceptions.
"""


class VoiageError(Exception):
    """Base class for all exceptions raised by the voiage library."""

    pass


# --- Configuration Errors ---
class ConfigurationError(VoiageError):
    """Raised when there is an issue with voiage configuration."""

    pass


# --- Input Validation Errors ---
class InputError(VoiageError, ValueError):
    """Raised when input data is invalid or inconsistent."""

    pass


class DimensionMismatchError(InputError):
    """Raised when array dimensions are incompatible for an operation."""

    pass


class MissingParameterError(InputError):
    """Raised when a required parameter is missing."""

    pass


class DataFormatError(InputError):
    """Raised when input data is in an incorrect format."""

    pass


# --- Calculation Errors ---
class CalculationError(VoiageError):
    """Raised during a VOI calculation if an error occurs."""

    pass


class NonConvergenceError(CalculationError):
    """Raised if a numerical method (e.g., optimization, MCMC) fails to converge."""

    pass


class InvalidMethodError(CalculationError):
    """Raised if an invalid method or algorithm is specified for a calculation."""

    pass


class ConvergenceError(CalculationError):
    """Raised when an algorithm fails to converge within specified tolerances."""

    pass


# --- EVSI Specific Errors ---
class EVSIMethodError(CalculationError):
    """Base class for errors specific to EVSI calculations."""

    pass


class RegressionMetamodelError(EVSIMethodError):
    """Raised for errors during the regression metamodeling step in EVSI."""

    pass


# --- Data Structure Errors ---
class DataStructureError(VoiageError):
    """Raised for issues related to voiage's internal data structures."""

    pass


# --- Plotting Errors ---
class PlottingError(VoiageError):
    """Raised if an error occurs during plot generation."""

    pass


# --- Backend Errors ---
class BackendError(VoiageError):
    """Raised for errors related to the computation backend (e.g., JAX, NumPy)."""

    pass


# --- Metamodel Errors ---
class MetamodelError(VoiageError):
    """Base class for errors related to metamodeling."""

    pass


class MetamodelFitError(MetamodelError):
    """Raised when a metamodel fails to fit the data."""

    pass


# Example usage:
# if not isinstance(param, expected_type):
#     raise InputError(f"Parameter '{param_name}' must be of type {expected_type}.")
#
# try:
#     # some calculation
# except SomeUnderlyingLibError as e:
#     raise CalculationError("Underlying library failed during calculation.") from e


class VoiageNotImplementedError(VoiageError, NotImplementedError):
    """Raised when a feature or method is not yet implemented."""

    pass


class OptionalDependencyError(VoiageError, ImportError):
    """Raised when an optional dependency is not installed but is required for a feature."""

    pass
