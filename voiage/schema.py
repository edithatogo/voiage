# voiage/schema.py

"""
Core data structures for voiage.

These structures are designed to hold and manage data used in Value of Information
analyses. They leverage Python's dataclasses for type hinting and validation where
appropriate, and are intended to work seamlessly with NumPy and Pandas/xarray.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import xarray as xr

from voiage.exceptions import InputError

# Try to import JAX for array support
try:
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jnp = None


@dataclass(frozen=True)
class ValueArray:
    """A container for net benefit values from a PSA."""

    dataset: xr.Dataset

    def __post_init__(self: "ValueArray"):
        """Validate the dataset."""
        if not isinstance(self.dataset, xr.Dataset):
            raise InputError("ValueArray 'dataset' must be a xarray.Dataset.")
        if "n_samples" not in self.dataset.dims:
            raise InputError("ValueArray 'dataset' must have a 'n_samples' dimension.")
        if "n_strategies" not in self.dataset.dims:
            raise InputError(
                "ValueArray 'dataset' must have a 'n_strategies' dimension."
            )
        if "net_benefit" not in self.dataset.data_vars:
            raise InputError(
                "ValueArray 'dataset' must have a 'net_benefit' data variable."
            )

    @property
    def values(self: "ValueArray") -> np.ndarray:
        """Return the net benefit values."""
        return self.dataset["net_benefit"].values

    @property
    def jax_values(self: "ValueArray") -> Optional["jnp.ndarray"]:
        """Return the net benefit values as a JAX array."""
        if not JAX_AVAILABLE:
            return None
        return jnp.asarray(self.dataset["net_benefit"].values, dtype=jnp.float64)

    @property
    def n_samples(self: "ValueArray") -> int:
        """Return the number of samples."""
        return self.dataset.dims["n_samples"]

    @property
    def n_strategies(self: "ValueArray") -> int:
        """Return the number of strategies."""
        return self.dataset.dims["n_strategies"]

    @property
    def strategy_names(self: "ValueArray") -> List[str]:
        """Return the names of the strategies."""
        return [str(name) for name in self.dataset["strategy"].values]

    @classmethod
    def from_numpy(cls, values: Union[np.ndarray, "jnp.ndarray"], strategy_names: Optional[List[str]] = None) -> "ValueArray":
        """Create a ValueArray from a numpy or JAX array.

        Args:
            values: A 2D array of shape (n_samples, n_strategies). Supports both NumPy and JAX arrays.
            strategy_names: Optional list of strategy names

        Returns
        -------
            ValueArray: A new ValueArray instance
        """
        # Handle JAX arrays if available
        if JAX_AVAILABLE and hasattr(values, 'dtype') and hasattr(values, 'shape'):
            # This could be a JAX array - convert to numpy for xarray
            values = np.asarray(values)

        if values.ndim != 2:
            raise InputError("values must be a 2D array")

        n_samples, n_strategies = values.shape

        if strategy_names is None:
            strategy_names = [f"Strategy {i}" for i in range(n_strategies)]
        elif len(strategy_names) != n_strategies:
            raise InputError(f"strategy_names must have {n_strategies} elements")

        import xarray as xr
        dataset = xr.Dataset(
            {"net_benefit": (("n_samples", "n_strategies"), values)},
            coords={
                "n_samples": np.arange(n_samples),
                "n_strategies": np.arange(n_strategies),
                "strategy": ("n_strategies", strategy_names),
            }
        )
        return cls(dataset=dataset)

    @classmethod
    def from_jax(cls, values: "jnp.ndarray", strategy_names: Optional[List[str]] = None) -> "ValueArray":
        """Create a ValueArray from a JAX array.

        Args:
            values: A 2D JAX array of shape (n_samples, n_strategies)
            strategy_names: Optional list of strategy names

        Returns
        -------
            ValueArray: A new ValueArray instance
        """
        if not JAX_AVAILABLE:
            raise ImportError("JAX is not available. Please install JAX to use from_jax().")
        
        if not hasattr(values, 'shape'):
            raise InputError("values must be a JAX array with a shape attribute")

        if values.ndim != 2:
            raise InputError("values must be a 2D array")

        # Convert JAX array to NumPy for xarray compatibility
        numpy_values = np.asarray(values)

        n_samples, n_strategies = numpy_values.shape

        if strategy_names is None:
            strategy_names = [f"Strategy {i}" for i in range(n_strategies)]
        elif len(strategy_names) != n_strategies:
            raise InputError(f"strategy_names must have {n_strategies} elements")

        import xarray as xr
        dataset = xr.Dataset(
            {"net_benefit": (("n_samples", "n_strategies"), numpy_values)},
            coords={
                "n_samples": np.arange(n_samples),
                "n_strategies": np.arange(n_strategies),
                "strategy": ("n_strategies", strategy_names),
            }
        )
        return cls(dataset=dataset)




@dataclass(frozen=True)
class ParameterSet:
    """A container for parameter samples from a PSA."""

    dataset: xr.Dataset

    def __post_init__(self: "ParameterSet"):
        """Validate the dataset."""
        if not isinstance(self.dataset, xr.Dataset):
            raise InputError("ParameterSet 'dataset' must be a xarray.Dataset.")
        if "n_samples" not in self.dataset.dims:
            raise InputError(
                "ParameterSet 'dataset' must have a 'n_samples' dimension."
            )

    @property
    def parameters(self: "ParameterSet") -> Dict[str, np.ndarray]:
        """Return the parameter samples."""
        return {str(name): self.dataset[name].values for name in self.dataset.data_vars}

    @property
    def jax_parameters(self: "ParameterSet") -> Optional[Dict[str, "jnp.ndarray"]]:
        """Return the parameter samples as JAX arrays."""
        if not JAX_AVAILABLE:
            return None
        return {str(name): jnp.asarray(self.dataset[name].values, dtype=jnp.float64) for name in self.dataset.data_vars}

    @property
    def n_samples(self: "ParameterSet") -> int:
        """Return the number of samples."""
        return self.dataset.dims["n_samples"]

    @property
    def parameter_names(self: "ParameterSet") -> List[str]:
        """Return the names of the parameters."""
        return list(self.dataset.data_vars.keys())

    @classmethod
    def from_numpy_or_dict(cls, parameters: Union[np.ndarray, Dict[str, np.ndarray], "jnp.ndarray", Dict[str, "jnp.ndarray"]]) -> "ParameterSet":
        """Create a ParameterSet from a numpy/JAX array or dictionary.

        Args:
            parameters: Either a 2D array of shape (n_samples, n_parameters)
                       or a dictionary mapping parameter names to 1D arrays.
                       Supports both NumPy and JAX arrays.

        Returns
        -------
            ParameterSet: A new ParameterSet instance
        """
        import xarray as xr

        # Handle JAX arrays if available
        if JAX_AVAILABLE and hasattr(parameters, 'ndim') and hasattr(parameters, 'dtype'):
            # This could be a JAX array - convert to numpy for xarray
            parameters = np.asarray(parameters)

        if isinstance(parameters, np.ndarray):
            if parameters.ndim != 2:
                raise InputError("parameters array must be 2D")
            n_samples, n_parameters = parameters.shape
            # Create parameter names
            param_names = [f"param_{i}" for i in range(n_parameters)]
            # Create dataset
            data_vars = {name: (("n_samples",), parameters[:, i]) for i, name in enumerate(param_names)}
            dataset = xr.Dataset(
                data_vars,
                coords={"n_samples": np.arange(n_samples)}
            )
        elif isinstance(parameters, dict):
            if not parameters:
                raise InputError("parameters dictionary cannot be empty")
            # Check that all arrays have the same length
            lengths = []
            converted_params = {}
            for name, arr in parameters.items():
                # Convert JAX arrays to numpy if needed
                if JAX_AVAILABLE and hasattr(arr, 'dtype') and hasattr(arr, 'shape'):
                    arr = np.asarray(arr)
                lengths.append(len(arr))
                converted_params[name] = arr
            
            if len(set(lengths)) > 1:
                raise InputError("All parameter arrays must have the same length")
            n_samples = lengths[0]
            # Create dataset
            data_vars = {name: (("n_samples",), arr) for name, arr in converted_params.items()}
            dataset = xr.Dataset(
                data_vars,
                coords={"n_samples": np.arange(n_samples)}
            )
        else:
            raise InputError("parameters must be a numpy array or dictionary")

        return cls(dataset=dataset)

    @classmethod
    def from_jax(cls, parameters: Union["jnp.ndarray", Dict[str, "jnp.ndarray"]]) -> "ParameterSet":
        """Create a ParameterSet from a JAX array or dictionary.

        Args:
            parameters: Either a 2D JAX array of shape (n_samples, n_parameters)
                       or a dictionary mapping parameter names to 1D JAX arrays

        Returns
        -------
            ParameterSet: A new ParameterSet instance
        """
        if not JAX_AVAILABLE:
            raise ImportError("JAX is not available. Please install JAX to use from_jax().")

        import xarray as xr

        if jnp is not None and hasattr(parameters, 'ndim'):  # JAX array
            if parameters.ndim != 2:
                raise InputError("parameters array must be 2D")
            n_samples, n_parameters = parameters.shape
            # Create parameter names
            param_names = [f"param_{i}" for i in range(n_parameters)]
            # Convert JAX array to NumPy for xarray compatibility
            numpy_parameters = np.asarray(parameters)
            # Create dataset
            data_vars = {name: (("n_samples",), numpy_parameters[:, i]) for i, name in enumerate(param_names)}
            dataset = xr.Dataset(
                data_vars,
                coords={"n_samples": np.arange(n_samples)}
            )
        elif isinstance(parameters, dict):
            if not parameters:
                raise InputError("parameters dictionary cannot be empty")
            # Check that all arrays have the same length and convert to numpy
            lengths = []
            converted_params = {}
            for name, arr in parameters.items():
                if not hasattr(arr, 'shape'):
                    raise InputError(f"Parameter {name} must be a JAX array")
                lengths.append(len(arr))
                converted_params[name] = np.asarray(arr)
            
            if len(set(lengths)) > 1:
                raise InputError("All parameter arrays must have the same length")
            n_samples = lengths[0]
            # Create dataset
            data_vars = {name: (("n_samples",), arr) for name, arr in converted_params.items()}
            dataset = xr.Dataset(
                data_vars,
                coords={"n_samples": np.arange(n_samples)}
            )
        else:
            raise InputError("parameters must be a JAX array or dictionary")

        return cls(dataset=dataset)




@dataclass(frozen=True)
class DecisionOption:
    """Represents a single arm in a clinical trial design.

    Attributes
    ----------
    name : str
        The name of the trial arm (e.g., "Treatment A", "Placebo").
    sample_size : int
        The number of subjects to be allocated to this arm.

    Raises
    ------
    InputError
        If `name` is not a non-empty string or `sample_size` is not a
        positive integer.
    """

    name: str
    sample_size: int

    def __post_init__(self: "DecisionOption"):
        """Validate the decision option."""
        if not isinstance(self.name, str) or not self.name:
            raise InputError("DecisionOption 'name' must be a non-empty string.")
        if not isinstance(self.sample_size, int) or self.sample_size <= 0:
            raise InputError("DecisionOption 'sample_size' must be a positive integer.")


@dataclass(frozen=True)
class TrialDesign:
    """Specifies the design of a proposed trial for EVSI calculations.

    Attributes
    ----------
    arms : List[DecisionOption]
        A list of `DecisionOption` objects that together define the trial.

    Raises
    ------
    InputError
        If `arms` is not a non-empty list of `DecisionOption` objects, or if
        any of the arm names are duplicated.
    """

    arms: List[DecisionOption]

    def __post_init__(self: "TrialDesign"):
        """Validate the trial design."""
        if not isinstance(self.arms, list) or not self.arms:
            raise InputError(
                "TrialDesign 'arms' must be a non-empty list of DecisionOption objects."
            )
        if not all(isinstance(arm, DecisionOption) for arm in self.arms):
            raise InputError("All elements in 'arms' must be DecisionOption objects.")
        arm_names = [arm.name for arm in self.arms]
        if len(arm_names) != len(set(arm_names)):
            raise InputError(
                "DecisionOption names within a TrialDesign must be unique."
            )

    @property
    def total_sample_size(self: "TrialDesign") -> int:
        """Return the total sample size across all arms."""
        return sum(arm.sample_size for arm in self.arms)


@dataclass(frozen=True)
class PortfolioStudy:
    """Represents a single candidate study within a research portfolio.

    Attributes
    ----------
    name : str
        The name of the candidate study.
    design : TrialDesign
        The `TrialDesign` object specifying the study's design.
    cost : float
        The estimated cost of conducting this study.

    Raises
    ------
    InputError
        If inputs are of the wrong type or `cost` is negative.
    """

    name: str
    design: TrialDesign
    cost: float

    def __post_init__(self: "PortfolioStudy"):
        """Validate the portfolio study."""
        if not isinstance(self.name, str) or not self.name:
            raise InputError("PortfolioStudy 'name' must be a non-empty string.")
        if not isinstance(self.design, TrialDesign):
            raise InputError("PortfolioStudy 'design' must be a TrialDesign object.")
        if not isinstance(self.cost, (int, float)) or self.cost < 0:
            raise InputError("PortfolioStudy 'cost' must be a non-negative number.")


@dataclass(frozen=True)
class PortfolioSpec:
    """Defines a portfolio of candidate research studies for optimization.

    Attributes
    ----------
    studies : List[PortfolioStudy]
        A list of `PortfolioStudy` objects representing the candidate studies.
    budget_constraint : Optional[float], optional
        The overall budget limit for the portfolio. Defaults to None.

    Raises
    ------
    InputError
        If `studies` is not a non-empty list of `PortfolioStudy` objects,
        if study names are duplicated, or if `budget_constraint` is negative.
    """

    studies: List[PortfolioStudy]
    budget_constraint: Optional[float] = None

    def __post_init__(self: "PortfolioSpec"):
        """Validate the portfolio spec."""
        if not isinstance(self.studies, list) or not self.studies:
            raise InputError(
                "PortfolioSpec 'studies' must be a non-empty list of PortfolioStudy objects."
            )
        if not all(isinstance(study, PortfolioStudy) for study in self.studies):
            raise InputError(
                "All elements in 'studies' must be PortfolioStudy objects."
            )
        study_names = [study.name for study in self.studies]
        if len(study_names) != len(set(study_names)):
            raise InputError(
                "PortfolioStudy names within a PortfolioSpec must be unique."
            )

        if self.budget_constraint is not None:
            if (
                not isinstance(self.budget_constraint, (int, float))
                or self.budget_constraint < 0
            ):
                raise InputError(
                    "PortfolioSpec 'budget_constraint' must be a non-negative number if specified."
                )


@dataclass(frozen=True)
class DynamicSpec:
    """Specification for dynamic or sequential VOI analyses.

    Attributes
    ----------
    time_steps : Sequence[float]
        A sequence of time points (e.g., years from present) at which
        decisions or data accrual occur.

    Raises
    ------
    InputError
        If `time_steps` is not a non-empty sequence of numbers.
    """

    time_steps: Sequence[float]

    def __post_init__(self: "DynamicSpec"):
        """Validate the dynamic spec."""
        if not isinstance(self.time_steps, Sequence) or not self.time_steps:
            raise InputError(
                "'time_steps' must be a non-empty sequence (list, tuple, np.array)."
            )
        if not all(isinstance(t, (int, float)) for t in self.time_steps):
            raise InputError("All elements in 'time_steps' must be numbers.")
