# voiage/schema.py

"""
Core data structures for voiage.

These structures are designed to hold and manage data used in Value of Information
analyses. They leverage Python's dataclasses for type hinting and validation where
appropriate, and are intended to work seamlessly with NumPy and Pandas/xarray.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional, Union, cast

import numpy as np
import xarray as xr

from voiage.exceptions import (
    raise_import_error,
    raise_input_error,
    raise_value_error,
)

# Try to import JAX for array support
try:
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jnp = None


@dataclass(frozen=True, eq=False)
class ValueArray:
    """Container for probabilistic sensitivity analysis net-benefit samples.

    Attributes
    ----------
    dataset : xarray.Dataset
        Dataset with ``n_samples`` and ``n_strategies`` dimensions and a
        ``net_benefit`` data variable.

    Examples
    --------
    >>> import numpy as np
    >>> from voiage.schema import ValueArray
    >>> values = np.array([[10.0, 12.0], [11.0, 9.5]])
    >>> va = ValueArray.from_numpy(values, ["A", "B"])
    >>> va.n_samples, va.n_strategies
    (2, 2)
    """

    __hash__ = None  # type: ignore[assignment]

    dataset: xr.Dataset

    def __post_init__(self: "ValueArray") -> None:
        """Validate the dataset."""
        if not isinstance(self.dataset, xr.Dataset):
            raise_input_error("ValueArray 'dataset' must be a xarray.Dataset.")
        if "n_samples" not in self.dataset.dims:
            raise_input_error("ValueArray 'dataset' must have a 'n_samples' dimension.")
        if "n_strategies" not in self.dataset.dims:
            raise_input_error(
                "ValueArray 'dataset' must have a 'n_strategies' dimension."
            )
        if "net_benefit" not in self.dataset.data_vars:
            raise_input_error(
                "ValueArray 'dataset' must have a 'net_benefit' data variable."
            )

    @property
    def values(self: "ValueArray") -> xr.DataArray:
        """Return the net benefit values as an xarray DataArray."""
        return self.dataset["net_benefit"]

    @property
    def numpy_values(self: "ValueArray") -> np.ndarray:
        """Return the net benefit values as a NumPy array."""
        return self.values.values

    @property
    def jax_values(self: "ValueArray") -> Optional["jnp.ndarray"]:
        """Return the net benefit values as a JAX array."""
        if not JAX_AVAILABLE:
            return None
        return jnp.asarray(self.numpy_values, dtype=jnp.float64)

    @property
    def n_samples(self: "ValueArray") -> int:
        """Return the number of samples."""
        return int(self.dataset.sizes["n_samples"])

    @property
    def n_strategies(self: "ValueArray") -> int:
        """Return the number of strategies."""
        return int(self.dataset.sizes["n_strategies"])

    @property
    def strategy_names(self: "ValueArray") -> list[str]:
        """Return the names of the strategies."""
        return [str(name) for name in self.dataset["strategy"].values]

    def copy(self: "ValueArray") -> "ValueArray":
        """Return a deep copy of the ValueArray."""
        return ValueArray(dataset=self.dataset.copy(deep=True))

    def get_strategy_index(self: "ValueArray", strategy_name: str) -> int:
        """Return the integer index for a strategy name."""
        try:
            return self.strategy_names.index(strategy_name)
        except ValueError as exc:
            raise_value_error(f"Strategy '{strategy_name}' not found.", exc)

    def slice_by_strategies(
        self: "ValueArray", strategy_names: Sequence[str]
    ) -> "ValueArray":
        """Return a new ValueArray containing only the requested strategies."""
        indices = [self.get_strategy_index(name) for name in strategy_names]
        sliced = self.dataset.isel(n_strategies=indices).copy(deep=True)
        return ValueArray(dataset=sliced)

    def __eq__(self: "ValueArray", other: object) -> bool:
        """Compare ValueArray instances by dataset contents and coordinates."""
        if not isinstance(other, ValueArray):
            return NotImplemented
        return cast("bool", self.dataset.identical(other.dataset))

    @classmethod
    def from_numpy(
        cls,
        values: Union[np.ndarray, "jnp.ndarray"],
        strategy_names: list[str] | None = None,
    ) -> "ValueArray":
        """Create a ValueArray from a numpy or JAX array.

        Args:
            values: A 2D array of shape (n_samples, n_strategies). Supports both NumPy and JAX arrays.
            strategy_names: Optional list of strategy names

        Returns
        -------
            ValueArray: A new ValueArray instance
        """
        # Handle JAX arrays if available
        expected_ndim = 2
        if JAX_AVAILABLE and hasattr(values, "dtype") and hasattr(values, "shape"):
            # This could be a JAX array - convert to numpy for xarray
            values = np.asarray(values)

        if values.ndim != expected_ndim:
            raise_input_error("values must be a 2D array")

        n_samples, n_strategies = values.shape

        if strategy_names is None:
            strategy_names = [f"Strategy {i}" for i in range(n_strategies)]
        elif len(strategy_names) != n_strategies:
            raise_input_error(f"strategy_names must have {n_strategies} elements")

        import xarray as xr

        dataset = xr.Dataset(
            {"net_benefit": (("n_samples", "n_strategies"), values)},
            coords={
                "n_samples": np.arange(n_samples),
                "n_strategies": np.arange(n_strategies),
                "strategy": ("n_strategies", strategy_names),
            },
        )
        return cls(dataset=dataset)

    @classmethod
    def from_numpy_perspectives(
        cls,
        values: Union[np.ndarray, "jnp.ndarray"],
        strategy_names: list[str] | None = None,
        perspective_names: list[str] | None = None,
    ) -> "ValueArray":
        """Create a multi-perspective ValueArray from a 3D array.

        Parameters
        ----------
        values : numpy.ndarray or jax.numpy.ndarray
            Net-benefit values with shape
            ``(n_samples, n_strategies, n_perspectives)``.
        strategy_names : list[str], optional
            Strategy labels aligned to the second dimension.
        perspective_names : list[str], optional
            Perspective labels aligned to the third dimension.

        Returns
        -------
        ValueArray
            ValueArray with a perspective dimension.
        """
        expected_ndim = 3
        if JAX_AVAILABLE and hasattr(values, "dtype") and hasattr(values, "shape"):
            values = np.asarray(values)

        if values.ndim != expected_ndim:
            raise_input_error("values must be a 3D array")

        n_samples, n_strategies, n_perspectives = values.shape

        if strategy_names is None:
            strategy_names = [f"Strategy {i}" for i in range(n_strategies)]
        elif len(strategy_names) != n_strategies:
            raise_input_error(f"strategy_names must have {n_strategies} elements")

        if perspective_names is None:
            perspective_names = [f"Perspective {i}" for i in range(n_perspectives)]
        elif len(perspective_names) != n_perspectives:
            raise_input_error(f"perspective_names must have {n_perspectives} elements")

        dataset = xr.Dataset(
            {
                "net_benefit": (
                    ("n_samples", "n_strategies", "n_perspectives"),
                    values,
                )
            },
            coords={
                "n_samples": np.arange(n_samples),
                "n_strategies": np.arange(n_strategies),
                "n_perspectives": np.arange(n_perspectives),
                "strategy": ("n_strategies", strategy_names),
                "perspective": ("n_perspectives", perspective_names),
            },
        )
        return cls(dataset=dataset)

    @property
    def perspective_names(self: "ValueArray") -> list[str] | None:
        """Return perspective labels when present."""
        if "perspective" not in self.dataset.coords:
            return None
        return [str(name) for name in self.dataset["perspective"].values]

    @classmethod
    def from_jax(
        cls, values: "jnp.ndarray", strategy_names: list[str] | None = None
    ) -> "ValueArray":
        """Create a ValueArray from a JAX array.

        Args:
            values: A 2D JAX array of shape (n_samples, n_strategies)
            strategy_names: Optional list of strategy names

        Returns
        -------
            ValueArray: A new ValueArray instance
        """
        if not JAX_AVAILABLE:
            raise_import_error(
                "JAX is not available. Please install JAX to use from_jax()."
            )

        if not hasattr(values, "shape"):
            raise_input_error("values must be a JAX array with a shape attribute")

        expected_ndim = 2
        if values.ndim != expected_ndim:
            raise_input_error("values must be a 2D array")

        # Convert JAX array to NumPy for xarray compatibility
        numpy_values = np.asarray(values)

        n_samples, n_strategies = numpy_values.shape

        if strategy_names is None:
            strategy_names = [f"Strategy {i}" for i in range(n_strategies)]
        elif len(strategy_names) != n_strategies:
            raise_input_error(f"strategy_names must have {n_strategies} elements")

        import xarray as xr

        dataset = xr.Dataset(
            {"net_benefit": (("n_samples", "n_strategies"), numpy_values)},
            coords={
                "n_samples": np.arange(n_samples),
                "n_strategies": np.arange(n_strategies),
                "strategy": ("n_strategies", strategy_names),
            },
        )
        return cls(dataset=dataset)


@dataclass(frozen=True, eq=False)
class ParameterSet:
    """Container for parameter samples from a probabilistic sensitivity analysis.

    Attributes
    ----------
    dataset : xarray.Dataset
        Dataset with ``n_samples`` as the sample dimension and one data
        variable per parameter.

    Examples
    --------
    >>> import numpy as np
    >>> from voiage.schema import ParameterSet
    >>> params = ParameterSet.from_numpy_or_dict({"cost": np.array([1.0, 2.0])})
    >>> params.parameter_names
    ['cost']
    """

    __hash__ = None  # type: ignore[assignment]

    dataset: xr.Dataset

    def __post_init__(self: "ParameterSet") -> None:
        """Validate the dataset."""
        if not isinstance(self.dataset, xr.Dataset):
            raise_input_error("ParameterSet 'dataset' must be a xarray.Dataset.")
        if "n_samples" not in self.dataset.dims:
            raise_input_error(
                "ParameterSet 'dataset' must have a 'n_samples' dimension."
            )

    @property
    def parameters(self: "ParameterSet") -> dict[str, np.ndarray]:
        """Return the parameter samples."""
        return {str(name): self.dataset[name].values for name in self.dataset.data_vars}

    @property
    def jax_parameters(self: "ParameterSet") -> dict[str, "jnp.ndarray"] | None:
        """Return the parameter samples as JAX arrays."""
        if not JAX_AVAILABLE:
            return None
        return {
            str(name): jnp.asarray(self.dataset[name].values, dtype=jnp.float64)
            for name in self.dataset.data_vars
        }

    @property
    def n_samples(self: "ParameterSet") -> int:
        """Return the number of samples."""
        return int(self.dataset.sizes["n_samples"])

    @property
    def parameter_names(self: "ParameterSet") -> list[str]:
        """Return the names of the parameters."""
        return [str(name) for name in self.dataset.data_vars]

    def copy(self: "ParameterSet") -> "ParameterSet":
        """Return a deep copy of the ParameterSet."""
        return ParameterSet(dataset=self.dataset.copy(deep=True))

    def subset_by_parameters(
        self: "ParameterSet", parameter_names: Sequence[str]
    ) -> "ParameterSet":
        """Return a new ParameterSet containing only the requested parameters."""
        parameter_list = list(parameter_names)
        missing = [
            name for name in parameter_list if name not in self.dataset.data_vars
        ]
        if missing:
            missing_names = ", ".join(sorted(missing))
            raise_value_error(f"Parameters not found: {missing_names}")
        subset = self.dataset[parameter_list].copy(deep=True)
        return ParameterSet(dataset=subset)

    def __eq__(self: "ParameterSet", other: object) -> bool:
        """Compare ParameterSet instances by dataset contents and coordinates."""
        if not isinstance(other, ParameterSet):
            return NotImplemented
        return cast("bool", self.dataset.identical(other.dataset))

    @classmethod
    def from_numpy_or_dict(
        cls,
        parameters: Union[
            np.ndarray, dict[str, np.ndarray], "jnp.ndarray", dict[str, "jnp.ndarray"]
        ],
    ) -> "ParameterSet":
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
        expected_ndim = 2
        if (
            JAX_AVAILABLE
            and hasattr(parameters, "ndim")
            and hasattr(parameters, "dtype")
        ):
            # This could be a JAX array - convert to numpy for xarray
            parameters = np.asarray(parameters)

        if isinstance(parameters, np.ndarray):
            if parameters.ndim != expected_ndim:
                raise_input_error("parameters array must be 2D")
            n_samples, n_parameters = parameters.shape
            # Create parameter names
            param_names = [f"param_{i}" for i in range(n_parameters)]
            # Create dataset
            data_vars = {
                name: (("n_samples",), parameters[:, i])
                for i, name in enumerate(param_names)
            }
            dataset = xr.Dataset(data_vars, coords={"n_samples": np.arange(n_samples)})
        elif isinstance(parameters, dict):
            if not parameters:
                raise_input_error("parameters dictionary cannot be empty")
            # Check that all arrays have the same length
            lengths = []
            converted_params = {}
            for name, arr in parameters.items():
                # Convert JAX arrays to numpy if needed
                if JAX_AVAILABLE and hasattr(arr, "dtype") and hasattr(arr, "shape"):
                    np_arr = np.asarray(arr)
                else:
                    np_arr = arr
                lengths.append(len(np_arr))
                converted_params[name] = np_arr
            if len(set(lengths)) > 1:
                raise_input_error("All parameter arrays must have the same length")
            n_samples = lengths[0]
            # Create dataset
            data_vars = {
                name: (("n_samples",), arr) for name, arr in converted_params.items()
            }
            dataset = xr.Dataset(data_vars, coords={"n_samples": np.arange(n_samples)})
        else:
            raise_input_error("parameters must be a numpy array or dictionary")

        return cls(dataset=dataset)

    @classmethod
    def from_jax(
        cls, parameters: Union["jnp.ndarray", dict[str, "jnp.ndarray"]]
    ) -> "ParameterSet":
        """Create a ParameterSet from a JAX array or dictionary.

        Args:
            parameters: Either a 2D JAX array of shape (n_samples, n_parameters)
                       or a dictionary mapping parameter names to 1D JAX arrays

        Returns
        -------
            ParameterSet: A new ParameterSet instance
        """
        if not JAX_AVAILABLE:
            raise_import_error(
                "JAX is not available. Please install JAX to use from_jax()."
            )

        import xarray as xr

        expected_ndim = 2
        if jnp is not None and hasattr(parameters, "ndim"):  # JAX array
            if parameters.ndim != expected_ndim:
                raise_input_error("parameters array must be 2D")
            numpy_parameters = np.asarray(parameters)
            n_samples, n_parameters = numpy_parameters.shape
            # Create parameter names
            param_names = [f"param_{i}" for i in range(n_parameters)]
            # Create dataset
            data_vars = {
                name: (("n_samples",), numpy_parameters[:, i])
                for i, name in enumerate(param_names)
            }
            dataset = xr.Dataset(data_vars, coords={"n_samples": np.arange(n_samples)})
        elif isinstance(parameters, dict):
            if not parameters:
                raise_input_error("parameters dictionary cannot be empty")
            # Check that all arrays have the same length and convert to numpy
            lengths = []
            converted_params = {}
            for name, arr in parameters.items():
                if not hasattr(arr, "shape"):
                    raise_input_error(f"Parameter {name} must be a JAX array")
                lengths.append(len(arr))
                converted_params[name] = np.asarray(arr)

            if len(set(lengths)) > 1:
                raise_input_error("All parameter arrays must have the same length")
            n_samples = lengths[0]
            # Create dataset
            data_vars = {
                name: (("n_samples",), arr) for name, arr in converted_params.items()
            }
            dataset = xr.Dataset(data_vars, coords={"n_samples": np.arange(n_samples)})
        else:
            raise_input_error("parameters must be a JAX array or dictionary")

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

    Examples
    --------
    >>> from voiage.schema import DecisionOption
    >>> arm = DecisionOption(name="Standard care", sample_size=100)
    >>> arm.to_dict()
    {'name': 'Standard care', 'sample_size': 100}

    Raises
    ------
    InputError
        If `name` is not a non-empty string or `sample_size` is not a
        positive integer.
    """

    name: str
    sample_size: int

    def __post_init__(self: "DecisionOption") -> None:
        """Validate the decision option."""
        if not isinstance(self.name, str) or not self.name:
            raise_input_error("DecisionOption 'name' must be a non-empty string.")
        if not isinstance(self.sample_size, int) or self.sample_size <= 0:
            raise_input_error(
                "DecisionOption 'sample_size' must be a positive integer."
            )

    def to_dict(self) -> dict[str, object]:
        """Serialize the decision option to a dictionary."""
        return {"name": self.name, "sample_size": self.sample_size}

    @classmethod
    def from_dict(cls, data: object) -> "DecisionOption":
        """Deserialize a decision option from a dictionary."""
        if not isinstance(data, dict):
            raise_input_error("DecisionOption data must be a dictionary.")
        if "name" not in data:
            raise_input_error("DecisionOption data must include 'name'.")
        if "sample_size" not in data:
            raise_input_error("DecisionOption data must include 'sample_size'.")

        name = data["name"]
        sample_size = data["sample_size"]
        if not isinstance(name, str):
            raise_input_error("DecisionOption 'name' must be a string.")
        if not isinstance(sample_size, int):
            raise_input_error("DecisionOption 'sample_size' must be an integer.")

        return cls(name=name, sample_size=sample_size)


@dataclass(frozen=True)
class TrialDesign:
    """Specifies the design of a proposed trial for EVSI calculations.

    Attributes
    ----------
    arms : List[DecisionOption]
        A list of `DecisionOption` objects that together define the trial.

    Examples
    --------
    >>> from voiage.schema import DecisionOption, TrialDesign
    >>> design = TrialDesign([DecisionOption("A", 50), DecisionOption("B", 50)])
    >>> design.total_sample_size
    100

    Raises
    ------
    InputError
        If `arms` is not a non-empty list of `DecisionOption` objects, or if
        any of the arm names are duplicated.
    """

    arms: list[DecisionOption]

    def __post_init__(self: "TrialDesign") -> None:
        """Validate the trial design."""
        if not isinstance(self.arms, list) or not self.arms:
            raise_input_error(
                "TrialDesign 'arms' must be a non-empty list of DecisionOption objects."
            )
        if not all(isinstance(arm, DecisionOption) for arm in self.arms):
            raise_input_error("All elements in 'arms' must be DecisionOption objects.")
        arm_names = [arm.name for arm in self.arms]
        if len(arm_names) != len(set(arm_names)):
            raise_input_error(
                "DecisionOption names within a TrialDesign must be unique."
            )

    def to_dict(self) -> dict[str, object]:
        """Serialize the trial design to a dictionary."""
        return {"arms": [arm.to_dict() for arm in self.arms]}

    @classmethod
    def from_dict(cls, data: object) -> "TrialDesign":
        """Deserialize a trial design from a dictionary."""
        if not isinstance(data, dict):
            raise_input_error("TrialDesign data must be a dictionary.")
        if "arms" not in data:
            raise_input_error("TrialDesign data must include 'arms'.")

        arms = data["arms"]
        if not isinstance(arms, Sequence) or isinstance(arms, (str, bytes)):
            raise_input_error("TrialDesign 'arms' must be a sequence of dictionaries.")

        option_arms = [DecisionOption.from_dict(arm) for arm in arms]
        return cls(arms=option_arms)

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

    Examples
    --------
    >>> from voiage.schema import DecisionOption, PortfolioStudy, TrialDesign
    >>> study = PortfolioStudy("Trial 1", TrialDesign([DecisionOption("A", 10)]), 1000.0)
    >>> study.cost
    1000.0

    Raises
    ------
    InputError
        If inputs are of the wrong type or `cost` is negative.
    """

    name: str
    design: TrialDesign
    cost: float

    def __post_init__(self: "PortfolioStudy") -> None:
        """Validate the portfolio study."""
        if not isinstance(self.name, str) or not self.name:
            raise_input_error("PortfolioStudy 'name' must be a non-empty string.")
        if not isinstance(self.design, TrialDesign):
            raise_input_error("PortfolioStudy 'design' must be a TrialDesign object.")
        if not isinstance(self.cost, (int, float)) or self.cost < 0:
            raise_input_error("PortfolioStudy 'cost' must be a non-negative number.")


@dataclass(frozen=True)
class PortfolioSpec:
    """Defines a portfolio of candidate research studies for optimization.

    Attributes
    ----------
    studies : List[PortfolioStudy]
        A list of `PortfolioStudy` objects representing the candidate studies.
    budget_constraint : Optional[float], optional
        The overall budget limit for the portfolio. Defaults to None.

    Examples
    --------
    >>> from voiage.schema import DecisionOption, PortfolioSpec, PortfolioStudy, TrialDesign
    >>> study = PortfolioStudy("Trial 1", TrialDesign([DecisionOption("A", 10)]), 1000.0)
    >>> spec = PortfolioSpec([study], budget_constraint=2000.0)
    >>> spec.budget_constraint
    2000.0

    Raises
    ------
    InputError
        If `studies` is not a non-empty list of `PortfolioStudy` objects,
        if study names are duplicated, or if `budget_constraint` is negative.
    """

    studies: list[PortfolioStudy]
    budget_constraint: float | None = None

    def __post_init__(self: "PortfolioSpec") -> None:
        """Validate the portfolio spec."""
        if not isinstance(self.studies, list) or not self.studies:
            raise_input_error(
                "PortfolioSpec 'studies' must be a non-empty list of PortfolioStudy objects."
            )
        if not all(isinstance(study, PortfolioStudy) for study in self.studies):
            raise_input_error(
                "All elements in 'studies' must be PortfolioStudy objects."
            )
        study_names = [study.name for study in self.studies]
        if len(study_names) != len(set(study_names)):
            raise_input_error(
                "PortfolioStudy names within a PortfolioSpec must be unique."
            )

        if self.budget_constraint is not None and (
            not isinstance(self.budget_constraint, (int, float))
            or self.budget_constraint < 0
        ):
            raise_input_error(
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

    Examples
    --------
    >>> from voiage.schema import DynamicSpec
    >>> spec = DynamicSpec([0.0, 1.0, 2.0])
    >>> list(spec.time_steps)
    [0.0, 1.0, 2.0]

    Raises
    ------
    InputError
        If `time_steps` is not a non-empty sequence of numbers.
    """

    time_steps: Sequence[float]

    def __post_init__(self: "DynamicSpec") -> None:
        """Validate the dynamic spec."""
        if not isinstance(self.time_steps, Sequence) or not self.time_steps:
            raise_input_error(
                "'time_steps' must be a non-empty sequence (list, tuple, np.array)."
            )
        if not all(isinstance(t, (int, float)) for t in self.time_steps):
            raise_input_error("All elements in 'time_steps' must be numbers.")
