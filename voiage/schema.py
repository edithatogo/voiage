# voiage/schema.py

"""
Core data structures for voiage.

These structures are designed to hold and manage data used in Value of Information
analyses. They leverage Python's dataclasses for type hinting and validation where
appropriate, and use wide Polars frames as the canonical in-memory representation
for their public contract (``specs/v3/stable-api.json``).
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from importlib import import_module
from importlib.util import find_spec
from typing import TYPE_CHECKING, cast

import numpy as np
import polars as pl

from voiage.exceptions import (
    raise_dimension_mismatch_error,
    raise_import_error,
    raise_input_error,
    raise_value_error,
)

if TYPE_CHECKING:
    from jax import Array as JaxArray


class _LazyJaxNumpy:
    """Load JAX only when a JAX-specific conversion is requested."""

    def __getattr__(self, name: str) -> object:
        return getattr(import_module("jax.numpy"), name)


try:
    JAX_AVAILABLE = find_spec("jax") is not None
except ImportError, ModuleNotFoundError:
    JAX_AVAILABLE = False
jnp = _LazyJaxNumpy() if JAX_AVAILABLE else None


def _validate_float_frame(
    frame: object,
    *,
    container: str,
    supports_3d: bool,
) -> None:
    """Validate a wide Polars frame of Float64 columns.

    The frame must be a ``polars.DataFrame`` whose columns are either all
    ``Float64`` (2D surface) or all ``Array(Float64, n)`` (3D multi-perspective
    surface) -- mixed dtypes are rejected, and an unrecognised/extra variable
    (any column that is neither) fails closed. Empty frames and null values are
    rejected. Polars rejects duplicate column names before a DataFrame can
    reach this validator; NumPy constructors also validate the supplied names.
    """
    role = "strategy" if container == "ValueArray" else "parameter"
    if not isinstance(frame, pl.DataFrame):
        raise_input_error(f"{container} 'frame' must be a polars.DataFrame.")
    if frame.width == 0:
        raise_input_error(f"{container} 'frame' must have at least one {role} column.")

    dtypes = list(frame.dtypes)
    is_3d = all(isinstance(dtype, pl.Array) for dtype in dtypes)
    is_2d = all(dtype == pl.Float64 for dtype in dtypes)
    if is_3d:
        if not supports_3d:
            raise_input_error(
                f"{container} 'frame' has Array columns but 3D multi-perspective "
                "inputs are not supported here."
            )
        if any(dtype.inner != pl.Float64 for dtype in dtypes):
            raise_input_error(
                f"{container} 'frame' must contain Array(Float64, n) columns "
                "with exactly one perspective dimension."
            )
        widths = {dtype.size for dtype in dtypes}
        if len(widths) != 1:
            raise_input_error(
                f"{container} 'frame' Array columns must all have the same width."
            )
    elif not is_2d:
        raise_input_error(
            f"{container} 'frame' must contain only Float64 columns (2D) or only "
            "Array(Float64, n) columns (3D); unrecognised extra variables are rejected."
        )

    # Null counts are cached column metadata. Avoid launching query plans for
    # these scalar validation checks on every container construction.
    if any(column.null_count() for column in frame.iter_columns()):
        raise_input_error(f"{container} 'frame' must not contain null values.")
    if is_3d and any(column.explode().null_count() for column in frame.iter_columns()):
        raise_input_error(f"{container} 'frame' must not contain null values.")


@dataclass(frozen=True, eq=False)
class ValueArray:
    """Container for probabilistic sensitivity analysis net-benefit samples.

    Attributes
    ----------
    frame : pl.DataFrame
        Wide Polars frame with one ``Float64`` column per strategy in declared
        order (2D surface), or one ``Array(Float64, n_perspectives)`` column per
        strategy (3D multi-perspective surface). Column names are the strategy
        names. Sample count is the frame height.

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

    frame: pl.DataFrame = field(repr=False)
    perspective_names: list[str] | None = field(default=None, repr=False)

    def __post_init__(self: ValueArray) -> None:
        """Validate the frame."""
        _validate_float_frame(self.frame, container="ValueArray", supports_3d=True)

        # Multi-perspective surface: perspective_names is a second dataclass
        # field; infer validation constraints here.
        frame = self.frame
        if all(isinstance(dtype, pl.Array) for dtype in frame.dtypes):
            n_perspectives = cast("int", frame.dtypes[0].size)
            if self.perspective_names is not None and (
                len(self.perspective_names) != n_perspectives
            ):
                raise_input_error(
                    f"perspective_names must have {n_perspectives} elements"
                )
        elif self.perspective_names is not None:
            raise_input_error(
                "perspective_names are only valid for a 3D multi-perspective ValueArray."
            )
        object.__setattr__(self, "frame", frame.clone())
        if self.perspective_names is not None:
            object.__setattr__(self, "perspective_names", list(self.perspective_names))

    @classmethod
    def from_polars(
        cls,
        frame: pl.DataFrame,
        perspective_names: list[str] | None = None,
    ) -> ValueArray:
        """Create a ValueArray from a wide Polars frame."""
        return cls(frame=frame, perspective_names=perspective_names)

    def to_polars(self: ValueArray) -> pl.DataFrame:
        """Return a deep copy of the wide Polars frame."""
        return self.frame.clone()

    @property
    def values(self: ValueArray) -> np.ndarray:
        """Return the net benefit values as a NumPy array (2D or 3D)."""
        return self.numpy_values

    @property
    def numpy_values(self: ValueArray) -> np.ndarray:
        """Return the net benefit values as a NumPy array.

        For the 2D surface this is ``(n_samples, n_strategies)``. For the 3D
        multi-perspective surface this is ``(n_samples, n_strategies,
        n_perspectives)``. The returned array is a fresh C-contiguous copy, so
        callers may mutate it without affecting the container.
        """
        frame = self.frame
        if all(isinstance(dtype, pl.Array) for dtype in frame.dtypes):
            return np.stack(
                [frame[column].to_numpy() for column in frame.columns], axis=1
            )
        return np.column_stack([frame[column].to_numpy() for column in frame.columns])

    @property
    def jax_values(self: ValueArray) -> JaxArray | None:
        """Return the net benefit values as a JAX array."""
        if not JAX_AVAILABLE:
            return None
        return jnp.asarray(self.numpy_values, dtype=jnp.float64)

    @property
    def n_samples(self: ValueArray) -> int:
        """Return the number of samples."""
        return self.frame.height

    @property
    def n_strategies(self: ValueArray) -> int:
        """Return the number of strategies."""
        return self.frame.width

    @property
    def strategy_names(self: ValueArray) -> list[str]:
        """Return the names of the strategies."""
        return list(self.frame.columns)

    def copy(self: ValueArray) -> ValueArray:
        """Return a deep copy of the ValueArray."""
        return ValueArray(
            frame=self.frame.clone(),
            perspective_names=(
                list(self.perspective_names)
                if self.perspective_names is not None
                else None
            ),
        )

    def get_strategy_index(self: ValueArray, strategy_name: str) -> int:
        """Return the integer index for a strategy name."""
        try:
            return self.strategy_names.index(strategy_name)
        except ValueError as exc:
            raise_value_error(f"Strategy '{strategy_name}' not found.", exc)

    def slice_by_strategies(
        self: ValueArray, strategy_names: Sequence[str]
    ) -> ValueArray:
        """Return a new ValueArray containing only the requested strategies."""
        indices = [self.get_strategy_index(name) for name in strategy_names]
        sliced = self.frame.select([self.strategy_names[i] for i in indices]).clone()
        return ValueArray(
            frame=sliced,
            perspective_names=(
                list(self.perspective_names)
                if self.perspective_names is not None
                else None
            ),
        )

    def __eq__(self: ValueArray, other: object) -> bool:
        """Compare ValueArray instances by frame contents and column names."""
        if not isinstance(other, ValueArray):
            return NotImplemented
        return cast(
            "bool",
            self.frame.equals(other.frame)
            and self.perspective_names == other.perspective_names,
        )

    @classmethod
    def from_numpy(
        cls,
        values: np.ndarray | JaxArray,
        strategy_names: list[str] | None = None,
    ) -> ValueArray:
        """Create a ValueArray from a numpy or JAX array.

        Args:
            values: A 2D Float64 array of shape (n_samples, n_strategies). Supports both NumPy and JAX arrays.
            strategy_names: Optional list of strategy names

        Returns
        -------
            ValueArray: A new ValueArray instance
        """
        # Handle JAX arrays if available
        expected_ndim = 2
        if JAX_AVAILABLE and hasattr(values, "dtype") and hasattr(values, "shape"):
            # This could be a JAX array - convert to numpy
            values = np.asarray(values)

        if values.ndim != expected_ndim:
            raise_dimension_mismatch_error("values must be a 2D array")

        n_strategies = values.shape[1]

        if strategy_names is None:
            strategy_names = [f"Strategy {i}" for i in range(n_strategies)]
        elif len(strategy_names) != n_strategies:
            raise_input_error(f"strategy_names must have {n_strategies} elements")

        if len(set(strategy_names)) != len(strategy_names):
            raise_input_error("strategy_names must be unique")

        frame = pl.DataFrame(
            {
                str(name): np.asarray(values[:, col_idx], dtype=np.float64)
                for col_idx, name in enumerate(strategy_names)
            }
        )
        return cls(frame=frame)

    @classmethod
    def from_numpy_perspectives(
        cls,
        values: np.ndarray | JaxArray,
        strategy_names: list[str] | None = None,
        perspective_names: list[str] | None = None,
    ) -> ValueArray:
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
            ValueArray with a multi-perspective surface.
        """
        expected_ndim = 3
        if JAX_AVAILABLE and hasattr(values, "dtype") and hasattr(values, "shape"):
            values = np.asarray(values)

        if values.ndim != expected_ndim:
            raise_input_error("values must be a 3D array")

        n_strategies = values.shape[1]
        n_perspectives = values.shape[2]

        if strategy_names is None:
            strategy_names = [f"Strategy {i}" for i in range(n_strategies)]
        elif len(strategy_names) != n_strategies:
            raise_input_error(f"strategy_names must have {n_strategies} elements")

        if len(set(strategy_names)) != len(strategy_names):
            raise_input_error("strategy_names must be unique")

        if perspective_names is None:
            perspective_names = [f"Perspective {i}" for i in range(n_perspectives)]
        elif len(perspective_names) != n_perspectives:
            raise_input_error(f"perspective_names must have {n_perspectives} elements")

        frame = pl.DataFrame(
            {
                str(name): pl.Series(
                    str(name),
                    np.asarray(values[:, col_idx, :], dtype=np.float64),
                    dtype=pl.Array(pl.Float64, n_perspectives),
                )
                for col_idx, name in enumerate(strategy_names)
            }
        )
        return cls(frame=frame, perspective_names=list(perspective_names))

    @property
    def perspective_count(self: ValueArray) -> int | None:
        """Return the number of perspectives when the surface is 3D."""
        dtypes = self.frame.dtypes
        if all(isinstance(dtype, pl.Array) for dtype in dtypes):
            return cast("int", dtypes[0].size)
        return None

    @classmethod
    def from_jax(
        cls, values: JaxArray, strategy_names: list[str] | None = None
    ) -> ValueArray:
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
            raise_dimension_mismatch_error("values must be a 2D array")

        # Convert JAX array to NumPy
        numpy_values = np.asarray(values)

        return cls.from_numpy(numpy_values, strategy_names=strategy_names)


@dataclass(frozen=True, eq=False)
class ParameterSet:
    """Container for parameter samples from a probabilistic sensitivity analysis.

    Attributes
    ----------
    frame : pl.DataFrame
        Wide Polars frame with one ``Float64`` column per parameter in declared
        order. Column names are the parameter names. Sample count is the frame
        height.

    Examples
    --------
    >>> import numpy as np
    >>> from voiage.schema import ParameterSet
    >>> params = ParameterSet.from_numpy_or_dict({"cost": np.array([1.0, 2.0])})
    >>> params.parameter_names
    ['cost']
    """

    __hash__ = None  # type: ignore[assignment]

    frame: pl.DataFrame = field(repr=False)

    def __post_init__(self: ParameterSet) -> None:
        """Validate the frame."""
        _validate_float_frame(self.frame, container="ParameterSet", supports_3d=False)
        object.__setattr__(self, "frame", self.frame.clone())

    @classmethod
    def from_polars(cls, frame: pl.DataFrame) -> ParameterSet:
        """Create a ParameterSet from a wide Polars frame."""
        return cls(frame=frame)

    def to_polars(self: ParameterSet) -> pl.DataFrame:
        """Return a deep copy of the wide Polars frame."""
        return self.frame.clone()

    @property
    def parameters(self: ParameterSet) -> dict[str, np.ndarray]:
        """Return the parameter samples."""
        return {
            str(name): self.frame[str(name)].to_numpy() for name in self.frame.columns
        }

    @property
    def jax_parameters(self: ParameterSet) -> dict[str, JaxArray] | None:
        """Return the parameter samples as JAX arrays."""
        if not JAX_AVAILABLE:
            return None
        return {
            str(name): jnp.asarray(self.frame[str(name)].to_numpy(), dtype=jnp.float64)
            for name in self.frame.columns
        }

    @property
    def n_samples(self: ParameterSet) -> int:
        """Return the number of samples."""
        return self.frame.height

    @property
    def parameter_names(self: ParameterSet) -> list[str]:
        """Return the names of the parameters."""
        return list(self.frame.columns)

    def copy(self: ParameterSet) -> ParameterSet:
        """Return a deep copy of the ParameterSet."""
        return ParameterSet(frame=self.frame.clone())

    def subset_by_parameters(
        self: ParameterSet, parameter_names: Sequence[str]
    ) -> ParameterSet:
        """Return a new ParameterSet containing only the requested parameters."""
        parameter_list = list(parameter_names)
        missing = [name for name in parameter_list if name not in self.frame.columns]
        if missing:
            missing_names = ", ".join(sorted(missing))
            raise_value_error(f"Parameters not found: {missing_names}")
        subset = self.frame.select(parameter_list).clone()
        return ParameterSet(frame=subset)

    def __eq__(self: ParameterSet, other: object) -> bool:
        """Compare ParameterSet instances by frame contents and column names."""
        if not isinstance(other, ParameterSet):
            return NotImplemented
        return cast("bool", self.frame.equals(other.frame))

    @classmethod
    def from_numpy_or_dict(
        cls,
        parameters: np.ndarray | dict[str, np.ndarray] | JaxArray | dict[str, JaxArray],
    ) -> ParameterSet:
        """Create a ParameterSet from a numpy/JAX array or dictionary.

        Args:
            parameters: Either a 2D array of shape (n_samples, n_parameters)
                       or a dictionary mapping parameter names to 1D arrays.
                       Supports both NumPy and JAX arrays.

        Returns
        -------
            ParameterSet: A new ParameterSet instance
        """
        # Handle JAX arrays if available
        expected_ndim = 2
        if (
            JAX_AVAILABLE
            and hasattr(parameters, "ndim")
            and hasattr(parameters, "dtype")
        ):
            # This could be a JAX array - convert to numpy
            parameters = np.asarray(parameters)

        if isinstance(parameters, np.ndarray):
            if parameters.ndim != expected_ndim:
                raise_input_error("parameters array must be 2D")
            n_parameters = parameters.shape[1]
            # Create parameter names
            param_names = [f"param_{i}" for i in range(n_parameters)]
            frame = pl.DataFrame(
                {
                    name: np.asarray(parameters[:, i], dtype=np.float64)
                    for i, name in enumerate(param_names)
                }
            )
        elif isinstance(parameters, dict):
            if not parameters:
                raise_input_error("parameters dictionary cannot be empty")
            # Check that all arrays have the same length
            lengths = []
            converted_params: dict[str, np.ndarray] = {}
            for name, arr in parameters.items():
                # Convert JAX arrays to numpy if needed
                if JAX_AVAILABLE and hasattr(arr, "dtype") and hasattr(arr, "shape"):
                    np_arr = np.asarray(arr)
                else:
                    np_arr = arr
                lengths.append(len(np_arr))
                converted_params[str(name)] = np.asarray(np_arr, dtype=np.float64)
            if len(set(lengths)) > 1:
                raise_input_error("All parameter arrays must have the same length")
            frame = pl.DataFrame(converted_params)
        else:
            raise_input_error("parameters must be a numpy array or dictionary")
        return cls(frame=frame)

    @classmethod
    def from_jax(cls, parameters: JaxArray | dict[str, JaxArray]) -> ParameterSet:
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

        expected_ndim = 2
        converted: np.ndarray | dict[str, np.ndarray]
        if jnp is not None and hasattr(parameters, "ndim"):  # JAX array
            if parameters.ndim != expected_ndim:
                raise_input_error("parameters array must be 2D")
            converted = np.asarray(parameters)
        elif isinstance(parameters, dict):
            if not parameters:
                raise_input_error("parameters dictionary cannot be empty")
            # Check that all arrays have the same length and convert to numpy
            lengths = []
            converted_params: dict[str, np.ndarray] = {}
            for name, arr in parameters.items():
                if not hasattr(arr, "shape"):
                    raise_input_error(f"Parameter {name} must be a JAX array")
                lengths.append(len(arr))
                converted_params[str(name)] = np.asarray(arr)

            if len(set(lengths)) > 1:
                raise_input_error("All parameter arrays must have the same length")
            converted = converted_params
        else:
            raise_input_error("parameters must be a JAX array or dictionary")
        return cls.from_numpy_or_dict(converted)


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

    def __post_init__(self: DecisionOption) -> None:
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
    def from_dict(cls, data: object) -> DecisionOption:
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

    def __post_init__(self: TrialDesign) -> None:
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
    def from_dict(cls, data: object) -> TrialDesign:
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
    def total_sample_size(self: TrialDesign) -> int:
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

    def __post_init__(self: PortfolioStudy) -> None:
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

    def __post_init__(self: PortfolioSpec) -> None:
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

    def __post_init__(self: DynamicSpec) -> None:
        """Validate the dynamic spec."""
        if not isinstance(self.time_steps, Sequence) or not self.time_steps:
            raise_input_error(
                "'time_steps' must be a non-empty sequence (list, tuple, np.array)."
            )
        if not all(isinstance(t, (int, float)) for t in self.time_steps):
            raise_input_error("All elements in 'time_steps' must be numbers.")


@dataclass(frozen=True)
class Intervention:
    """Represents a candidate intervention or strategy in a decision problem.

    Attributes
    ----------
    intervention_id : str
        Stable identifier for the intervention or strategy.
    name : str
        Human-readable intervention name.
    description : str, optional
        Optional description of the intervention.
    is_reference : bool, default=False
        True when this intervention is the comparator or reference arm.
    category : str, optional
        Optional grouping label for the intervention.

    Examples
    --------
    >>> from voiage.schema import Intervention
    >>> arm = Intervention(intervention_id="int_01", name="Treatment A", is_reference=True)
    >>> arm.name
    'Treatment A'
    """

    intervention_id: str
    name: str
    description: str | None = None
    is_reference: bool = False
    category: str | None = None

    def __post_init__(self) -> None:
        """Validate intervention properties."""
        if not isinstance(self.intervention_id, str) or not self.intervention_id:
            raise_input_error(
                "Intervention 'intervention_id' must be a non-empty string."
            )
        if not isinstance(self.name, str) or not self.name:
            raise_input_error("Intervention 'name' must be a non-empty string.")
        if self.description is not None and not isinstance(self.description, str):
            raise_input_error(
                "Intervention 'description' must be a string if specified."
            )
        if not isinstance(self.is_reference, bool):
            raise_input_error("Intervention 'is_reference' must be a boolean.")
        if self.category is not None and not isinstance(self.category, str):
            raise_input_error("Intervention 'category' must be a string if specified.")

    def to_dict(self) -> dict[str, object]:
        """Serialize intervention to dictionary conforming to InterventionV1 schema."""
        result: dict[str, object] = {
            "intervention_id": self.intervention_id,
            "name": self.name,
            "is_reference": self.is_reference,
        }
        if self.description is not None:
            result["description"] = self.description
        if self.category is not None:
            result["category"] = self.category
        return result

    @classmethod
    def from_dict(cls, data: object) -> Intervention:
        """Deserialize intervention from dictionary."""
        if not isinstance(data, dict):
            raise_input_error("Intervention data must be a dictionary.")
        if "intervention_id" not in data or "name" not in data:
            raise_input_error(
                "Intervention data must include 'intervention_id' and 'name'."
            )
        return cls(
            intervention_id=str(data["intervention_id"]),
            name=str(data["name"]),
            description=cast("str | None", data.get("description")),
            is_reference=bool(data.get("is_reference", False)),
            category=cast("str | None", data.get("category")),
        )


@dataclass(frozen=True)
class DecisionProblem:
    """Represents a canonical Decision Problem (DecisionProblemV1).

    Attributes
    ----------
    decision_problem_id : str
        Stable identifier for the decision problem.
    title : str
        Short human-readable title for the problem.
    willingness_to_pay : float
        Decision threshold used to calculate net benefit. Must be strictly positive.
    interventions : list[Intervention]
        Candidate interventions compared by the problem.
    currency : str, default="USD"
        Currency code or symbol used for cost outputs (min length 3).
    analysis_type : str, default="net-benefit-first"
        Analysis framework (must be "net-benefit-first").
    outcome_names : list[str], optional
        Ordered list of outcome labels used by the problem.

    Examples
    --------
    >>> from voiage.schema import DecisionProblem, Intervention
    >>> prob = DecisionProblem(
    ...     decision_problem_id="prob_01",
    ...     title="Screening Decision",
    ...     willingness_to_pay=50000.0,
    ...     interventions=[
    ...         Intervention(intervention_id="i1", name="No Screening", is_reference=True),
    ...         Intervention(intervention_id="i2", name="Annual Screening"),
    ...     ],
    ... )
    >>> prob.title
    'Screening Decision'
    """

    decision_problem_id: str
    title: str
    willingness_to_pay: float
    interventions: list[Intervention]
    currency: str = "USD"
    analysis_type: str = "net-benefit-first"
    outcome_names: list[str] | None = None

    def __post_init__(self) -> None:
        """Validate decision problem properties."""
        if (
            not isinstance(self.decision_problem_id, str)
            or not self.decision_problem_id
        ):
            raise_input_error(
                "DecisionProblem 'decision_problem_id' must be a non-empty string."
            )
        if not isinstance(self.title, str) or not self.title:
            raise_input_error("DecisionProblem 'title' must be a non-empty string.")
        if (
            not isinstance(self.willingness_to_pay, (int, float))
            or self.willingness_to_pay <= 0
        ):
            raise_input_error(
                "DecisionProblem 'willingness_to_pay' must be a positive number."
            )
        if not isinstance(self.currency, str) or len(self.currency) < 3:
            raise_input_error(
                "DecisionProblem 'currency' must be a string with at least 3 characters."
            )
        if self.analysis_type != "net-benefit-first":
            raise_input_error(
                "DecisionProblem 'analysis_type' must be 'net-benefit-first'."
            )
        if not isinstance(self.interventions, list) or not self.interventions:
            raise_input_error(
                "DecisionProblem 'interventions' must be a non-empty list of Intervention objects."
            )
        if not all(isinstance(item, Intervention) for item in self.interventions):
            raise_input_error(
                "All elements in 'interventions' must be Intervention objects."
            )
        ids = [item.intervention_id for item in self.interventions]
        if len(ids) != len(set(ids)):
            raise_input_error(
                "Intervention IDs within a DecisionProblem must be unique."
            )
        if self.outcome_names is not None:
            if not isinstance(self.outcome_names, list) or not self.outcome_names:
                raise_input_error(
                    "DecisionProblem 'outcome_names' must be a non-empty list of strings if specified."
                )
            if not all(
                isinstance(name, str) and bool(name) for name in self.outcome_names
            ):
                raise_input_error("All outcome names must be non-empty strings.")

    @property
    def reference_intervention(self) -> Intervention | None:
        """Return the designated reference intervention if defined."""
        for intervention in self.interventions:
            if intervention.is_reference:
                return intervention
        return None

    @property
    def intervention_names(self) -> list[str]:
        """Return the list of intervention names in order."""
        return [item.name for item in self.interventions]

    @property
    def intervention_ids(self) -> list[str]:
        """Return the list of intervention IDs in order."""
        return [item.intervention_id for item in self.interventions]

    def to_dict(self) -> dict[str, object]:
        """Serialize decision problem to dictionary conforming to DecisionProblemV1 schema."""
        result: dict[str, object] = {
            "decision_problem_id": self.decision_problem_id,
            "title": self.title,
            "analysis_type": self.analysis_type,
            "currency": self.currency,
            "willingness_to_pay": float(self.willingness_to_pay),
            "interventions": [item.to_dict() for item in self.interventions],
        }
        if self.outcome_names is not None:
            result["outcome_names"] = list(self.outcome_names)
        return result

    @classmethod
    def from_dict(cls, data: object) -> DecisionProblem:
        """Deserialize decision problem from dictionary."""
        if not isinstance(data, dict):
            raise_input_error("DecisionProblem data must be a dictionary.")
        required_keys = [
            "decision_problem_id",
            "title",
            "analysis_type",
            "currency",
            "willingness_to_pay",
            "interventions",
        ]
        for key in required_keys:
            if key not in data:
                raise_input_error(f"DecisionProblem data must include '{key}'.")

        raw_interventions = data["interventions"]
        if not isinstance(raw_interventions, Sequence) or isinstance(
            raw_interventions, (str, bytes)
        ):
            raise_input_error(
                "DecisionProblem 'interventions' must be a sequence of dictionaries."
            )
        interventions = [Intervention.from_dict(item) for item in raw_interventions]
        raw_outcomes = data.get("outcome_names")
        outcome_names = list(raw_outcomes) if raw_outcomes is not None else None

        return cls(
            decision_problem_id=str(data["decision_problem_id"]),
            title=str(data["title"]),
            analysis_type=str(data["analysis_type"]),
            currency=str(data["currency"]),
            willingness_to_pay=float(data["willingness_to_pay"]),
            interventions=interventions,
            outcome_names=outcome_names,
        )
