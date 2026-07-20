# voiage/analysis.py

"""A module for the core decision analysis interface."""

from collections import deque
from collections.abc import Callable, Generator, Sequence
from typing import Any, Protocol
import warnings

import numpy as np

from voiage.config import DEFAULT_DTYPE
from voiage.core.utils import check_input_array
from voiage.exceptions import (
    raise_calculation_error,
    raise_dimension_mismatch_error,
    raise_input_error,
    raise_optional_dependency_error,
)
from voiage.main_backends import get_backend
from voiage.schema import ParameterSet, PortfolioSpec, PortfolioStudy, ValueArray

# Check for JAX availability
JAX_AVAILABLE = False
try:
    import jax.numpy  # noqa: F401

    JAX_AVAILABLE = True
except ImportError:
    pass

SKLEARN_AVAILABLE = False
LinearRegression = None
try:
    from sklearn.linear_model import LinearRegression as SklearnLinearRegression

    LinearRegression = SklearnLinearRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    # sklearn is an optional dependency, only required for EVPPI.
    # Users will be warned if they try to use EVPPI without it.
    pass


class RegressionModelProtocol(Protocol):
    """Minimal regression model contract used by EVPPI helpers."""

    def fit(self, x: np.ndarray, y: np.ndarray) -> object:
        """Fit the model."""

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict outputs for the given design matrix."""


class DecisionAnalysis:
    """Stateful decision-analysis interface for VOI calculations.

    The class wraps a net-benefit surface and optional parameter samples, then
    exposes the core EVPI/EVPPI/EVSI and downstream analysis methods through a
    single object. It also manages backend selection, caching, and streaming
    updates for callers that accumulate data over time.

    Parameters
    ----------
    nb_array : numpy.ndarray or ValueArray
        Net-benefit samples with shape ``(n_samples, n_strategies)``.
    parameter_samples : numpy.ndarray, ParameterSet, dict[str, numpy.ndarray], optional
        Optional parameter samples used by EVPPI, EVSI, and related methods.
    backend : str, optional
        Backend name to use. If omitted, the backend is auto-detected.
    use_jit : bool, default=False
        Enable JAX JIT compilation where available.
    streaming_window_size : int, optional
        If provided, enable rolling-window buffering for incremental updates.
    enable_caching : bool, default=False
        Cache intermediate results when repeated calculations are expected.

    Attributes
    ----------
    nb_array : ValueArray
        Normalized net-benefit container used by the analysis methods.
    parameter_samples : ParameterSet or None
        Normalized parameter samples, or ``None`` if the analysis is
        net-benefit only.
    backend : object
        Selected computational backend implementation.
    use_jit : bool
        Whether JAX JIT compilation is enabled.
    streaming_window_size : int or None
        Size of the streaming buffer, if enabled.
    enable_caching : bool
        Whether the instance cache is active.

    Examples
    --------
    >>> import numpy as np
    >>> from voiage.analysis import DecisionAnalysis
    >>> analysis = DecisionAnalysis(np.array([[10.0, 12.0], [11.0, 9.5]]))
    >>> round(analysis.evpi(), 2)
    1.0
    """

    def __init__(
        self,
        nb_array: np.ndarray | ValueArray,
        parameter_samples: np.ndarray
        | ParameterSet
        | dict[str, np.ndarray]
        | None = None,
        backend: str | None = None,
        use_jit: bool = False,
        streaming_window_size: int | None = None,
        enable_caching: bool = False,
    ):
        if isinstance(nb_array, ValueArray):
            self.nb_array = nb_array
        elif isinstance(nb_array, np.ndarray):
            if nb_array.ndim == 3:
                self.nb_array = ValueArray.from_numpy_perspectives(nb_array)
            else:
                # Assuming strategy names are not provided
                self.nb_array = ValueArray.from_numpy(nb_array)
        else:
            raise_input_error("`nb_array` must be a NumPy array or ValueArray object.")

        if parameter_samples is not None:
            if isinstance(parameter_samples, ParameterSet):
                self.parameter_samples: ParameterSet | None = parameter_samples
            elif isinstance(parameter_samples, (dict, np.ndarray)):
                self.parameter_samples = ParameterSet.from_numpy_or_dict(
                    parameter_samples
                )
            else:
                raise_input_error(
                    f"`parameter_samples` must be a NumPy array, ParameterSet, or Dict. Got {type(parameter_samples)}."
                )
        else:
            self.parameter_samples = None

        # Set the computational backend
        # Auto-detect JAX arrays and select JAX backend if appropriate
        if backend is None:
            backend = self._auto_detect_backend(nb_array, parameter_samples)

        self.backend: Any = get_backend(backend)
        self.use_jit = use_jit

        # Streaming data support
        self.streaming_window_size = streaming_window_size
        self._streaming_data_buffer: deque[np.ndarray] | None = None
        self._streaming_parameter_buffer: deque[Any] | None = None
        if streaming_window_size is not None:
            self._initialize_streaming_buffers()

        # Caching support
        self.enable_caching = enable_caching
        self._cache: dict[str, Any] | None = {} if enable_caching else None

        # Optional caller-provided decision scoring function used by higher-level wrappers.
        self.decision_function: Any | None = None

        # Track data changes to invalidate cache
        self._data_hash = self._compute_data_hash()

    def _auto_detect_backend(
        self,
        nb_array: np.ndarray | ValueArray,
        parameter_samples: np.ndarray
        | ParameterSet
        | dict[str, np.ndarray]
        | None = None,
    ) -> str:
        """
        Automatically detect and select the appropriate backend based on input data.

        If JAX arrays are detected in the input data, select JAX backend.
        Otherwise, use the default NumPy backend.

        Args:
            nb_array: Net benefit array or ValueArray
            parameter_samples: Parameter samples

        Returns
        -------
            Backend name to use
        """
        # Check if JAX is available
        if not JAX_AVAILABLE:
            return "numpy"

        # Check net benefit array for JAX arrays
        if isinstance(nb_array, ValueArray):
            jax_values = nb_array.jax_values
            if jax_values is not None:
                return "jax"
        elif isinstance(nb_array, np.ndarray):
            # Check if numpy array is actually a JAX array in disguise
            if (
                hasattr(nb_array, "dtype")
                and hasattr(nb_array, "shape")
                and (hasattr(nb_array, "device") or hasattr(nb_array, "aval"))
            ):
                return "jax"

        # Check parameter samples for JAX arrays
        if parameter_samples is not None:
            if isinstance(parameter_samples, ParameterSet):
                jax_params = parameter_samples.jax_parameters
                if jax_params is not None and jax_params:
                    return "jax"
            elif isinstance(parameter_samples, dict):
                # Check dictionary for JAX arrays
                for values in parameter_samples.values():
                    if hasattr(values, "device") or hasattr(values, "aval"):
                        return "jax"

        # Default to NumPy backend
        return "numpy"

    def _compute_data_hash(self) -> int:
        """
        Compute a hash of the current data to detect changes.

        Returns
        -------
            int: Hash value representing the current data state
        """
        # Create a hash based on the net benefit array and parameter samples
        hash_components = []

        # Hash net benefit array
        nb_values = self.nb_array.numpy_values
        hash_components.append(hash(nb_values.tobytes()))

        # Hash parameter samples if they exist
        if (
            self.parameter_samples is not None
            and hasattr(self.parameter_samples, "parameters")
            and isinstance(self.parameter_samples.parameters, dict)
        ):
            for param_name, param_values in self.parameter_samples.parameters.items():
                hash_components.append(hash(param_name))
                hash_components.append(hash(param_values.tobytes()))

        # Combine all hash components
        return hash(tuple(hash_components))

    def _invalidate_cache_if_needed(self) -> None:
        """Invalidate cache if data has changed."""
        if self.enable_caching and self._cache is not None:
            current_hash = self._compute_data_hash()
            if current_hash != self._data_hash:
                self._cache.clear()
                self._data_hash = current_hash

    def _cache_get(self, key: str) -> float | None:
        """
        Get a value from the cache.

        Args:
            key: Cache key

        Returns
        -------
            Cached value or None if not found
        """
        if self.enable_caching and self._cache is not None:
            self._invalidate_cache_if_needed()
            return self._cache.get(key)
        return None

    def _cache_set(self, key: str, value: object) -> None:
        """
        Set a value in the cache.

        Args:
            key: Cache key
            value: Value to cache
        """
        if self.enable_caching and self._cache is not None:
            self._invalidate_cache_if_needed()
            self._cache[key] = value

    def _initialize_streaming_buffers(self) -> None:
        """Initialize buffers for streaming data."""
        if self.streaming_window_size is not None:
            # Initialize buffers as deques with maximum length
            self._streaming_data_buffer = deque(maxlen=self.streaming_window_size)
            if self.parameter_samples is not None:
                self._streaming_parameter_buffer = deque(
                    maxlen=self.streaming_window_size
                )

    def update_with_new_data(
        self,
        new_nb_data: np.ndarray | ValueArray,
        new_parameter_samples: np.ndarray
        | ParameterSet
        | dict[str, np.ndarray]
        | None = None,
    ) -> None:
        """
        Update the decision analysis with new data for streaming VOI calculations.

        Args:
            new_nb_data: New net benefit data to add
            new_parameter_samples: New parameter samples corresponding to the net benefit data
        """
        # Convert new data to appropriate format
        if isinstance(new_nb_data, ValueArray):
            new_nb_values = new_nb_data.numpy_values
        elif isinstance(new_nb_data, np.ndarray):
            new_nb_values = new_nb_data
        else:
            raise_input_error(
                "`new_nb_data` must be a NumPy array or ValueArray object."
            )

        # Validate dimensions
        if new_nb_values.ndim != 2:
            raise_dimension_mismatch_error(
                "New net benefit data must be 2-dimensional."
            )

        # If we have streaming buffers, add the new data
        if self._streaming_data_buffer is not None:
            # Add new data to buffer
            for i in range(new_nb_values.shape[0]):
                self._streaming_data_buffer.append(new_nb_values[i : i + 1, :])

                # If we have parameter samples, add them too
                if (
                    new_parameter_samples is not None
                    and self._streaming_parameter_buffer is not None
                ):
                    # Convert parameter samples to appropriate format
                    if isinstance(new_parameter_samples, ParameterSet):
                        param_values = new_parameter_samples
                    elif isinstance(new_parameter_samples, (dict, np.ndarray)):
                        param_values = ParameterSet.from_numpy_or_dict(
                            new_parameter_samples
                        )
                    else:
                        raise_input_error(
                            f"`new_parameter_samples` must be a NumPy array, ParameterSet, or Dict. Got {type(new_parameter_samples)}."
                        )

                    # Add parameter sample to buffer
                    if hasattr(param_values, "parameters") and isinstance(
                        param_values.parameters, dict
                    ):
                        # Extract the i-th sample for each parameter
                        sample_dict = {}
                        for param_name, param_array in param_values.parameters.items():
                            if i < len(param_array):
                                sample_dict[param_name] = param_array[i : i + 1]
                        self._streaming_parameter_buffer.append(sample_dict)

            # Update the main data arrays with buffered data
            self._update_main_arrays_from_buffer()
        else:
            # If no streaming buffers, just append to existing data
            self._append_to_existing_data(new_nb_values, new_parameter_samples)

    def _update_main_arrays_from_buffer(self) -> None:
        """Update the main data arrays from the streaming buffers."""
        if self._streaming_data_buffer:
            # Convert buffered data to numpy array
            buffered_data = np.vstack(list(self._streaming_data_buffer))
            self.nb_array = ValueArray.from_numpy(buffered_data)

        if self._streaming_parameter_buffer and self.parameter_samples:
            # Combine all buffered parameter samples.
            combined_params_lists: dict[str, list[float]] = {}
            for buffered_sample in self._streaming_parameter_buffer:
                for param_name, param_value in buffered_sample.items():
                    combined_params_lists.setdefault(param_name, []).extend(param_value)

            # Convert to numpy arrays.
            combined_params: dict[str, np.ndarray] = {
                param_name: np.array(param_values)
                for param_name, param_values in combined_params_lists.items()
            }

            self.parameter_samples = ParameterSet.from_numpy_or_dict(combined_params)

    def _append_to_existing_data(
        self,
        new_nb_values: np.ndarray,
        new_parameter_samples: np.ndarray
        | ParameterSet
        | dict[str, np.ndarray]
        | None = None,
    ) -> None:
        """Append new data to existing data arrays."""
        # Append to net benefit data
        current_nb_values = self.nb_array.numpy_values
        combined_nb_values = np.vstack([current_nb_values, new_nb_values])
        self.nb_array = ValueArray.from_numpy(combined_nb_values)

        # Append to parameter samples if provided
        if new_parameter_samples is not None and self.parameter_samples is not None:
            # Convert new parameter samples to appropriate format
            if isinstance(new_parameter_samples, ParameterSet):
                new_params = new_parameter_samples
            elif isinstance(new_parameter_samples, (dict, np.ndarray)):
                new_params = ParameterSet.from_numpy_or_dict(new_parameter_samples)
            else:
                raise_input_error(
                    f"`new_parameter_samples` must be a NumPy array, ParameterSet, or Dict. Got {type(new_parameter_samples)}."
                )

            # Combine parameter samples
            if hasattr(new_params, "parameters") and isinstance(
                new_params.parameters, dict
            ):
                combined_params = {}
                for param_name in self.parameter_samples.parameters:
                    if param_name in new_params.parameters:
                        combined_params[param_name] = np.concatenate(
                            [
                                self.parameter_samples.parameters[param_name],
                                new_params.parameters[param_name],
                            ]
                        )
                    else:
                        combined_params[param_name] = self.parameter_samples.parameters[
                            param_name
                        ]
                self.parameter_samples = ParameterSet.from_numpy_or_dict(
                    combined_params
                )

    def streaming_evpi(self) -> Generator[float, None, None]:
        """Yield EVPI repeatedly for the current data state.

        Yields
        ------
        float
            EVPI value calculated from the current buffered data.
        """
        while True:
            # Calculate EVPI with current data
            evpi_value = self.evpi()
            yield evpi_value

    def streaming_evppi(self) -> Generator[float, None, None]:
        """Yield EVPPI repeatedly for the current data state.

        Yields
        ------
        float
            EVPPI value calculated from the current buffered data.
        """
        while True:
            # Calculate EVPPI with current data
            evppi_value = self.evppi()
            yield evppi_value

    def _scale_to_population(
        self,
        value: float,
        population: float,
        time_horizon: float,
        discount_rate: float | None,
        metric_name: str,
    ) -> float:
        """Scale a per-decision value to a population level."""
        # Validate population parameter
        if not isinstance(population, (int, float)):
            raise_input_error(f"Population must be a number. Got {type(population)}.")
        if population <= 0:
            raise_input_error(f"Population must be positive. Got {population}.")
        if not np.isfinite(population):
            raise_input_error(f"Population must be finite. Got {population}.")

        # Validate time_horizon parameter
        if not isinstance(time_horizon, (int, float)):
            raise_input_error(
                f"Time horizon must be a number. Got {type(time_horizon)}."
            )
        if time_horizon <= 0:
            raise_input_error(f"Time horizon must be positive. Got {time_horizon}.")
        if not np.isfinite(time_horizon):
            raise_input_error(f"Time horizon must be finite. Got {time_horizon}.")

        # Validate discount_rate parameter
        current_dr = discount_rate
        if current_dr is None:
            current_dr = 0.0

        if not isinstance(current_dr, (int, float)):
            raise_input_error(
                f"Discount rate must be a number. Got {type(current_dr)}."
            )
        if not (0 <= current_dr <= 1):
            raise_input_error(
                f"Discount rate must be between 0 and 1. Got {current_dr}."
            )
        if not np.isfinite(current_dr):
            raise_input_error(f"Discount rate must be finite. Got {current_dr}.")

        # Calculate annuity factor
        if current_dr == 0:
            annuity_factor = time_horizon
        else:
            annuity_factor = (1 - (1 + current_dr) ** (-time_horizon)) / current_dr

        result = float(value * population * annuity_factor)

        # Validate result
        if not np.isfinite(result):
            raise_calculation_error(f"Calculated {metric_name} is not finite: {result}")

        return result

    def _prepare_evppi_inputs(
        self,
        parameters_of_interest: list[str] | None,
        nb_values: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Validate and extract inputs for EVPPI regression."""
        if self.parameter_samples is None:
            raise_input_error(
                "`parameter_samples` must be provided for EVPPI calculation."
            )

        if parameters_of_interest is None:
            parameters_of_interest = list(self.parameter_samples.parameter_names)

        if not isinstance(parameters_of_interest, list):
            raise_input_error(
                "`parameters_of_interest` must be a list of parameter names."
            )

        # Validate that all parameters of interest exist in the parameter set
        param_names = self.parameter_samples.parameter_names
        for param in parameters_of_interest:
            if param not in param_names:
                raise_input_error(
                    "All `parameters_of_interest` must be in the ParameterSet"
                )

        check_input_array(nb_values, expected_ndim=2, name="nb_array")

        x_all = self._get_parameter_samples_as_ndarray()

        # Select only the columns corresponding to parameters of interest
        x_indices = [
            i for i, name in enumerate(param_names) if name in parameters_of_interest
        ]
        x = x_all[:, x_indices]

        return x, nb_values

    def _fit_evppi_regression(
        self,
        x: np.ndarray,
        nb_values: np.ndarray,
        n_regression_samples: int | None,
        regression_model: RegressionModelProtocol
        | type[RegressionModelProtocol]
        | None,
    ) -> np.ndarray:
        """Subsample and fit the regression model for each strategy."""
        if not SKLEARN_AVAILABLE:
            raise_optional_dependency_error(
                "scikit-learn is required for the Python EVPPI estimator. "
                "Please install it (e.g., `pip install scikit-learn`)."
            )
        n_samples, n_strategies = nb_values.shape

        if n_regression_samples is not None:
            if not isinstance(n_regression_samples, int):
                raise_input_error(
                    f"n_regression_samples must be an integer. Got {type(n_regression_samples)}."
                )
            if n_regression_samples <= 0:
                raise_input_error(
                    f"n_regression_samples must be positive. Got {n_regression_samples}."
                )
            if not np.isfinite(n_regression_samples):
                raise_input_error(
                    f"n_regression_samples must be finite. Got {n_regression_samples}."
                )
            if n_regression_samples > n_samples:
                raise_input_error(
                    f"n_regression_samples ({n_regression_samples}) cannot exceed total samples ({n_samples})."
                )

            # Subsample for regression fitting
            indices = np.random.choice(n_samples, n_regression_samples, replace=False)
            x_fit = x[indices, :]
            nb_values_fit = nb_values[indices, :]
        else:
            x_fit = x
            nb_values_fit = nb_values

        # Fit regression model for each strategy
        fitted_nb_on_params = np.zeros_like(nb_values, dtype=DEFAULT_DTYPE)

        for i in range(n_strategies):
            y_fit = nb_values_fit[:, i]

            if regression_model is None:
                model: RegressionModelProtocol = SklearnLinearRegression()
            elif isinstance(regression_model, type):
                model = regression_model()
            else:
                model = regression_model

            try:
                model.fit(x_fit, y_fit)
                # Predict on the full set of parameter samples X
                fitted_nb_on_params[:, i] = model.predict(x)
            except Exception as e:
                raise_calculation_error(
                    f"Error during regression for strategy {i}: {e}", e
                )

        return fitted_nb_on_params

    def evpi(
        self,
        population: float | None = None,
        time_horizon: float | None = None,
        discount_rate: float | None = None,
        chunk_size: int | None = None,
    ) -> float:
        r"""Calculate expected value of perfect information.

        Parameters
        ----------
        population : float, optional
            Population size for population-scaled EVPI.
        time_horizon : float, optional
            Time horizon in years for population scaling.
        discount_rate : float, optional
            Annual discount rate used for population scaling.
        chunk_size : int, optional
            Optional chunk size for incremental computation.

        Returns
        -------
        float
            Per-decision EVPI unless population scaling is requested.

        Notes
        -----
        EVPI is computed as :math:`E[\\max_d NB_d] - \\max_d E[NB_d]`.
        """
        # Check cache first
        cache_key = f"evpi_{population}_{time_horizon}_{discount_rate}_{chunk_size}"
        cached_result = self._cache_get(cache_key)
        if cached_result is not None:
            return float(cached_result)

        nb_values = self.nb_array.numpy_values
        check_input_array(nb_values, expected_ndim=2, name="nb_array", allow_empty=True)

        if nb_values.size == 0:
            return 0.0
        if nb_values.shape[1] == 1:  # Single strategy
            return 0.0

        try:
            # Use incremental computation if chunk_size is specified
            if chunk_size is not None:
                per_decision_evpi = self._incremental_evpi(nb_values, chunk_size)
            elif type(self.backend).__name__ == "NumpyBackend" and not self.use_jit:
                # The stable NumPy facade now delegates its canonical kernel to
                # Rust.  Keep the backend fallback only for environments that
                # intentionally lack the optional native extension.
                try:
                    from voiage import _runtime

                    per_decision_evpi = _runtime.compute_evpi(nb_values.tolist())
                except (ModuleNotFoundError, AttributeError):
                    warnings.warn(
                        "Python EVPI fallback is transitional; the Rust kernel is "
                        "the v1 execution target.",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                    per_decision_evpi = self.backend.evpi(nb_values)
            # Use the selected backend for computation
            elif self.use_jit and hasattr(self.backend, "evpi_jit"):
                # Use JIT compilation if available and requested
                per_decision_evpi = self.backend.evpi_jit(nb_values)
            else:
                # Use regular computation
                per_decision_evpi = self.backend.evpi(nb_values)

            # EVPI should theoretically be non-negative. Small negative values can occur due to float precision.
            per_decision_evpi = max(0.0, float(per_decision_evpi))

        except Exception as e:
            raise_calculation_error(f"Error during EVPI calculation: {e}", e)

        if population is not None and time_horizon is not None:
            result = self._scale_to_population(
                per_decision_evpi, population, time_horizon, discount_rate, "EVPI"
            )
            self._cache_set(cache_key, result)
            return result

        if (
            population is not None
            or time_horizon is not None
            or discount_rate is not None
        ):
            raise_input_error(
                "To calculate population EVPI, 'population' and 'time_horizon' must be provided. "
                "'discount_rate' is optional (defaults to 0 if not provided)."
            )

        # Cache the result
        self._cache_set(cache_key, float(per_decision_evpi))
        return float(per_decision_evpi)

    def calculate_evpi(
        self,
        population: float | None = None,
        time_horizon: float | None = None,
        discount_rate: float | None = None,
        chunk_size: int | None = None,
    ) -> float:
        """Compatibility wrapper around :meth:`evpi`.

        Parameters
        ----------
        population : float, optional
            Population size for population scaling.
        time_horizon : float, optional
            Time horizon in years for population scaling.
        discount_rate : float, optional
            Annual discount rate used for population scaling.
        chunk_size : int, optional
            Optional chunk size for incremental computation.

        Returns
        -------
        float
            EVPI value returned by :meth:`evpi`.
        """
        return self.evpi(
            population=population,
            time_horizon=time_horizon,
            discount_rate=discount_rate,
            chunk_size=chunk_size,
        )

    def _incremental_evpi(self, nb_values: np.ndarray, chunk_size: int) -> float:
        """
        Calculate EVPI incrementally in chunks to handle large datasets.

        Args:
            nb_values: Net benefit array
            chunk_size: Size of chunks for processing

        Returns
        -------
            float: Calculated EVPI value
        """
        n_samples, n_strategies = nb_values.shape

        # Initialize accumulators for incremental computation
        # For E[max(NB)] we need to track the sum of max values
        max_nb_sum = 0.0
        # For max(E[NB]) we need to track the sum for each strategy
        strategy_sums = np.zeros(n_strategies, dtype=DEFAULT_DTYPE)

        # Process data in chunks
        n_processed = 0
        for start_idx in range(0, n_samples, chunk_size):
            end_idx = min(start_idx + chunk_size, n_samples)
            chunk = nb_values[start_idx:end_idx]
            chunk_size_actual = chunk.shape[0]

            # Calculate max net benefit for each sample in chunk
            chunk_max_nb = np.max(chunk, axis=1)
            max_nb_sum += np.sum(chunk_max_nb)

            # Calculate sum of net benefits for each strategy
            chunk_strategy_sums = np.sum(chunk, axis=0)
            strategy_sums += chunk_strategy_sums

            n_processed += chunk_size_actual

        # Calculate final results
        expected_max_nb = max_nb_sum / n_processed
        expected_nb_options = strategy_sums / n_processed
        max_expected_nb = np.max(expected_nb_options)

        return float(expected_max_nb - max_expected_nb)

    def evppi(
        self,
        parameters_of_interest: list[str] | None = None,
        population: float | None = None,
        time_horizon: float | None = None,
        discount_rate: float | None = None,
        n_regression_samples: int | None = None,
        regression_model: RegressionModelProtocol
        | type[RegressionModelProtocol]
        | None = None,
        chunk_size: int | None = None,
    ) -> float:
        """Calculate expected value of partial perfect information.

        Parameters
        ----------
        parameters_of_interest : list[str], optional
            Parameter names to analyze. Defaults to all parameters.
        population : float, optional
            Population size for population scaling.
        time_horizon : float, optional
            Time horizon in years for population scaling.
        discount_rate : float, optional
            Annual discount rate used for population scaling.
        n_regression_samples : int, optional
            Number of samples used to fit the regression approximation.
        regression_model : RegressionModelProtocol or type, optional
            Optional scikit-learn-compatible regression model.
        chunk_size : int, optional
            Optional chunk size for incremental computation of the baseline term.

        Returns
        -------
        float
            Per-decision EVPPI unless population scaling is requested.

        Notes
        -----
        EVPPI is computed by regressing net benefit on the parameters of
        interest and comparing the conditional and unconditional maxima.
        """
        # Check cache first
        cache_key = (
            "evppi_"
            f"{tuple(parameters_of_interest) if parameters_of_interest is not None else '__all__'}_"
            f"{population}_{time_horizon}_{discount_rate}_{n_regression_samples}_{chunk_size}_{regression_model!s}"
        )
        cached_result = self._cache_get(cache_key)
        if cached_result is not None:
            return float(cached_result)

        # Check cache first
        cache_key = (
            "evppi_"
            f"{tuple(parameters_of_interest) if parameters_of_interest is not None else '__all__'}_"
            f"{population}_{time_horizon}_{discount_rate}_{n_regression_samples}_{chunk_size}_{regression_model!s}"
        )
        cached_result = self._cache_get(cache_key)
        if cached_result is not None:
            return float(cached_result)

        nb_values = self.nb_array.numpy_values
        if nb_values.size == 0:
            raise_input_error("`nb_array` cannot be empty")
        _, n_strategies = nb_values.shape
        if n_strategies <= 1:
            return 0.0

        x, nb_values = self._prepare_evppi_inputs(parameters_of_interest, nb_values)

        native_result: float | None = None
        if (
            regression_model is None
            and n_regression_samples is None
            and chunk_size is None
        ):
            from voiage import _runtime

            try:
                native_result = _runtime.compute_evppi(nb_values.tolist(), x.tolist())
            except (ModuleNotFoundError, AttributeError):
                # The optional extension is unavailable or predates EVPPI;
                # retain the established Python estimator in that case.
                native_result = None

        if native_result is None:
            fitted_nb_on_params = self._fit_evppi_regression(
                x, nb_values, n_regression_samples, regression_model
            )
            # Calculate E_p [max_d E[NB_d|p]].
            e_max_enb_conditional = np.mean(np.max(fitted_nb_on_params, axis=1))
        else:
            e_max_enb_conditional = native_result + float(
                np.max(np.mean(nb_values, axis=0))
            )

        # Calculate max_d E[NB_d] using incremental computation if chunk_size is specified
        if chunk_size is not None:
            max_e_nb = self._incremental_max_expected_nb(nb_values, chunk_size)
        else:
            # Standard calculation
            max_e_nb = float(np.max(np.mean(nb_values, axis=0)))

        per_decision_evppi = float(e_max_enb_conditional - max_e_nb)

        # EVPPI should theoretically be non-negative. Small negative values can occur due to float precision or regression error.
        per_decision_evppi = max(0.0, per_decision_evppi)

        if population is not None and time_horizon is not None:
            result = self._scale_to_population(
                per_decision_evppi, population, time_horizon, discount_rate, "EVPPI"
            )
            self._cache_set(cache_key, result)
            return result

        if (
            population is not None
            or time_horizon is not None
            or discount_rate is not None
        ):
            raise_input_error(
                "To calculate population EVPPI, 'population' and 'time_horizon' must be provided. "
                "'discount_rate' is optional (defaults to 0 if not provided)."
            )

        # Cache the result
        self._cache_set(cache_key, float(per_decision_evppi))
        return float(per_decision_evppi)

    def enbs(
        self,
        research_cost: float,
        strategy_of_interest: int | str | None = None,
        population: float | None = None,
        time_horizon: float | None = None,
        discount_rate: float | None = None,
    ) -> float:
        """Calculate expected net benefit of sampling.

        Parameters
        ----------
        research_cost : float
            Total cost of the proposed research study.
        strategy_of_interest : int or str, optional
            Optional strategy selector for downstream decision reporting.
        population : float, optional
            Population size for population scaling.
        time_horizon : float, optional
            Time horizon in years for population scaling.
        discount_rate : float, optional
            Annual discount rate used for population scaling.

        Returns
        -------
        float
            ENBS value, clipped at zero, unless population scaling is requested.
        """
        # Check cache first
        cache_key = f"enbs_{research_cost}_{strategy_of_interest}_{population}_{time_horizon}_{discount_rate}"
        cached_result = self._cache_get(cache_key)
        if cached_result is not None:
            return float(cached_result)

        nb_values = self.nb_array.numpy_values
        check_input_array(nb_values, expected_ndim=2, name="nb_array", allow_empty=True)

        if nb_values.size == 0:
            return 0.0
        if nb_values.shape[1] == 1:  # Single strategy
            return 0.0

        try:
            # Use the selected backend for computation
            if self.use_jit and hasattr(self.backend, "enbs_simple_jit"):
                # Use JIT compilation if available and requested
                per_decision_enbs = self.backend.enbs_simple_jit(
                    nb_values, research_cost
                )
            else:
                # Use regular computation
                per_decision_enbs = self.backend.enbs_simple(nb_values, research_cost)

            # ENBS should be non-negative (won't conduct research if it costs more than it's worth)
            per_decision_enbs = max(0.0, float(per_decision_enbs))

        except Exception as e:
            raise_calculation_error(f"Error during ENBS calculation: {e}", e)

        if population is not None and time_horizon is not None:
            result = self._scale_to_population(
                per_decision_enbs, population, time_horizon, discount_rate, "ENBS"
            )
            self._cache_set(cache_key, result)
            return result

        if (
            population is not None
            or time_horizon is not None
            or discount_rate is not None
        ):
            raise_input_error(
                "To calculate population ENBS, 'population' and 'time_horizon' must be provided. "
                "'discount_rate' is optional (defaults to 0 if not provided)."
            )

        # Cache the result
        self._cache_set(cache_key, float(per_decision_enbs))
        return float(per_decision_enbs)

    def ceaf(
        self,
        wtp_thresholds: Sequence[float],
        strategy_names: Sequence[str] | None = None,
        confidence_level: float = 0.95,
    ) -> Any:
        """Calculate the cost-effectiveness acceptability frontier.

        Parameters
        ----------
        wtp_thresholds : sequence of float
            Willingness-to-pay thresholds to evaluate.
        strategy_names : sequence of str, optional
            Optional strategy labels.
        confidence_level : float, default=0.95
            Confidence level used to build the probability band.

        Returns
        -------
        object
            CEAF result from :func:`voiage.methods.ceaf.calculate_ceaf`.
        """
        from voiage.methods.ceaf import calculate_ceaf

        return calculate_ceaf(
            self.nb_array,
            wtp_thresholds=wtp_thresholds,
            strategy_names=strategy_names,
            confidence_level=confidence_level,
        )

    def dominance(
        self,
        costs: Sequence[float],
        effects: Sequence[float],
        strategy_names: Sequence[str] | None = None,
    ) -> Any:
        """Calculate strong and extended dominance for cost/effect pairs.

        Parameters
        ----------
        costs : sequence of float
            Strategy costs.
        effects : sequence of float
            Strategy effects.
        strategy_names : sequence of str, optional
            Optional strategy labels.

        Returns
        -------
        object
            Dominance result from :func:`voiage.methods.dominance.calculate_dominance`.
        """
        from voiage.methods.dominance import calculate_dominance

        return calculate_dominance(
            costs=costs,
            effects=effects,
            strategy_names=strategy_names,
        )

    def value_of_heterogeneity(
        self,
        subgroups: Sequence[object],
        strategy_names: Sequence[str] | None = None,
        n_bins: int | None = None,
    ) -> Any:
        """Calculate the value of subgroup-specific decisions.

        Parameters
        ----------
        subgroups : sequence of object
            Subgroup label for each sample.
        strategy_names : sequence of str, optional
            Optional strategy labels.
        n_bins : int, optional
            Quantile bin count for numeric subgroup values.

        Returns
        -------
        object
            Heterogeneity result from :func:`voiage.methods.heterogeneity.value_of_heterogeneity`.
        """
        from voiage.methods.heterogeneity import value_of_heterogeneity

        return value_of_heterogeneity(
            self.nb_array,
            subgroups=subgroups,
            strategy_names=strategy_names,
            n_bins=n_bins,
        )

    def value_of_distributional_equity(
        self,
        subgroups: Sequence[object],
        strategy_names: Sequence[str] | None = None,
        equity_weights: Sequence[float] | dict[str, float] | None = None,
        n_bins: int | None = None,
    ) -> Any:
        """Calculate the value of distributional and equity-weighted decision tailoring."""
        from voiage.methods.distributional import value_of_distributional_equity

        return value_of_distributional_equity(
            self.nb_array,
            subgroups=subgroups,
            strategy_names=strategy_names,
            equity_weights=equity_weights,
            n_bins=n_bins,
        )

    def value_of_ambiguity_distribution_shift(
        self,
        shift_weights: Sequence[Sequence[float]],
        strategy_names: Sequence[str] | None = None,
        scenario_names: Sequence[str] | None = None,
        scenario_probabilities: Sequence[float] | None = None,
        ambiguity_radius: float = 0.0,
        information_cost: float = 0.0,
    ) -> Any:
        """Calculate robust VOI under ambiguity and distribution shift."""
        from voiage.methods.ambiguity_distribution_shift import (
            value_of_ambiguity_distribution_shift,
        )

        return value_of_ambiguity_distribution_shift(
            self.nb_array,
            shift_weights=shift_weights,
            strategy_names=list(strategy_names) if strategy_names else None,
            scenario_names=list(scenario_names) if scenario_names else None,
            scenario_probabilities=scenario_probabilities,
            ambiguity_radius=ambiguity_radius,
            information_cost=information_cost,
        )

    def value_of_adaptive_learning_bandit(
        self,
        policy: str = "ucb",
        horizon: int | None = None,
        exploration_cost: float = 0.0,
        epsilon: float = 0.1,
        confidence: float = 2.0,
        stop_regret: float | None = None,
        arm_names: Sequence[str] | None = None,
        seed: int = 0,
    ) -> Any:
        """Calculate fixture-backed value of adaptive bandit learning."""
        from voiage.methods.adaptive_learning_bandit import (
            value_of_adaptive_learning_bandit,
        )

        return value_of_adaptive_learning_bandit(
            self.nb_array.numpy_values.T,
            policy=policy,
            horizon=horizon,
            exploration_cost=exploration_cost,
            epsilon=epsilon,
            confidence=confidence,
            stop_regret=stop_regret,
            arm_names=list(arm_names)
            if arm_names
            else list(self.nb_array.strategy_names),
            seed=seed,
        )

    def value_of_capacity_budget_constrained(self, **kwargs: object) -> object:
        """Evaluate value of information under resource constraints."""
        from voiage.methods.capacity_budget_constrained import (
            value_of_capacity_budget_constrained,
        )

        return value_of_capacity_budget_constrained(**kwargs)

    def value_of_ai_assisted_evidence_triage(self, **kwargs: object) -> object:
        """Evaluate the decision value of human-in-the-loop evidence triage."""
        from voiage.methods.ai_assisted_evidence_triage import (
            value_of_ai_assisted_evidence_triage,
        )

        return value_of_ai_assisted_evidence_triage(**kwargs)

    def value_of_federated_privacy_preserving(self, **kwargs: object) -> object:
        """Evaluate site-local evidence under privacy-preserving aggregation."""
        from voiage.methods.federated_privacy_preserving import (
            value_of_federated_privacy_preserving,
        )

        return value_of_federated_privacy_preserving(**kwargs)

    def value_of_explainability_transparency(self, **kwargs: object) -> object:
        """Evaluate adoption and governance value of transparent explanations."""
        from voiage.methods.explainability_transparency import (
            value_of_explainability_transparency,
        )

        return value_of_explainability_transparency(**kwargs)

    def value_of_interoperability_standardization(self, **kwargs: object) -> object:
        """Evaluate harmonization and cross-site evidence reuse value."""
        from voiage.methods.interoperability_standardization import (
            value_of_interoperability_standardization,
        )

        return value_of_interoperability_standardization(**kwargs)

    def value_of_regulatory_market_access(self, **kwargs: object) -> object:
        """Evaluate regulatory approval, reimbursement, and access value."""
        from voiage.methods.regulatory_market_access import (
            value_of_regulatory_market_access,
        )

        return value_of_regulatory_market_access(**kwargs)

    def value_of_replication_reproducibility(self, **kwargs: object) -> object:
        """Evaluate replication and reproducibility information value."""
        from voiage.methods.replication_reproducibility import (
            value_of_replication_reproducibility,
        )

        return value_of_replication_reproducibility(**kwargs)

    def value_of_evidence_obsolescence_refresh(self, **kwargs: object) -> object:
        """Evaluate evidence obsolescence and refresh information value."""
        from voiage.methods.evidence_obsolescence_refresh import (
            value_of_evidence_obsolescence_refresh,
        )

        return value_of_evidence_obsolescence_refresh(**kwargs)

    def value_of_strategic_behavior(self, **kwargs: object) -> object:
        """Evaluate strategic behavior and game-theoretic information value."""
        from voiage.methods.strategic_behavior import value_of_strategic_behavior

        return value_of_strategic_behavior(**kwargs)

    def value_of_equity_information(
        self,
        subgroups: Sequence[object],
        equity_weights: Sequence[float],
        resolved_equity_weights: Sequence[Sequence[float]],
        scenario_probabilities: Sequence[float] | None = None,
        information_cost: float = 0.0,
        strategy_names: Sequence[str] | None = None,
        policy_strata: Sequence[str] | None = None,
    ) -> Any:
        """Calculate the value of resolving equity-relevant uncertainty."""
        from voiage.methods.equity_information import value_of_equity_information

        return value_of_equity_information(
            self.nb_array,
            subgroups=subgroups,
            equity_weights=equity_weights,
            resolved_equity_weights=resolved_equity_weights,
            scenario_probabilities=scenario_probabilities,
            information_cost=information_cost,
            strategy_names=list(strategy_names) if strategy_names else None,
            policy_strata=list(policy_strata) if policy_strata else None,
        )

    def value_of_implementation(
        self,
        uptake: float = 1.0,
        adherence: float = 1.0,
        coverage: float = 1.0,
        implementation_delay: float = 0.0,
        implementation_uncertainty: float = 0.0,
        discount_rate: float = 0.0,
        time_horizon: float | None = None,
        population: float | None = None,
        strategy_names: Sequence[str] | None = None,
    ) -> Any:
        """Calculate implementation-adjusted VOI summaries."""
        from voiage.methods.implementation import value_of_implementation

        return value_of_implementation(
            self.nb_array,
            uptake=uptake,
            adherence=adherence,
            coverage=coverage,
            implementation_delay=implementation_delay,
            implementation_uncertainty=implementation_uncertainty,
            discount_rate=discount_rate,
            time_horizon=time_horizon,
            population=population,
            strategy_names=strategy_names,
        )

    def value_of_perspective(
        self,
        perspectives: Any | None = None,
        strategy_names: Sequence[str] | None = None,
        perspective_names: Sequence[str] | None = None,
        perspective_weights: Sequence[float] | dict[str, float] | None = None,
        reference_perspective: str | int | None = None,
    ) -> Any:
        """Compare decision value across multiple perspectives.

        Parameters
        ----------
        perspectives : :class:`~voiage.methods.perspective.PerspectiveSet` or sequence, optional
            Ordered perspective metadata or perspective identifiers.
        strategy_names : sequence of str, optional
            Optional strategy labels.
        perspective_names : sequence of str, optional
            Optional perspective labels when full metadata is not provided.
        perspective_weights : sequence or dict, optional
            Non-negative weights used for consensus and switching-value
            summaries.
        reference_perspective : str or int, optional
            Reference perspective used for switching-value summaries.

        Returns
        -------
        object
            Result from :func:`voiage.methods.perspective.value_of_perspective`.
        """
        from voiage.methods.perspective import value_of_perspective

        return value_of_perspective(
            self.nb_array,
            perspectives=perspectives,
            strategy_names=strategy_names,
            perspective_names=perspective_names,
            perspective_weights=perspective_weights,
            reference_perspective=reference_perspective,
        )

    def value_of_preference(
        self,
        preference_profiles: Any | None = None,
        strategy_names: Sequence[str] | None = None,
        preference_profile_names: Sequence[str] | None = None,
        preference_profile_weights: Sequence[float] | dict[str, float] | None = None,
        reference_preference_profile: str | int | None = None,
        analysis_id: str | None = None,
        decision_problem_id: str | None = None,
        decision_context: str | None = None,
    ) -> Any:
        """Compare decision value across multiple preference profiles."""
        from voiage.methods.preference import value_of_preference

        return value_of_preference(
            self.nb_array,
            preference_profiles=preference_profiles,
            strategy_names=strategy_names,
            preference_profile_names=preference_profile_names,
            preference_profile_weights=preference_profile_weights,
            reference_preference_profile=reference_preference_profile,
            analysis_id=analysis_id,
            decision_problem_id=decision_problem_id,
            decision_context=decision_context,
        )

    def value_of_model_validation(
        self,
        validation_profiles: Any | None = None,
        strategy_names: Sequence[str] | None = None,
        validation_profile_names: Sequence[str] | None = None,
        validation_profile_weights: Sequence[float] | dict[str, float] | None = None,
        reference_validation_profile: str | int | None = None,
        analysis_id: str | None = None,
        decision_problem_id: str | None = None,
        decision_context: str | None = None,
    ) -> Any:
        """Compare decision value across multiple validation profiles."""
        from voiage.methods.validation import value_of_model_validation

        return value_of_model_validation(
            self.nb_array,
            validation_profiles=validation_profiles,
            strategy_names=strategy_names,
            validation_profile_names=validation_profile_names,
            validation_profile_weights=validation_profile_weights,
            reference_validation_profile=reference_validation_profile,
            analysis_id=analysis_id,
            decision_problem_id=decision_problem_id,
            decision_context=decision_context,
        )

    def value_of_threshold_information(
        self,
        threshold_profiles: Any | None = None,
        strategy_names: Sequence[str] | None = None,
        threshold_profile_names: Sequence[str] | None = None,
        threshold_profile_weights: Sequence[float] | dict[str, float] | None = None,
        reference_threshold_profile: str | int | None = None,
        analysis_id: str | None = None,
        decision_problem_id: str | None = None,
        decision_context: str | None = None,
    ) -> Any:
        """Compare decision value across multiple threshold profiles."""
        from voiage.methods.threshold import value_of_threshold_information

        return value_of_threshold_information(
            self.nb_array,
            threshold_profiles=threshold_profiles,
            strategy_names=strategy_names,
            threshold_profile_names=threshold_profile_names,
            threshold_profile_weights=threshold_profile_weights,
            reference_threshold_profile=reference_threshold_profile,
            analysis_id=analysis_id,
            decision_problem_id=decision_problem_id,
            decision_context=decision_context,
        )

    def portfolio_voi(
        self,
        portfolio_specification: PortfolioSpec,
        study_value_calculator: Callable[[PortfolioStudy], float],
        optimization_method: str = "greedy",
        **kwargs: object,
    ) -> dict[str, object]:
        """Optimize a research portfolio from the analysis surface.

        Parameters
        ----------
        portfolio_specification : PortfolioSpec
            Portfolio definition to optimize.
        study_value_calculator : callable
            Study value function used for ranking.
        optimization_method : str, default="greedy"
            Portfolio optimization algorithm.
        **kwargs : object
            Additional algorithm-specific options.

        Returns
        -------
        dict[str, object]
            Portfolio optimization result.
        """
        from voiage.methods.portfolio import portfolio_voi

        return portfolio_voi(
            portfolio_specification=portfolio_specification,
            study_value_calculator=study_value_calculator,
            optimization_method=optimization_method,
            **kwargs,
        )

    def get_decision_recommendations(self) -> list[dict[str, Any]]:
        """Summarize the strategies with the highest expected net benefit.

        Returns
        -------
        list[dict[str, Any]]
            Ranked strategy recommendations with mean net benefit and rank.
        """
        nb_values = self.nb_array.numpy_values
        check_input_array(nb_values, expected_ndim=2, name="nb_array", allow_empty=True)

        if nb_values.size == 0:
            return []

        mean_net_benefit = np.mean(nb_values, axis=0)
        strategy_names = self.nb_array.strategy_names
        best_index = int(np.argmax(mean_net_benefit))

        return [
            {
                "strategy": strategy_name,
                "mean_net_benefit": float(mean_value),
                "recommended": index == best_index,
                "rank": int(np.argsort(-mean_net_benefit).tolist().index(index) + 1),
            }
            for index, (strategy_name, mean_value) in enumerate(
                zip(strategy_names, mean_net_benefit, strict=True)
            )
        ]

    def _incremental_max_expected_nb(
        self, nb_values: np.ndarray, chunk_size: int
    ) -> float:
        """Calculate ``max(E[NB_d])`` incrementally in chunks.

        Parameters
        ----------
        nb_values : numpy.ndarray
            Net-benefit array.
        chunk_size : int
            Chunk size used during accumulation.

        Returns
        -------
        float
            Maximum expected net benefit across strategies.
        """
        n_samples, n_strategies = nb_values.shape

        # Initialize accumulators for each strategy
        strategy_sums = np.zeros(n_strategies, dtype=DEFAULT_DTYPE)

        # Process data in chunks
        n_processed = 0
        for start_idx in range(0, n_samples, chunk_size):
            end_idx = min(start_idx + chunk_size, n_samples)
            chunk = nb_values[start_idx:end_idx]
            chunk_size_actual = chunk.shape[0]

            # Calculate sum of net benefits for each strategy in chunk
            chunk_strategy_sums = np.sum(chunk, axis=0)
            strategy_sums += chunk_strategy_sums

            n_processed += chunk_size_actual

        # Calculate expected net benefit for each strategy
        expected_nb_options = strategy_sums / n_processed
        max_expected_nb = np.max(expected_nb_options)

        return float(max_expected_nb)

    def _get_parameter_samples_as_ndarray(self) -> np.ndarray:
        """Return parameter samples as a 2D NumPy array for regression.

        Returns
        -------
        numpy.ndarray
            Parameter matrix with shape ``(n_samples, n_parameters)``.
        """
        if self.parameter_samples is None:
            raise_input_error("`parameter_samples` are not available.")

        if isinstance(self.parameter_samples.parameters, dict):
            x = np.stack(list(self.parameter_samples.parameters.values()), axis=1)
        else:
            # Handle xarray or other types if necessary
            raise_input_error(
                "PSASample with non-dict parameters not yet supported for EVPPI."
            )

        if x.ndim == 1:  # pragma: no cover - stacked parameter samples are 2D
            x = x.reshape(-1, 1)

        if x.shape[0] != self.nb_array.n_samples:
            raise_dimension_mismatch_error(
                f"Number of samples in `parameter_samples` ({x.shape[0]}) "
                f"does not match `nb_array` ({self.nb_array.n_samples})."
            )
        return x
