# voiage/analysis.py

"""A module for the core decision analysis interface."""

from collections import deque
from typing import Any, Dict, Generator, Optional, Union

import numpy as np

from voiage.backends import get_backend
from voiage.config import DEFAULT_DTYPE
from voiage.core.utils import check_input_array
from voiage.exceptions import (
    CalculationError,
    DimensionMismatchError,
    InputError,
    OptionalDependencyError,
)
from voiage.schema import ParameterSet, ValueArray

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


class DecisionAnalysis:
    """A class to represent a decision analysis problem."""

    def __init__(
        self,
        nb_array: Union[np.ndarray, ValueArray],
        parameter_samples: Optional[
            Union[np.ndarray, ParameterSet, Dict[str, np.ndarray]]
        ] = None,
        backend: Optional[str] = None,
        use_jit: bool = False,
        streaming_window_size: Optional[int] = None,
        enable_caching: bool = False,
    ):
        if isinstance(nb_array, ValueArray):
            self.nb_array = nb_array
        elif isinstance(nb_array, np.ndarray):
            # Assuming strategy names are not provided
            self.nb_array = ValueArray.from_numpy(nb_array)
        else:
            raise InputError("`nb_array` must be a NumPy array or ValueArray object.")

        if parameter_samples is not None:
            if isinstance(parameter_samples, ParameterSet):
                self.parameter_samples: Optional[ParameterSet] = parameter_samples
            elif isinstance(parameter_samples, (dict, np.ndarray)):
                self.parameter_samples = ParameterSet.from_numpy_or_dict(
                    parameter_samples
                )
            else:
                raise InputError(
                    f"`parameter_samples` must be a NumPy array, ParameterSet, or Dict. Got {type(parameter_samples)}."
                )
        else:
            self.parameter_samples = None

        # Set the computational backend
        self.backend = get_backend(backend)
        self.use_jit = use_jit

        # Streaming data support
        self.streaming_window_size = streaming_window_size
        self._streaming_data_buffer = None
        self._streaming_parameter_buffer = None
        if streaming_window_size is not None:
            self._initialize_streaming_buffers()

        # Caching support
        self.enable_caching = enable_caching
        self._cache = {} if enable_caching else None

        # Track data changes to invalidate cache
        self._data_hash = self._compute_data_hash()

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
        nb_values = self.nb_array.values
        hash_components.append(hash(nb_values.tobytes()))

        # Hash parameter samples if they exist
        if self.parameter_samples is not None:
            if hasattr(self.parameter_samples, 'parameters') and isinstance(self.parameter_samples.parameters, dict):
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

    def _cache_get(self, key: str) -> Optional[Any]:
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

    def _cache_set(self, key: str, value: Any) -> None:
        """
        Set a value in the cache.

        Args:
            key: Cache key
            value: Value to cache
        """
        if self.enable_caching and self._cache is not None:
            self._invalidate_cache_if_needed()
            self._cache[key] = value

    def _initialize_streaming_buffers(self):
        """Initialize buffers for streaming data."""
        if self.streaming_window_size is not None:
            # Initialize buffers as deques with maximum length
            self._streaming_data_buffer = deque(maxlen=self.streaming_window_size)
            if self.parameter_samples is not None:
                self._streaming_parameter_buffer = deque(maxlen=self.streaming_window_size)

    def update_with_new_data(
        self,
        new_nb_data: Union[np.ndarray, ValueArray],
        new_parameter_samples: Optional[Union[np.ndarray, ParameterSet, Dict[str, np.ndarray]]] = None
    ) -> None:
        """
        Update the decision analysis with new data for streaming VOI calculations.

        Args:
            new_nb_data: New net benefit data to add
            new_parameter_samples: New parameter samples corresponding to the net benefit data
        """
        # Convert new data to appropriate format
        if isinstance(new_nb_data, ValueArray):
            new_nb_values = new_nb_data.values
        elif isinstance(new_nb_data, np.ndarray):
            new_nb_values = new_nb_data
        else:
            raise InputError("`new_nb_data` must be a NumPy array or ValueArray object.")

        # Validate dimensions
        if new_nb_values.ndim != 2:
            raise DimensionMismatchError("New net benefit data must be 2-dimensional.")

        # If we have streaming buffers, add the new data
        if self._streaming_data_buffer is not None:
            # Add new data to buffer
            for i in range(new_nb_values.shape[0]):
                self._streaming_data_buffer.append(new_nb_values[i:i+1, :])

                # If we have parameter samples, add them too
                if new_parameter_samples is not None and self._streaming_parameter_buffer is not None:
                    # Convert parameter samples to appropriate format
                    if isinstance(new_parameter_samples, ParameterSet):
                        param_values = new_parameter_samples
                    elif isinstance(new_parameter_samples, (dict, np.ndarray)):
                        param_values = ParameterSet.from_numpy_or_dict(new_parameter_samples)
                    else:
                        raise InputError(
                            f"`new_parameter_samples` must be a NumPy array, ParameterSet, or Dict. Got {type(new_parameter_samples)}."
                        )

                    # Add parameter sample to buffer
                    if hasattr(param_values, 'parameters') and isinstance(param_values.parameters, dict):
                        # Extract the i-th sample for each parameter
                        sample_dict = {}
                        for param_name, param_array in param_values.parameters.items():
                            if i < len(param_array):
                                sample_dict[param_name] = param_array[i:i+1]
                        self._streaming_parameter_buffer.append(sample_dict)

            # Update the main data arrays with buffered data
            self._update_main_arrays_from_buffer()
        else:
            # If no streaming buffers, just append to existing data
            self._append_to_existing_data(new_nb_values, new_parameter_samples)

    def _update_main_arrays_from_buffer(self):
        """Update the main data arrays from the streaming buffers."""
        if self._streaming_data_buffer:
            # Convert buffered data to numpy array
            buffered_data = np.vstack(list(self._streaming_data_buffer))
            self.nb_array = ValueArray.from_numpy(buffered_data)

        if self._streaming_parameter_buffer and self.parameter_samples:
            # Convert buffered parameters to ParameterSet
            if self._streaming_parameter_buffer:
                # Combine all buffered parameter samples
                combined_params = {}
                for buffered_sample in self._streaming_parameter_buffer:
                    for param_name, param_value in buffered_sample.items():
                        if param_name not in combined_params:
                            combined_params[param_name] = []
                        combined_params[param_name].extend(param_value)

                # Convert to numpy arrays
                for param_name, param_values in combined_params.items():
                    combined_params[param_name] = np.array(param_values)

                self.parameter_samples = ParameterSet.from_numpy_or_dict(combined_params)

    def _append_to_existing_data(
        self,
        new_nb_values: np.ndarray,
        new_parameter_samples: Optional[Union[np.ndarray, ParameterSet, Dict[str, np.ndarray]]] = None
    ) -> None:
        """Append new data to existing data arrays."""
        # Append to net benefit data
        current_nb_values = self.nb_array.values
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
                raise InputError(
                    f"`new_parameter_samples` must be a NumPy array, ParameterSet, or Dict. Got {type(new_parameter_samples)}."
                )

            # Combine parameter samples
            if hasattr(new_params, 'parameters') and isinstance(new_params.parameters, dict):
                combined_params = {}
                for param_name in self.parameter_samples.parameters.keys():
                    if param_name in new_params.parameters:
                        combined_params[param_name] = np.concatenate([
                            self.parameter_samples.parameters[param_name],
                            new_params.parameters[param_name]
                        ])
                    else:
                        combined_params[param_name] = self.parameter_samples.parameters[param_name]
                self.parameter_samples = ParameterSet.from_numpy_or_dict(combined_params)

    def streaming_evpi(self) -> Generator[float, None, None]:
        """
        Calculate EVPI continuously as new data arrives.

        Yields
        ------
            float: EVPI value calculated with current data
        """
        while True:
            # Calculate EVPI with current data
            evpi_value = self.evpi()
            yield evpi_value

    def streaming_evppi(self) -> Generator[float, None, None]:
        """
        Calculate EVPPI continuously as new data arrives.

        Yields
        ------
            float: EVPPI value calculated with current data
        """
        while True:
            # Calculate EVPPI with current data
            evppi_value = self.evppi()
            yield evppi_value

    def evpi(
        self,
        population: Optional[float] = None,
        time_horizon: Optional[float] = None,
        discount_rate: Optional[float] = None,
        chunk_size: Optional[int] = None,
    ) -> float:
        """Calculate the Expected Value of Perfect Information (EVPI).

        EVPI = E[max(NB)] - max(E[NB])
        where E is the expectation over the PSA samples.

        Args:
            population (Optional[float]): The relevant population size. If provided
                along with `time_horizon`, EVPI will be scaled to population level.
            time_horizon (Optional[float]): The relevant time horizon in years.
                If provided along with `population`, EVPI will be scaled.
            discount_rate (Optional[float]): The annual discount rate (e.g., 0.03 for 3%).
                Used for population scaling. Defaults to 0 if `population` and
                `time_horizon` are provided but `discount_rate` is not.
            chunk_size (Optional[int]): Size of chunks for incremental computation.
                If provided, data will be processed in chunks to reduce memory usage.
                Useful for large datasets.

        Returns
        -------
            float: The calculated EVPI. If population parameters are provided,
                   returns population-adjusted EVPI, otherwise per-decision EVPI.

        Raises
        ------
            InputError: If inputs are invalid (e.g., wrong types, shapes, values).
            DimensionMismatchError: If `nb_array` does not have 2 dimensions.
            CalculationError: For issues during calculation.
        """
        # Check cache first
        cache_key = f"evpi_{population}_{time_horizon}_{discount_rate}_{chunk_size}"
        cached_result = self._cache_get(cache_key)
        if cached_result is not None:
            return cached_result

        nb_values = self.nb_array.values
        check_input_array(nb_values, expected_ndim=2, name="nb_array", allow_empty=True)

        if nb_values.size == 0:
            return 0.0
        if nb_values.shape[1] == 1:  # Single strategy
            return 0.0

        try:
            # Use incremental computation if chunk_size is specified
            if chunk_size is not None:
                per_decision_evpi = self._incremental_evpi(nb_values, chunk_size)
            else:
                # Use the selected backend for computation
                if self.use_jit and hasattr(self.backend, 'evpi_jit'):
                    # Use JIT compilation if available and requested
                    per_decision_evpi = self.backend.evpi_jit(nb_values)
                else:
                    # Use regular computation
                    per_decision_evpi = self.backend.evpi(nb_values)

            # EVPI should theoretically be non-negative. Small negative values can occur due to float precision.
            per_decision_evpi = max(0.0, float(per_decision_evpi))

        except Exception as e:
            raise CalculationError(f"Error during EVPI calculation: {e}") from e

        if population is not None and time_horizon is not None:
            # Validate population parameter
            if not isinstance(population, (int, float)):
                raise InputError(f"Population must be a number. Got {type(population)}.")
            if population <= 0:
                raise InputError(f"Population must be positive. Got {population}.")
            if not np.isfinite(population):
                raise InputError(f"Population must be finite. Got {population}.")

            # Validate time_horizon parameter
            if not isinstance(time_horizon, (int, float)):
                raise InputError(f"Time horizon must be a number. Got {type(time_horizon)}.")
            if time_horizon <= 0:
                raise InputError(f"Time horizon must be positive. Got {time_horizon}.")
            if not np.isfinite(time_horizon):
                raise InputError(f"Time horizon must be finite. Got {time_horizon}.")

            # Validate discount_rate parameter
            current_dr = discount_rate
            if current_dr is None:
                current_dr = 0.0  # Default to no discounting if not provided

            if not isinstance(current_dr, (int, float)):
                raise InputError(f"Discount rate must be a number. Got {type(current_dr)}.")
            if not (0 <= current_dr <= 1):
                raise InputError(f"Discount rate must be between 0 and 1. Got {current_dr}.")
            if not np.isfinite(current_dr):
                raise InputError(f"Discount rate must be finite. Got {current_dr}.")

            # Calculate annuity factor
            if current_dr == 0:
                annuity_factor = time_horizon
            else:
                annuity_factor = (1 - (1 + current_dr) ** (-time_horizon)) / current_dr

            result = float(per_decision_evpi * population * annuity_factor)

            # Validate result
            if not np.isfinite(result):
                raise CalculationError(f"Calculated EVPI is not finite: {result}")

            # Cache the result
            self._cache_set(cache_key, result)
            return result
        elif (
            population is not None
            or time_horizon is not None
            or discount_rate is not None
        ):
            raise InputError(
                "To calculate population EVPI, 'population' and 'time_horizon' must be provided. "
                "'discount_rate' is optional (defaults to 0 if not provided)."
            )

        # Cache the result
        self._cache_set(cache_key, float(per_decision_evpi))
        return float(per_decision_evpi)

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

        evpi = expected_max_nb - max_expected_nb
        return evpi

    def evppi(
        self,
        parameters_of_interest: list[str],
        population: Optional[float] = None,
        time_horizon: Optional[float] = None,
        discount_rate: Optional[float] = None,
        n_regression_samples: Optional[int] = None,
        regression_model: Optional[Any] = None,
        chunk_size: Optional[int] = None,
    ) -> float:
        """Calculate the Expected Value of Partial Perfect Information (EVPPI).

        EVPPI quantifies the value of learning the true value of a specific
        subset of model parameters. It is typically calculated using a regression-based
        approach (e.g., Strong & Oakley).

        EVPPI = E_p [max_d E[NB_d|p]] - max_d E[NB_d]
        where E_p is the expectation over the parameter(s) of interest,
        and E[NB_d|p] is the expected net benefit of strategy d conditional
        on the parameter(s) p, usually estimated via regression.

        Args:
            parameters_of_interest (list[str]): List of parameter names to analyze.
            population (Optional[float]): Population size for scaling.
            time_horizon (Optional[float]): Time horizon for scaling.
            discount_rate (Optional[float]): Discount rate for scaling.
            n_regression_samples (Optional[int]): Number of samples to use for fitting
                the regression model. If None, all samples are used. Useful for
                large datasets to speed up computation, at the cost of precision.
            regression_model (Optional[Any]): An unfitted scikit-learn compatible
                regression model. If None, defaults to `sklearn.linear_model.LinearRegression`.
            chunk_size (Optional[int]): Size of chunks for incremental computation
                of the second term (max_d E[NB_d]). If provided, data will be
                processed in chunks to reduce memory usage. Useful for large datasets.

        Returns
        -------
            float: The calculated EVPPI. Scaled if population args are provided.

        Raises
        ------
            InputError: For invalid inputs.
            DimensionMismatchError: If array dimensions are inconsistent.
            OptionalDependencyError: If scikit-learn is not installed.
            CalculationError: For issues during calculation.
        """
        # Check cache first
        cache_key = f"evppi_{population}_{time_horizon}_{discount_rate}_{n_regression_samples}_{chunk_size}_{regression_model!s}"
        cached_result = self._cache_get(cache_key)
        if cached_result is not None:
            return cached_result

        if not SKLEARN_AVAILABLE:
            raise OptionalDependencyError(
                "scikit-learn is required for EVPPI calculation. "
                "Please install it (e.g., `pip install scikit-learn`)."
            )

        if self.parameter_samples is None:
            raise InputError(
                "`parameter_samples` must be provided for EVPPI calculation."
            )

        if not isinstance(parameters_of_interest, list):
            raise InputError(
                "`parameters_of_interest` must be a list of parameter names."
            )

        # Validate that all parameters of interest exist in the parameter set
        param_names = self.parameter_samples.parameter_names
        for param in parameters_of_interest:
            if param not in param_names:
                raise InputError(
                    "All `parameters_of_interest` must be in the ParameterSet"
                )

        nb_values = self.nb_array.values
        check_input_array(nb_values, expected_ndim=2, name="nb_array")
        n_samples, n_strategies = nb_values.shape

        if n_strategies == 0:
            return 0.0
        if n_strategies == 1:  # Single strategy
            return 0.0
        if n_samples == 0:
            raise InputError("`nb_array` cannot be empty (no samples).")

        x_all = self._get_parameter_samples_as_ndarray()

        # Select only the columns corresponding to parameters of interest
        x_indices = [
            i for i, name in enumerate(param_names) if name in parameters_of_interest
        ]
        x = x_all[:, x_indices]

        if n_regression_samples is not None:
            if not isinstance(n_regression_samples, int):
                raise InputError(
                    f"n_regression_samples must be an integer. Got {type(n_regression_samples)}."
                )
            if n_regression_samples <= 0:
                raise InputError(
                    f"n_regression_samples must be positive. Got {n_regression_samples}."
                )
            if not np.isfinite(n_regression_samples):
                raise InputError(
                    f"n_regression_samples must be finite. Got {n_regression_samples}."
                )
            if n_regression_samples > n_samples:
                raise InputError(
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
                model = SklearnLinearRegression()
            else:
                # Ensure it's a new instance if a class is passed, or use instance if provided
                try:
                    model = regression_model()  # if it's a class
                except TypeError:
                    model = regression_model  # if it's an instance

            try:
                model.fit(x_fit, y_fit)
                # Predict on the full set of parameter samples X
                fitted_nb_on_params[:, i] = model.predict(x)
            except Exception as e:
                raise CalculationError(
                    f"Error during regression for strategy {i}: {e}"
                ) from e

        # Calculate E_p [max_d E[NB_d|p]]
        # This is the mean of the maximum (over strategies) of the fitted net benefits
        e_max_enb_conditional = np.mean(np.max(fitted_nb_on_params, axis=1))

        # Calculate max_d E[NB_d] using incremental computation if chunk_size is specified
        if chunk_size is not None:
            max_e_nb = self._incremental_max_expected_nb(nb_values, chunk_size)
        else:
            # Standard calculation
            max_e_nb: float = np.max(np.mean(nb_values, axis=0))

        per_decision_evppi = float(e_max_enb_conditional - max_e_nb)

        # EVPPI should theoretically be non-negative. Small negative values can occur due to float precision or regression error.
        per_decision_evppi = max(0.0, per_decision_evppi)

        if population is not None and time_horizon is not None:
            # Validate population parameter
            if not isinstance(population, (int, float)):
                raise InputError(f"Population must be a number. Got {type(population)}.")
            if population <= 0:
                raise InputError(f"Population must be positive. Got {population}.")
            if not np.isfinite(population):
                raise InputError(f"Population must be finite. Got {population}.")

            # Validate time_horizon parameter
            if not isinstance(time_horizon, (int, float)):
                raise InputError(f"Time horizon must be a number. Got {type(time_horizon)}.")
            if time_horizon <= 0:
                raise InputError(f"Time horizon must be positive. Got {time_horizon}.")
            if not np.isfinite(time_horizon):
                raise InputError(f"Time horizon must be finite. Got {time_horizon}.")

            # Validate discount_rate parameter
            current_dr = discount_rate
            if current_dr is None:
                current_dr = 0.0

            if not isinstance(current_dr, (int, float)):
                raise InputError(f"Discount rate must be a number. Got {type(current_dr)}.")
            if not (0 <= current_dr <= 1):
                raise InputError(f"Discount rate must be between 0 and 1. Got {current_dr}.")
            if not np.isfinite(current_dr):
                raise InputError(f"Discount rate must be finite. Got {current_dr}.")

            # Calculate annuity factor
            if current_dr == 0:
                annuity_factor = time_horizon
            else:
                annuity_factor = (1 - (1 + current_dr) ** (-time_horizon)) / current_dr

            result = float(per_decision_evppi * population * annuity_factor)

            # Validate result
            if not np.isfinite(result):
                raise CalculationError(f"Calculated EVPPI is not finite: {result}")

            # Cache the result
            self._cache_set(cache_key, result)
            return result
        elif (
            population is not None
            or time_horizon is not None
            or discount_rate is not None
        ):
            raise InputError(
                "To calculate population EVPPI, 'population' and 'time_horizon' must be provided. "
                "'discount_rate' is optional (defaults to 0 if not provided)."
            )

        # Cache the result
        self._cache_set(cache_key, float(per_decision_evppi))
        return float(per_decision_evppi)

    def _incremental_max_expected_nb(self, nb_values: np.ndarray, chunk_size: int) -> float:
        """
        Calculate max(E[NB_d]) incrementally in chunks to handle large datasets.

        Args:
            nb_values: Net benefit array
            chunk_size: Size of chunks for processing

        Returns
        -------
            float: Maximum expected net benefit across strategies
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
        """Get parameter samples as a numpy array for regression."""
        if self.parameter_samples is None:
            raise InputError("`parameter_samples` are not available.")

        if isinstance(self.parameter_samples.parameters, dict):
            x = np.stack(list(self.parameter_samples.parameters.values()), axis=1)
        else:
            # Handle xarray or other types if necessary
            raise InputError(
                "PSASample with non-dict parameters not yet supported for EVPPI."
            )

        if x.ndim == 1:
            x = x.reshape(-1, 1)

        if x.shape[0] != self.nb_array.n_samples:
            raise DimensionMismatchError(
                f"Number of samples in `parameter_samples` ({x.shape[0]}) "
                f"does not match `nb_array` ({self.nb_array.n_samples})."
            )
        return x
