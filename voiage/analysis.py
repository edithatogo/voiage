# voiage/analysis.py

"""A module for the core decision analysis interface."""

from typing import Any, Dict, Optional, Union

import numpy as np

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

    def evpi(
        self,
        population: Optional[float] = None,
        time_horizon: Optional[float] = None,
        discount_rate: Optional[float] = None,
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
        nb_values = self.nb_array.values
        check_input_array(nb_values, expected_ndim=2, name="nb_array", allow_empty=True)

        if nb_values.size == 0:
            return 0.0
        if nb_values.shape[1] == 1:  # Single strategy
            return 0.0

        try:
            # E[max(NB_d)] = Mean over samples of (max NB across strategies for that sample)
            e_max_nb = np.mean(np.max(nb_values, axis=1))

            # max(E[NB_d]) = Max over strategies of (mean NB for that strategy)
            max_e_nb: float = np.max(np.mean(nb_values, axis=0))

            per_decision_evpi = float(e_max_nb - max_e_nb)

            # EVPI should theoretically be non-negative. Small negative values can occur due to float precision.
            per_decision_evpi = max(0.0, per_decision_evpi)

        except Exception as e:
            raise CalculationError(f"Error during EVPI calculation: {e}") from e

        if population is not None and time_horizon is not None:
            if not isinstance(population, (int, float)) or population <= 0:
                raise InputError("Population must be a positive number.")
            if not isinstance(time_horizon, (int, float)) or time_horizon <= 0:
                raise InputError("Time horizon must be a positive number.")

            current_dr = discount_rate
            if current_dr is None:
                current_dr = 0.0  # Default to no discounting if not provided

            if not isinstance(current_dr, (int, float)) or not (0 <= current_dr <= 1):
                raise InputError("Discount rate must be a number between 0 and 1.")

            if current_dr == 0:
                annuity_factor = time_horizon
            else:
                annuity_factor = (1 - (1 + current_dr) ** (-time_horizon)) / current_dr

            return float(per_decision_evpi * population * annuity_factor)
        elif (
            population is not None
            or time_horizon is not None
            or discount_rate is not None
        ):
            raise InputError(
                "To calculate population EVPI, 'population' and 'time_horizon' must be provided. "
                "'discount_rate' is optional (defaults to 0 if not provided)."
            )

        return float(per_decision_evpi)

    def evppi(
        self,
        parameters_of_interest: list[str],
        population: Optional[float] = None,
        time_horizon: Optional[float] = None,
        discount_rate: Optional[float] = None,
        n_regression_samples: Optional[int] = None,
        regression_model: Optional[Any] = None,
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
            if not isinstance(n_regression_samples, int) or n_regression_samples <= 0:
                raise InputError(
                    "n_regression_samples, if provided, must be a positive integer."
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

        # Calculate max_d E[NB_d] (same as in EVPI)
        max_e_nb: float = np.max(np.mean(nb_values, axis=0))

        per_decision_evppi = float(e_max_enb_conditional - max_e_nb)

        # EVPPI should theoretically be non-negative. Small negative values can occur due to float precision or regression error.
        per_decision_evppi = max(0.0, per_decision_evppi)

        if population is not None and time_horizon is not None:
            if not isinstance(population, (int, float)) or population <= 0:
                raise InputError("Population must be a positive number.")
            if not isinstance(time_horizon, (int, float)) or time_horizon <= 0:
                raise InputError("Time horizon must be a positive number.")

            current_dr = discount_rate
            if current_dr is None:
                current_dr = 0.0

            if not isinstance(current_dr, (int, float)) or not (0 <= current_dr <= 1):
                raise InputError("Discount rate must be a number between 0 and 1.")

            if current_dr == 0:
                annuity_factor = time_horizon
            else:
                annuity_factor = (1 - (1 + current_dr) ** (-time_horizon)) / current_dr
            return float(per_decision_evppi * population * annuity_factor)
        elif (
            population is not None
            or time_horizon is not None
            or discount_rate is not None
        ):
            raise InputError(
                "To calculate population EVPPI, 'population' and 'time_horizon' must be provided. "
                "'discount_rate' is optional (defaults to 0 if not provided)."
            )

        return float(per_decision_evppi)

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
