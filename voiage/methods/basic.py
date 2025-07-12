# voiage/methods/basic.py

"""Implementation of basic Value of Information methods.

- EVPI (Expected Value of Perfect Information)
- EVPPI (Expected Value of Partial Perfect Information)
"""

from typing import Any, Dict, Optional, Union

import numpy as np

from voiage.config import DEFAULT_DTYPE
from voiage.core.data_structures import NetBenefitArray, PSASample
from voiage.core.utils import check_input_array
from voiage.exceptions import (
    CalculationError,
    DimensionMismatchError,
    InputError,
    OptionalDependencyError,
)

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


def check_parameter_samples(parameter_samples, n_samples):
    if isinstance(parameter_samples, np.ndarray):
        x = parameter_samples
    elif isinstance(parameter_samples, PSASample):
        if isinstance(parameter_samples.parameters, dict):
            x = np.stack(list(parameter_samples.parameters.values()), axis=1)
        else:
            # Handle xarray or other types if necessary
            raise InputError("PSASample with non-dict parameters not yet supported for EVPPI.")
    elif isinstance(parameter_samples, dict):
        x = np.stack(list(parameter_samples.values()), axis=1)
    else:
        raise InputError(
            f"`parameter_samples` must be a NumPy array, PSASample, or Dict. Got {type(parameter_samples)}."
        )

    if x.ndim == 1:
        x = x.reshape(-1, 1)

    if x.shape[0] != n_samples:
        raise DimensionMismatchError(
            f"Number of samples in `parameter_samples` ({x.shape[0]}) "
            f"does not match `nb_array` ({n_samples})."
        )
    return x


def evpi(
    nb_array: Union[np.ndarray, NetBenefitArray],
    population: Optional[float] = None,
    time_horizon: Optional[float] = None,
    discount_rate: Optional[float] = None,
    # wtp: Optional[float] = None, # WTP is implicit in NetBenefitArray
) -> float:
    """Calculate the Expected Value of Perfect Information (EVPI).

    EVPI = E[max(NB)] - max(E[NB])
    where E is the expectation over the PSA samples.

    Args:
        nb_array (Union[np.ndarray, NetBenefitArray]): A 2D NumPy array or
            NetBenefitArray of shape (n_samples, n_strategies), representing
            the net benefit for each PSA sample and each strategy.
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
    if isinstance(nb_array, NetBenefitArray):
        nb_values = nb_array.values
    elif isinstance(nb_array, np.ndarray):
        nb_values = nb_array
    else:
        raise InputError("`nb_array` must be a NumPy array or NetBenefitArray object.")

    check_input_array(nb_values, expected_ndim=2, name="nb_array", allow_empty=True)

    if nb_values.size == 0:
        return 0.0
    if nb_values.shape[1] == 1:  # Single strategy
        return 0.0

    try:
        # E[max(NB_d)] = Mean over samples of (max NB across strategies for that sample)
        e_max_nb = np.mean(np.max(nb_values, axis=1))

        # max(E[NB_d]) = Max over strategies of (mean NB for that strategy)
        max_e_nb = np.max(np.mean(nb_values, axis=0))

        per_decision_evpi = e_max_nb - max_e_nb

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

        return per_decision_evpi * population * annuity_factor
    elif (
        population is not None or time_horizon is not None or discount_rate is not None
    ):
        raise InputError(
            "To calculate population EVPI, 'population' and 'time_horizon' must be provided. "
            "'discount_rate' is optional (defaults to 0 if not provided)."
        )

    return per_decision_evpi


def evppi(
    nb_array: Union[np.ndarray, NetBenefitArray],
    parameter_samples: Union[np.ndarray, PSASample, Dict[str, np.ndarray]],
    population: Optional[float] = None,
    time_horizon: Optional[float] = None,
    discount_rate: Optional[float] = None,
    # wtp: Optional[float] = None, # WTP is implicit in NetBenefitArray
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
        nb_array (Union[np.ndarray, NetBenefitArray]): A 2D NumPy array or
            NetBenefitArray of shape (n_samples, n_strategies), representing
            the net benefit for each PSA sample and each strategy.
        parameter_samples (Union[np.ndarray, PSASample, Dict[str, np.ndarray]]):
            Samples of the parameter(s) of interest.
            - If np.ndarray: Shape (n_samples,) or (n_samples, n_params).
            - If PSASample: Will extract parameters. If multiple, they'll be stacked.
            - If Dict: Values are np.ndarray of shape (n_samples,). Keys are param names.
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

    if isinstance(nb_array, NetBenefitArray):
        nb_values = nb_array.values
    elif isinstance(nb_array, np.ndarray):
        nb_values = nb_array
    else:
        raise InputError("`nb_array` must be a NumPy array or NetBenefitArray object.")

    check_input_array(nb_values, expected_ndim=2, name="nb_array")
    n_samples, n_strategies = nb_values.shape

    if n_strategies == 0:
        return 0.0
    if n_strategies == 1:  # Single strategy
        return 0.0
    if n_samples == 0:
        raise InputError("`nb_array` cannot be empty (no samples).")

    x = check_parameter_samples(parameter_samples, n_samples)

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
            model = LinearRegression()
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
    max_e_nb = np.max(np.mean(nb_values, axis=0))

    per_decision_evppi = e_max_enb_conditional - max_e_nb

    # EVPPI should theoretically be non-negative. Small negative values can occur due to float precision or regression error.
    per_decision_evppi = max(0.0, per_decision_evppi)

    # Further, EVPPI should not exceed EVPI.
    # This can be used as a sanity check, though regression error might violate it slightly.
    # evpi_val = evpi(nb_values) # Calculate per-decision EVPI for comparison
    # per_decision_evppi = min(per_decision_evppi, evpi_val) # Cap EVPPI at EVPI

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
        return per_decision_evppi * population * annuity_factor
    elif (
        population is not None or time_horizon is not None or discount_rate is not None
    ):
        raise InputError(
            "To calculate population EVPPI, 'population' and 'time_horizon' must be provided. "
            "'discount_rate' is optional (defaults to 0 if not provided)."
        )

    return per_decision_evppi
