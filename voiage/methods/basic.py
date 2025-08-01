# voiage/methods/basic.py

"""Implementation of basic Value of Information methods.

- EVPI (Expected Value of Perfect Information)
- EVPPI (Expected Value of Partial Perfect Information)
"""

from typing import Any, Dict, Optional, Union

import numpy as np

from voiage.core.data_structures import NetBenefitArray, PSASample
from voiage.exceptions import (
    DimensionMismatchError,
    InputError,
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
    """Check and format parameter samples for EVPPI calculation."""
    if isinstance(parameter_samples, np.ndarray):
        x = parameter_samples
    elif isinstance(parameter_samples, PSASample):
        if isinstance(parameter_samples.parameters, dict):
            x = np.stack(list(parameter_samples.parameters.values()), axis=1)
        else:
            # Handle xarray or other types if necessary
            raise InputError(
                "PSASample with non-dict parameters not yet supported for EVPPI."
            )
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
    nb_array: Union[np.ndarray, "NetBenefitArray"],
    population: Optional[float] = None,
    time_horizon: Optional[float] = None,
    discount_rate: Optional[float] = None,
) -> float:
    """Functional wrapper for the :meth:`DecisionAnalysis.evpi` method.

    This function provides a simple, stateless interface for calculating EVPI.
    It creates a temporary :class:`DecisionAnalysis` object internally.

    Args:
        nb_array (Union[np.ndarray, NetBenefitArray]): A 2D NumPy array or
            NetBenefitArray of shape (n_samples, n_strategies).
        population (Optional[float]): The relevant population size.
        time_horizon (Optional[float]): The relevant time horizon in years.
        discount_rate (Optional[float]): The annual discount rate.

    Returns
    -------
        float: The calculated EVPI.
    """
    from voiage.analysis import DecisionAnalysis  # Local import to avoid circularity

    analysis = DecisionAnalysis(nb_array=nb_array)
    return analysis.evpi(
        population=population,
        time_horizon=time_horizon,
        discount_rate=discount_rate,
    )


def evppi(
    nb_array: Union[np.ndarray, "NetBenefitArray"],
    parameter_samples: Union[np.ndarray, "PSASample", Dict[str, np.ndarray]],
    population: Optional[float] = None,
    time_horizon: Optional[float] = None,
    discount_rate: Optional[float] = None,
    n_regression_samples: Optional[int] = None,
    regression_model: Optional[Any] = None,
) -> float:
    """Functional wrapper for the :meth:`DecisionAnalysis.evppi` method.

    This function provides a simple, stateless interface for calculating EVPPI.
    It creates a temporary :class:`DecisionAnalysis` object internally.

    Args:
        nb_array (Union[np.ndarray, NetBenefitArray]): Net benefit array.
        parameter_samples (Union[np.ndarray, PSASample, Dict[str, np.ndarray]]):
            Samples of the parameter(s) of interest.
        population (Optional[float]): Population size for scaling.
        time_horizon (Optional[float]): Time horizon for scaling.
        discount_rate (Optional[float]): Discount rate for scaling.
        n_regression_samples (Optional[int]): Number of samples for regression.
        regression_model (Optional[Any]): scikit-learn compatible regression model.

    Returns
    -------
        float: The calculated EVPPI.
    """
    from voiage.analysis import DecisionAnalysis  # Local import to avoid circularity

    analysis = DecisionAnalysis(nb_array=nb_array, parameter_samples=parameter_samples)
    return analysis.evppi(
        population=population,
        time_horizon=time_horizon,
        discount_rate=discount_rate,
        n_regression_samples=n_regression_samples,
        regression_model=regression_model,
    )
