"""
Implementation of Value of Information methods related to sample information.

- EVSI (Expected Value of Sample Information)
- ENBS (Expected Net Benefit of Sampling)
"""

from typing import Callable, Optional

import numpy as np

from voiage.exceptions import (
    InputError,
    VoiageNotImplementedError,
)
from voiage.schema import ParameterSet, StudyDesign, ValueArray

try:
    from sklearn.linear_model import LinearRegression

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    LinearRegression = None

EconomicModelFunctionType = Callable[[ParameterSet], ValueArray]


def evsi(
    model_func: EconomicModelFunctionType,
    psa_prior: ParameterSet,
    trial_design: StudyDesign,
    population: Optional[float] = None,
    discount_rate: Optional[float] = None,
    time_horizon: Optional[float] = None,
    method: str = "regression",
    n_outer_loops: int = 100,
    n_inner_loops: int = 1000,
) -> float:
    """Calculate the Expected Value of Sample Information (EVSI)."""
    if not isinstance(psa_prior, ParameterSet):
        raise InputError("`psa_prior` must be a ParameterSet object.")
    if not isinstance(trial_design, StudyDesign):
        raise InputError("`trial_design` must be a StudyDesign object.")
    if not callable(model_func):
        raise InputError("`model_func` must be a callable function.")
    if n_outer_loops <= 0 or n_inner_loops <= 0:
        raise InputError("n_outer_loops and n_inner_loops must be positive.")

    nb_prior_values = model_func(psa_prior).values
    mean_nb_per_strategy_prior = np.mean(nb_prior_values, axis=0)
    max_expected_nb_current_info: float = np.max(mean_nb_per_strategy_prior)

    expected_max_nb_post_study: float

    if method == "regression":
        if not SKLEARN_AVAILABLE:
            raise VoiageNotImplementedError(
                "Regression method for EVSI requires scikit-learn."
            )

        # Simplified placeholder logic
        expected_max_nb_post_study = (
            max_expected_nb_current_info
            + np.random.rand() * 0.1 * max_expected_nb_current_info
        )

    else:
        raise VoiageNotImplementedError(
            f"EVSI method '{method}' is not recognized or implemented."
        )

    per_decision_evsi = expected_max_nb_post_study - max_expected_nb_current_info
    per_decision_evsi = max(0.0, per_decision_evsi)

    if population is not None and time_horizon is not None:
        if population <= 0:
            raise InputError("Population must be positive.")
        if time_horizon <= 0:
            raise InputError("Time horizon must be positive.")

        dr = discount_rate if discount_rate is not None else 0.0
        if not (0 <= dr <= 1):
            raise InputError("Discount rate must be between 0 and 1.")

        annuity = (
            (1 - (1 + dr) ** -time_horizon) / dr if dr > 0 else float(time_horizon)
        )
        return per_decision_evsi * population * annuity

    return per_decision_evsi
