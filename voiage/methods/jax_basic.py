# voiage/methods/jax_basic.py

"""
JAX implementation of basic Value of Information methods.
"""

from functools import partial
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jnp

from voiage.config import DEFAULT_DTYPE
from voiage.exceptions import (
    CalculationError,
    DimensionMismatchError,
    InputError,
    OptionalDependencyError,
)
from voiage.schema import ValueArray, ParameterSet


@partial(jax.jit, static_argnums=(1, 2, 3))
def evpi(
    nb_array: jnp.ndarray,
    population: Optional[float] = None,
    time_horizon: Optional[float] = None,
    discount_rate: Optional[float] = None,
) -> float:
    """
    Calculate the Expected Value of Perfect Information (EVPI) using JAX.
    """
    if nb_array.ndim != 2:
        raise DimensionMismatchError("nb_array must be a 2D array")

    e_max_nb = jnp.mean(jnp.max(nb_array, axis=1))
    max_e_nb = jnp.max(jnp.mean(nb_array, axis=0))

    per_decision_evpi = e_max_nb - max_e_nb
    per_decision_evpi = jnp.maximum(0.0, per_decision_evpi)

    if population is not None and time_horizon is not None:
        if discount_rate is None:
            discount_rate = 0.0
        annuity_factor = (1 - (1 + discount_rate) ** -time_horizon) / discount_rate
        return per_decision_evpi * population * annuity_factor
    else:
        return per_decision_evpi


@partial(jax.jit, static_argnums=(2, 3, 4, 5, 6))
def evppi(
    nb_array: jnp.ndarray,
    parameter_samples: jnp.ndarray,
    population: Optional[float] = None,
    time_horizon: Optional[float] = None,
    discount_rate: Optional[float] = None,
    n_regression_samples: Optional[int] = None,
    regression_model: Optional[Any] = None,
) -> float:
    """
    Calculate the Expected Value of Partial Perfect Information (EVPPI) using JAX.
    """
    if nb_array.ndim != 2:
        raise DimensionMismatchError("nb_array must be a 2D array")
    if parameter_samples.ndim != 2:
        raise DimensionMismatchError("parameter_samples must be a 2D array")

    n_samples = nb_array.shape[0]
    if n_regression_samples is None:
        n_regression_samples = n_samples

    # In JAX, we can't use sklearn directly. We need to implement a simple
    # linear regression using JAX.
    if regression_model is None:
        # Simple linear regression using least squares
        X = jnp.concatenate([jnp.ones((n_regression_samples, 1)), parameter_samples[:n_regression_samples]], axis=1)
        y = nb_array[:n_regression_samples]
        beta = jnp.linalg.lstsq(X, y, rcond=None)[0]
        fitted_nb_on_params = jnp.concatenate([jnp.ones((n_samples, 1)), parameter_samples], axis=1) @ beta
    else:
        # For now, we don't support custom regression models in JAX
        raise VoiageNotImplementedError("Custom regression models are not yet supported for the JAX backend.")

    e_max_enb_conditional = jnp.mean(jnp.max(fitted_nb_on_params, axis=1))
    max_e_nb = jnp.max(jnp.mean(nb_array, axis=0))

    per_decision_evppi = e_max_enb_conditional - max_e_nb
    per_decision_evppi = jnp.maximum(0.0, per_decision_evppi)

    if population is not None and time_horizon is not None:
        if discount_rate is None:
            discount_rate = 0.0
        annuity_factor = (1 - (1 + discount_rate) ** -time_horizon) / discount_rate
        return per_decision_evppi * population * annuity_factor
    else:
        return per_decision_evppi
