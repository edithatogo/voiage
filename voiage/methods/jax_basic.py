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
