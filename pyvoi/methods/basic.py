"""
Basic Value of Information (VOI) methods.

This module provides implementations for the Expected Value of Perfect Information (EVPI)
and Expected Value of Partial Perfect Information (EVPPI).
"""

from typing import List, Optional, Union

import numpy as np

from pyvoi.core.data_structures import NetBenefitArray, PSASample
from pyvoi.exceptions import DimensionMismatchError, InputError


def _calculate_population_multiplier(
    population: float,
    time_horizon: float,
    discount_rate: Optional[float] = None,
) -> float:
    """Calculate the population multiplier for scaling VOI."""
    if not isinstance(population, (int, float)) or population <= 0:
        raise InputError("Population must be positive.")
    if not isinstance(time_horizon, (int, float)) or time_horizon <= 0:
        raise InputError("Time horizon must be positive.")
    if discount_rate is not None and not (0 <= discount_rate <= 1):
        raise InputError("Discount rate must be between 0 and 1.")

    effective_population = float(population)
    if discount_rate is not None:
        if discount_rate == 0:
            annuity_factor = float(time_horizon)
        else:
            annuity_factor = (
                1 - (1 + discount_rate) ** (-time_horizon)
            ) / discount_rate
        effective_population *= annuity_factor
    else:
        effective_population *= float(time_horizon)

    return effective_population


def evpi(
    net_benefit_array: Union[np.ndarray, NetBenefitArray],
    population: Optional[float] = None,
    time_horizon: Optional[float] = None,
    discount_rate: Optional[float] = None,
) -> float:
    """Calculate the Expected Value of Perfect Information (EVPI).

    EVPI represents the maximum amount one should be willing to pay for perfect
    information about all uncertain parameters in a decision problem. It is
    calculated as the difference between the expected net benefit with perfect
    information and the expected net benefit with current information.

    Parameters
    ----------
    net_benefit_array : NetBenefitArray
        An object containing a 2D NumPy array of net benefits from a
        Probabilistic Sensitivity Analysis (PSA). The array should have
        shape (n_samples, n_strategies), where rows are PSA samples and
        columns are different treatment strategies.

    Returns
    -------
    float
        The calculated EVPI.

    Raises
    ------
    InputError
        If `net_benefit_array` is not an instance of `NetBenefitArray`.
    DimensionMismatchError
        If the `net_benefit_array.values` does not have the expected 2D shape.

    Examples
    --------
    >>> from pyvoi.core.data_structures import NetBenefitArray
    >>> import numpy as np
    >>> # Example with 3 samples and 2 strategies
    >>> nb_values = np.array([
    ...     [10, 12],  # Sample 1: Strategy B is better
    ...     [15, 11],  # Sample 2: Strategy A is better
    ...     [8, 10]    # Sample 3: Strategy B is better
    ... ])
    >>> nba = NetBenefitArray(values=nb_values)
    >>> evpi_value = evpi(nba)
    >>> print(f"EVPI: {evpi_value:.2f}")
    EVPI: 1.00
    """
    if isinstance(net_benefit_array, np.ndarray):
        if net_benefit_array.ndim != 2:
            raise DimensionMismatchError(
                "Input NumPy array for net_benefit_array must be 2D (samples x strategies)."
            )
        if net_benefit_array.size == 0:
            raise InputError("Input NumPy array for net_benefit_array cannot be empty.")
        nb_array_obj = NetBenefitArray(values=net_benefit_array)
    elif isinstance(net_benefit_array, NetBenefitArray):
        nb_array_obj = net_benefit_array
    else:
        raise InputError("Input must be a NumPy array or NetBenefitArray.")

    # Expected net benefit with perfect information (ENB_PI)
    # For each sample, choose the maximum net benefit across strategies
    max_net_benefits_per_sample = np.max(nb_array_obj.values, axis=1)
    expected_net_benefit_perfect_info = np.mean(max_net_benefits_per_sample)

    # Expected net benefit with current information (ENB_CI)
    # Choose the strategy with the highest expected net benefit over all samples
    expected_net_benefits_per_strategy = np.mean(nb_array_obj.values, axis=0)
    expected_net_benefit_current_info: float = np.max(
        expected_net_benefits_per_strategy
    )

    per_decision_evpi = (
        expected_net_benefit_perfect_info - expected_net_benefit_current_info
    )

    # Apply population scaling if parameters are provided
    if population is not None or time_horizon is not None or discount_rate is not None:
        if population is None or time_horizon is None:
            raise InputError(
                "To calculate population EVPI, 'population' and 'time_horizon' must be provided. "
                "'discount_rate' is optional."
            )
        multiplier = _calculate_population_multiplier(
            population, time_horizon, discount_rate
        )
        return float(per_decision_evpi * multiplier)

    return float(per_decision_evpi)


def evppi(
    net_benefit_array: NetBenefitArray,
    psa_sample: PSASample,
    parameters_of_interest: Union[str, List[str]],
    population: Optional[float] = None,
    time_horizon: Optional[float] = None,
    discount_rate: Optional[float] = None,
) -> float:
    """
    Calculate the Expected Value of Partial Perfect Information (EVPPI).

    EVPPI quantifies the value of obtaining perfect information about a subset
    of uncertain parameters. It is more complex to calculate than EVPI, often
    requiring nested Monte Carlo simulation or non-parametric regression methods.

    This initial implementation will be a placeholder, as a full implementation
    requires more advanced techniques (e.g., Gaussian Process regression,
    Generalized Additive Models, or a nested Monte Carlo approach).

    Parameters
    ----------
    net_benefit_array : NetBenefitArray
        An object containing a 2D NumPy array of net benefits from a PSA.
        Shape: (n_samples, n_strategies).
    psa_sample : PSASample
        An object containing the PSA parameter samples.
    parameters_of_interest : Union[str, List[str]]
        The name(s) of the parameter(s) for which EVPPI is to be calculated.
        Must correspond to keys in `psa_sample.parameters`.

    Returns
    -------
    float
        The calculated EVPPI. (Placeholder: currently returns 0.0)

    Raises
    ------
    InputError
        If inputs are not of the expected types or parameters are not found.
    NotImplementedError
        As this is a placeholder for a more complex implementation.

    Examples
    --------
    >>> from pyvoi.core.data_structures import NetBenefitArray, PSASample
    >>> import numpy as np
    >>> nb_values = np.array([
    ...     [10, 12], [15, 11], [8, 10], [13, 14], [9, 7]
    ... ])
    >>> nba = NetBenefitArray(values=nb_values)
    >>> params_dict = {
    ...     "param_a": np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
    ...     "param_b": np.array([100, 110, 120, 130, 140]),
    ... }
    >>> psa = PSASample(parameters=params_dict)
    >>> evppi_value = evppi(nba, psa, "param_a")
    >>> print(f"EVPPI for param_a: {evppi_value:.2f}")
    EVPPI for param_a: 0.00
    """
    if not isinstance(net_benefit_array, (np.ndarray, NetBenefitArray)):
        raise InputError("Input must be a NumPy array or NetBenefitArray.")
    if not isinstance(psa_sample, (dict, PSASample, np.ndarray)):
        raise InputError("parameter_samples must be a NumPy array, PSASample, or Dict")
    if not parameters_of_interest:
        raise InputError("parameters_of_interest cannot be empty.")

    per_decision_evppi = 0.0

    # Apply population scaling if parameters are provided
    if population is not None or time_horizon is not None or discount_rate is not None:
        if population is None or time_horizon is None:
            raise InputError(
                "To calculate population EVPPI, 'population' and 'time_horizon' must be provided. "
                "'discount_rate' is optional."
            )
        multiplier = _calculate_population_multiplier(
            population, time_horizon, discount_rate
        )
        return per_decision_evppi * multiplier

    return per_decision_evppi
