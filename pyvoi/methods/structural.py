# pyvoi/methods/structural.py

"""Implementation of VOI methods for structural uncertainty.

- Structural EVPI (Expected Value of Perfect Information for Model Structure)
- Structural EVPPI (Expected Value of Partial Perfect Information for Model Structure)

Structural uncertainty refers to uncertainty about the fundamental form or
components of the model itself (e.g., choice of parametric family for survival,
inclusion/exclusion of certain pathways, alternative model paradigms like
decision tree vs. Markov model vs. DES).

These methods are often more complex to implement generically as they require
ways to define, sample from, and evaluate alternative model structures.
"""

from typing import Callable, List, Optional, Union

import numpy as np

from pyvoi.core.data_structures import NetBenefitArray, PSASample
from pyvoi.exceptions import InputError, PyVoiNotImplementedError

# Type alias for a function that can evaluate a specific model structure
# It would take parameters and return net benefits for that structure.
ModelStructureEvaluator = Callable[[PSASample], NetBenefitArray]


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
            annuity_factor = (1 - (1 + discount_rate) ** -time_horizon) / discount_rate
        effective_population *= annuity_factor
    else:
        effective_population *= float(time_horizon)

    return effective_population


def structural_evpi(
    model_structure_evaluators: List[ModelStructureEvaluator],
    structure_probabilities: Union[np.ndarray, List[float]],
    psa_samples_per_structure: List[PSASample],
    population: Optional[float] = None,
    discount_rate: Optional[float] = None,
    time_horizon: Optional[float] = None,
) -> float:
    """Calculate the Expected Value of Perfect Information for Model Structure (Structural EVPI).

    Structural EVPI quantifies the expected gain from knowing with certainty which
    model structure is the "true" or most appropriate one.

    Conceptual Formula:
    SEVPI = E_S [max_d E_theta|S [NB(d, theta, S)]] - max_d E_S [E_theta|S [NB(d, theta, S)]]
    where:
        S represents a specific model structure.
        E_S is the expectation over the distribution of model structures (structure_probabilities).
        E_theta|S is the expectation over parameters theta, given structure S.

    This implementation assumes:
    - A discrete set of alternative model structures.
    - Probabilities assigned to each structure being correct.
    - Each structure might have its own set of parameters and PSA samples.

    Args:
        model_structure_evaluators (List[ModelStructureEvaluator]):
            A list of functions. Each function takes a PSASample object (parameters
            relevant to that structure) and returns a NetBenefitArray for that model structure.
        structure_probabilities (Union[np.ndarray, List[float]]):
            Probabilities associated with each model structure being the true one.
            Must sum to 1 and have the same length as `model_structure_evaluators`.
        psa_samples_per_structure (List[PSASample]):
            A list of PSASample objects. Each PSASample corresponds to the parameters
            for the respective model structure in `model_structure_evaluators`.
            Length must match `model_structure_evaluators`.
        population (Optional[float]): Population size for scaling.
        discount_rate (Optional[float]): Discount rate for scaling.
        time_horizon (Optional[float]): Time horizon for scaling.

    Returns
    -------
        float: The calculated Structural EVPI.

    Raises
    ------
        InputError: If inputs are inconsistent (e.g., list lengths don't match,
                    probabilities don't sum to 1).
    """
    if len(model_structure_evaluators) != len(structure_probabilities) or len(
        model_structure_evaluators
    ) != len(psa_samples_per_structure):
        raise InputError(
            "Input lists for structures, probabilities, and PSA samples must have the same length."
        )
    if not np.isclose(np.sum(structure_probabilities), 1.0):
        raise InputError("Structure probabilities must sum to 1.")
    if not model_structure_evaluators:
        return 0.0

    n_structures = len(model_structure_evaluators)
    n_strategies = -1

    expected_nb_given_structure_s_decision_d = []
    expected_max_nb_given_structure_s = []

    for i in range(n_structures):
        evaluator = model_structure_evaluators[i]
        psa_for_s = psa_samples_per_structure[i]
        nb_array_for_s = evaluator(psa_for_s)

        if n_strategies == -1:
            n_strategies = nb_array_for_s.n_strategies
        elif n_strategies != nb_array_for_s.n_strategies:
            raise InputError(
                "All model structures must evaluate the same number of decision strategies."
            )

        mean_nb_d_given_s = np.mean(nb_array_for_s.values, axis=0)
        expected_nb_given_structure_s_decision_d.append(mean_nb_d_given_s)

        max_nb_per_sample_s = np.max(nb_array_for_s.values, axis=1)
        expected_max_nb_s = np.mean(max_nb_per_sample_s)
        expected_max_nb_given_structure_s.append(expected_max_nb_s)

    term1_sevpi: float = np.sum(
        np.array(structure_probabilities) * np.array(expected_max_nb_given_structure_s)
    )

    all_expected_nb_d_s = np.array(expected_nb_given_structure_s_decision_d)
    weighted_avg_nb_d = np.sum(
        np.array(structure_probabilities)[:, np.newaxis] * all_expected_nb_d_s,
        axis=0,
    )
    term2_sevpi: float = np.max(weighted_avg_nb_d)

    per_decision_sevpi = max(0.0, term1_sevpi - term2_sevpi)

    if population is not None or time_horizon is not None or discount_rate is not None:
        if population is None or time_horizon is None:
            raise InputError(
                "To calculate population SEVPI, 'population' and 'time_horizon' must be provided. "
                "'discount_rate' is optional."
            )
        multiplier = _calculate_population_multiplier(
            population, time_horizon, discount_rate
        )
        return float(per_decision_sevpi * multiplier)

    return float(per_decision_sevpi)


def structural_evppi(
    # Similar complex arguments as structural_evpi, plus definition of
    # which part of structural uncertainty is being resolved (e.g., a subset of models,
    # or parameters governing the choice between models).
    *args,
    **kwargs,
) -> float:
    """Calculate the Expected Value of Partial Perfect Information for Model Structure (Structural EVPPI).

    SEVPPI quantifies the expected gain from resolving uncertainty about a specific
    aspect of model structure, or distinguishing between a subset of model structures.

    This is a highly complex VOI metric, often requiring bespoke problem formulation.

    Args:
        *args, **kwargs: Placeholder for arguments.

    Returns
    -------
        float: The calculated Structural EVPPI.

    Raises
    ------
        NotImplementedError: This method is a placeholder for v0.1.
    """
    raise PyVoiNotImplementedError(
        "Structural EVPPI is a highly specialized and complex VOI method. "
        "Not implemented in v0.1.",
    )


if __name__ == "__main__":
    print("--- Testing structural.py (Placeholders) ---")

    try:
        structural_evpi([], [], [])  # type: ignore
    except PyVoiNotImplementedError as e:
        print(f"Caught expected error for structural_evpi: {e}")
    else:
        raise AssertionError("structural_evpi did not raise PyVoiNotImplementedError.")

    try:
        structural_evppi()
    except PyVoiNotImplementedError as e:
        print(f"Caught expected error for structural_evppi: {e}")
    else:
        raise AssertionError("structural_evppi did not raise PyVoiNotImplementedError.")

    print("--- structural.py placeholder tests completed ---")
