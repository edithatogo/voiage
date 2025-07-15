# voiage/methods/structural.py

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

from voiage.schema import ValueArray, ParameterSet
from voiage.exceptions import VoiageNotImplementedError

# Type alias for a function that can evaluate a specific model structure
# It would take parameters and return net benefits for that structure.
ModelStructureEvaluator = Callable[[ParameterSet], ValueArray]


def structural_evpi(
    model_structure_evaluators: List[ModelStructureEvaluator],
    structure_probabilities: Union[np.ndarray, List[float]],
    psa_samples_per_structure: List[
        ParameterSet
    ],
    population: Optional[float] = None,
    discount_rate: Optional[float] = None,
    time_horizon: Optional[float] = None,
) -> float:
    raise VoiageNotImplementedError(
        "Structural EVPI is a complex method requiring careful definition of "
        "model structure evaluation and parameter handling. Not fully implemented in v0.1.",
    )

    # --- Input Validation (Conceptual) ---
    # if len(model_structure_evaluators) != len(structure_probabilities) or \
    #    len(model_structure_evaluators) != len(psa_samples_per_structure):
    #     raise InputError("Input lists for structures, probabilities, and PSA samples must have the same length.")
    # if not np.isclose(np.sum(structure_probabilities), 1.0):
    #     raise InputError("Structure probabilities must sum to 1.")
    # if not model_structure_evaluators:
    #     return 0.0 # No structural uncertainty if no alternative structures

    # --- Calculation (Conceptual) ---
    # n_structures = len(model_structure_evaluators)
    # n_strategies = -1 # Determine from first model output, assume consistent for now

    # Store E_theta|S [NB(d, theta, S)] for each structure S and decision d
    # expected_nb_given_structure_S_decision_d = [] # List of arrays (n_strategies)

    # Store E_theta|S [max_d NB(d, theta, S)] for each structure S
    # expected_max_nb_given_structure_S = [] # List of floats

    # for i in range(n_structures):
    #     evaluator = model_structure_evaluators[i]
    #     psa_for_S = psa_samples_per_structure[i]
    #     nb_array_for_S = evaluator(psa_for_S) # NetBenefitArray (samples x strategies)

    #     if n_strategies == -1:
    #         n_strategies = nb_array_for_S.n_strategies
    #     elif n_strategies != nb_array_for_S.n_strategies:
    #         raise InputError("All model structures must evaluate the same number of decision strategies.")

    #     # E_theta|S [NB(d, theta, S)] for this S
    #     mean_nb_d_given_S = np.mean(nb_array_for_S.values, axis=0) # Shape (n_strategies,)
    #     expected_nb_given_structure_S_decision_d.append(mean_nb_d_given_S)

    #     # E_theta|S [max_d NB(d, theta, S)] for this S
    #     max_nb_per_sample_S = np.max(nb_array_for_S.values, axis=1)
    #     expected_max_nb_S = np.mean(max_nb_per_sample_S)
    #     expected_max_nb_given_structure_S.append(expected_max_nb_S)

    # # Term 1: E_S [max_d E_theta|S [NB(d, theta, S)]] - This is incorrect.
    # # Term 1 should be: E_S [ E_theta|S [max_d NB(d, theta, S)] ]
    # # This is the expectation over S of (the expected value, within S, of choosing optimally if S is known)
    # # This is equivalent to: sum_S ( P(S) * E_theta|S [max_d NB(d, theta, S)] )
    # term1_sevpi = np.sum(
    #     np.array(structure_probabilities) * np.array(expected_max_nb_given_structure_S)
    # )

    # Term 2: max_d E_S [E_theta|S [NB(d, theta, S)]]
    # This is: max_d sum_S ( P(S) * E_theta|S [NB(d, theta, S)] )
    # First, calculate the overall expected NB for each decision d, averaging over structures
    # E_overall_NB_d = sum_S ( P(S) * E_theta|S [NB(d, theta, S)] )
    # expected_nb_given_structure_S_decision_d is a list of arrays, needs to be (n_structures, n_strategies)
    # all_expected_nb_d_S = np.array(expected_nb_given_structure_S_decision_d) # (n_structures, n_strategies)
    # weighted_avg_nb_d = np.sum(
    #     np.array(structure_probabilities)[:, np.newaxis] * all_expected_nb_d_S,
    #     axis=0
    # ) # Shape (n_strategies,)
    # term2_sevpi = np.max(weighted_avg_nb_d)

    # per_decision_sevpi = term1_sevpi - term2_sevpi
    # per_decision_sevpi = max(0.0, per_decision_sevpi)

    # Population scaling (similar to EVPI)
    # ... (omitted for brevity as function raises NotImplementedError) ...

    # return per_decision_sevpi_scaled_or_not


def structural_evppi(
    *args,
    **kwargs,
) -> float:
    raise VoiageNotImplementedError(
        "Structural EVPPI is a highly specialized and complex VOI method. "
        "Not implemented in v0.1.",
    )

