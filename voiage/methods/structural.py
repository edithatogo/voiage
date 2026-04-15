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

from collections.abc import Callable

import numpy as np

from voiage.exceptions import InputError
from voiage.schema import ParameterSet as PSASample
from voiage.schema import ValueArray as NetBenefitArray

# Try to import JAX for JIT compilation
try:
    import jax
    from jax import jit, vmap
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

# Type alias for a function that can evaluate a specific model structure
# It would take parameters and return net benefits for that structure.
ModelStructureEvaluator = Callable[[PSASample], NetBenefitArray]


def structural_evpi(
    model_structure_evaluators: list[ModelStructureEvaluator],
    structure_probabilities: np.ndarray | list[float],
    psa_samples_per_structure: list[
        PSASample
    ],  # PSA samples relevant to each structure
    population: float | None = None,
    discount_rate: float | None = None,
    time_horizon: float | None = None,
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
    # --- Input Validation ---
    if len(model_structure_evaluators) != len(structure_probabilities) or \
       len(model_structure_evaluators) != len(psa_samples_per_structure):
        raise InputError("Input lists for structures, probabilities, and PSA samples must have the same length.")

    # Handle empty lists case
    if not model_structure_evaluators:
        return 0.0  # No structural uncertainty if no alternative structures

    if not np.isclose(np.sum(structure_probabilities), 1.0):
        raise InputError("Structure probabilities must sum to 1.")

    # Convert to numpy arrays for easier handling
    prob_arr = np.asarray(structure_probabilities, dtype=float)

    # --- Calculation ---
    n_structures = len(model_structure_evaluators)
    n_strategies = -1 # Determine from first model output, assume consistent for now

    # Store E_theta|S [max_d NB(d, theta, S)] for each structure S
    expected_max_nb_given_structure_S = [] # List of floats

    # Store E_theta|S [NB(d, theta, S)] for each structure S and decision d
    expected_nb_given_structure_S_decision_d = [] # List of arrays (n_strategies)

    for i in range(n_structures):
        evaluator = model_structure_evaluators[i]
        psa_for_S = psa_samples_per_structure[i]
        nb_array_for_S = evaluator(psa_for_S) # NetBenefitArray (samples x strategies)

        if n_strategies == -1:
            n_strategies = nb_array_for_S.n_strategies
        elif n_strategies != nb_array_for_S.n_strategies:
            raise InputError("All model structures must evaluate the same number of decision strategies.")

        # E_theta|S [NB(d, theta, S)] for this S
        mean_nb_d_given_S = np.mean(nb_array_for_S.numpy_values, axis=0) # Shape (n_strategies,)
        expected_nb_given_structure_S_decision_d.append(mean_nb_d_given_S)

        # E_theta|S [max_d NB(d, theta, S)] for this S
        max_nb_per_sample_S = np.max(nb_array_for_S.numpy_values, axis=1)
        expected_max_nb_S = np.mean(max_nb_per_sample_S)
        expected_max_nb_given_structure_S.append(expected_max_nb_S)

    # Term 1: E_S [ E_theta|S [max_d NB(d, theta, S)] ]
    # This is the expectation over S of (the expected value, within S, of choosing optimally if S is known)
    # This is equivalent to: sum_S ( P(S) * E_theta|S [max_d NB(d, theta, S)] )
    term1_sevpi = np.sum(prob_arr * np.array(expected_max_nb_given_structure_S))

    # Term 2: max_d E_S [E_theta|S [NB(d, theta, S)]]
    # This is: max_d sum_S ( P(S) * E_theta|S [NB(d, theta, S)] )
    # First, calculate the overall expected NB for each decision d, averaging over structures
    # E_overall_NB_d = sum_S ( P(S) * E_theta|S [NB(d, theta, S)] )
    # expected_nb_given_structure_S_decision_d is a list of arrays, needs to be (n_structures, n_strategies)
    all_expected_nb_d_S = np.array(expected_nb_given_structure_S_decision_d) # (n_structures, n_strategies)
    weighted_avg_nb_d = np.sum(
        prob_arr[:, np.newaxis] * all_expected_nb_d_S,
        axis=0
    ) # Shape (n_strategies,)
    term2_sevpi = np.max(weighted_avg_nb_d)

    per_decision_sevpi = term1_sevpi - term2_sevpi
    per_decision_sevpi = max(0.0, per_decision_sevpi)

    # Population scaling (similar to EVPI)
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
        return float(per_decision_sevpi * population * annuity_factor)
    if (
        population is not None
        or time_horizon is not None
        or discount_rate is not None
    ):
        raise InputError(
            "To calculate population SEVPI, 'population' and 'time_horizon' must be provided. "
            "'discount_rate' is optional (defaults to 0 if not provided)."
        )

    return per_decision_sevpi


def structural_evppi(
    model_structure_evaluators: list[ModelStructureEvaluator],
    structure_probabilities: np.ndarray | list[float],
    psa_samples_per_structure: list[PSASample],
    # For SEVPPI, we need to specify which structural uncertainty is being resolved
    # This could be a subset of structures or a partition of the structure space
    structures_of_interest: list[int],  # Indices of structures we're learning about
    population: float | None = None,
    discount_rate: float | None = None,
    time_horizon: float | None = None,
) -> float:
    """Calculate the Expected Value of Partial Perfect Information for Model Structure (Structural EVPPI).

    SEVPPI quantifies the expected gain from resolving uncertainty about a specific
    aspect of model structure, or distinguishing between a subset of model structures.

    Args:
        model_structure_evaluators (List[ModelStructureEvaluator]):
            A list of functions. Each function takes a PSASample object and returns a NetBenefitArray.
        structure_probabilities (Union[np.ndarray, List[float]]):
            Probabilities associated with each model structure being the true one.
        psa_samples_per_structure (List[PSASample]):
            A list of PSASample objects for each model structure.
        structures_of_interest (List[int]):
            Indices of the structures we're learning about (0-indexed).
        population (Optional[float]): Population size for scaling.
        discount_rate (Optional[float]): Discount rate for scaling.
        time_horizon (Optional[float]): Time horizon for scaling.

    Returns
    -------
        float: The calculated Structural EVPPI.

    Raises
    ------
        InputError: If inputs are invalid.
    """
    # --- Input Validation ---
    if len(model_structure_evaluators) != len(structure_probabilities) or \
       len(model_structure_evaluators) != len(psa_samples_per_structure):
        raise InputError("Input lists for structures, probabilities, and PSA samples must have the same length.")

    # Handle empty lists case
    if not model_structure_evaluators:
        return 0.0  # No structural uncertainty if no alternative structures

    if not np.isclose(np.sum(structure_probabilities), 1.0):
        raise InputError("Structure probabilities must sum to 1.")

    # Validate structures_of_interest
    if not structures_of_interest:
        return 0.0  # If no structures of interest, EVPPI is 0
    if not all(isinstance(i, int) and 0 <= i < len(model_structure_evaluators) for i in structures_of_interest):
        raise InputError("structures_of_interest must contain valid indices.")

    # Convert to numpy arrays for easier handling
    prob_arr = np.asarray(structure_probabilities, dtype=float)

    # --- Calculation ---
    n_structures = len(model_structure_evaluators)
    n_strategies = -1 # Determine from first model output, assume consistent for now

    # Store E_theta|S [max_d NB(d, theta, S)] for each structure S
    expected_max_nb_given_structure_S = [] # List of floats

    # Store E_theta|S [NB(d, theta, S)] for each structure S and decision d
    expected_nb_given_structure_S_decision_d = [] # List of arrays (n_strategies)

    for i in range(n_structures):
        evaluator = model_structure_evaluators[i]
        psa_for_S = psa_samples_per_structure[i]
        nb_array_for_S = evaluator(psa_for_S) # NetBenefitArray (samples x strategies)

        if n_strategies == -1:
            n_strategies = nb_array_for_S.n_strategies
        elif n_strategies != nb_array_for_S.n_strategies:
            raise InputError("All model structures must evaluate the same number of decision strategies.")

        # E_theta|S [NB(d, theta, S)] for this S
        mean_nb_d_given_S = np.mean(nb_array_for_S.numpy_values, axis=0) # Shape (n_strategies,)
        expected_nb_given_structure_S_decision_d.append(mean_nb_d_given_S)

        # E_theta|S [max_d NB(d, theta, S)] for this S
        max_nb_per_sample_S = np.max(nb_array_for_S.numpy_values, axis=1)
        expected_max_nb_S = np.mean(max_nb_per_sample_S)
        expected_max_nb_given_structure_S.append(expected_max_nb_S)

    # For SEVPPI, we need to calculate:
    # SEVPPI = E_S_known [max_d E_{S_unknown|S_known} [max_d' E_theta|S_known,S_unknown [NB(d', theta, S_known,S_unknown)]]]
    #          - max_d E_S_known [E_{S_unknown|S_known} [E_theta|S_known,S_unknown [NB(d, theta, S_known,S_unknown)]]]

    # Partition structures into those of interest (known) and others (unknown)
    structures_known = structures_of_interest
    structures_unknown = [i for i in range(n_structures) if i not in structures_known]

    # If all structures are of interest, this becomes SEVPI
    if len(structures_known) == n_structures:
        return structural_evpi(model_structure_evaluators, structure_probabilities, psa_samples_per_structure,
                              population, discount_rate, time_horizon)

    # If no structures are of interest to learn about, EVPPI is 0 (handled above)
    if len(structures_known) == 0:
        return 0.0

    # Calculate conditional probabilities
    prob_known = np.sum(prob_arr[structures_known])
    prob_unknown = np.sum(prob_arr[structures_unknown])

    if prob_known == 0:
        return 0.0  # No probability assigned to structures of interest

    # Normalize probabilities within each partition
    prob_known_normalized = prob_arr[structures_known] / prob_known
    _ = prob_arr[structures_unknown] / prob_unknown if prob_unknown > 0 else np.array([])

    # Term 1: E_S_known [max_d E_{S_unknown|S_known} [max_d' E_theta|S_known,S_unknown [NB(d', theta, S)]]]
    term1_evppi = 0.0

    # For each known structure, calculate the expected value with optimal decision
    for i in structures_known:
        # This is just E_theta|S [max_d NB(d, theta, S)] for structure i
        term1_evppi += prob_known_normalized[structures_known.index(i)] * expected_max_nb_given_structure_S[i]

    # Term 2: max_d E_S_known [E_{S_unknown|S_known} [E_theta|S_known,S_unknown [NB(d, theta, S)]]]
    # Calculate the overall expected NB for each decision d, averaging over structures
    all_expected_nb_d_S = np.array(expected_nb_given_structure_S_decision_d) # (n_structures, n_strategies)

    # Weighted average over known structures
    weighted_avg_nb_d_known = np.sum(
        prob_known_normalized[:, np.newaxis] * all_expected_nb_d_S[structures_known],
        axis=0
    ) # Shape (n_strategies,)

    term2_evppi = np.max(weighted_avg_nb_d_known)

    per_decision_sevppi = term1_evppi - term2_evppi
    per_decision_sevppi = max(0.0, per_decision_sevppi)

    # Population scaling (similar to EVPI)
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
        return float(per_decision_sevppi * population * annuity_factor)
    if (
        population is not None
        or time_horizon is not None
        or discount_rate is not None
    ):
        raise InputError(
            "To calculate population SEVPPI, 'population' and 'time_horizon' must be provided. "
            "'discount_rate' is optional (defaults to 0 if not provided)."
        )

    return per_decision_sevppi


# --- JAX-accelerated versions ---

def structural_evpi_jit(
    all_nb_arrays: list[np.ndarray],
    structure_probabilities: np.ndarray | list[float],
    population: float | None = None,
    discount_rate: float | None = None,
    time_horizon: float | None = None,
) -> float:
    """JIT-compiled version of Structural EVPI for improved performance.

    This version takes pre-evaluated net benefit arrays directly instead of
    evaluator functions, allowing JAX to compile and optimize the computation.

    Args:
        all_nb_arrays (List[np.ndarray]):
            List of net benefit arrays, one per structure.
            Each array has shape (n_samples, n_strategies).
        structure_probabilities (Union[np.ndarray, List[float]]):
            Probabilities associated with each model structure.
        population (Optional[float]): Population size for scaling.
        discount_rate (Optional[float]): Discount rate for scaling.
        time_horizon (Optional[float]): Time horizon for scaling.

    Returns
    -------
        float: The calculated Structural EVPI.

    Raises
    ------
        InputError: If JAX is not available or inputs are invalid.
    """
    if not JAX_AVAILABLE:
        raise InputError("JAX is required for JIT-compiled structural EVPI. Install JAX first.")

    # Input validation
    if len(all_nb_arrays) != len(structure_probabilities):
        raise InputError("Number of net benefit arrays must match number of structure probabilities.")

    if not all_nb_arrays:
        return 0.0

    prob_arr = np.asarray(structure_probabilities, dtype=float)
    if not np.isclose(np.sum(prob_arr), 1.0):
        raise InputError("Structure probabilities must sum to 1.")

    @jit
    def _compute_structural_evpi(nb_all, probs):
        nb_all.shape[0]

        # Calculate expected max NB for each structure
        max_per_sample = jnp.max(nb_all, axis=2)  # (n_structures, n_samples)
        expected_max_per_structure = jnp.mean(max_per_sample, axis=1)  # (n_structures,)

        # Term 1: E_S[E_theta|S[max_d NB]]
        term1 = jnp.sum(probs * expected_max_per_structure)

        # Term 2: max_d E_S[E_theta|S[NB]]
        mean_nb_per_structure = jnp.mean(nb_all, axis=1)  # (n_structures, n_strategies)
        weighted_avg_nb = jnp.sum(probs[:, jnp.newaxis] * mean_nb_per_structure, axis=0)
        term2 = jnp.max(weighted_avg_nb)

        return jnp.maximum(0.0, term1 - term2)

    # Stack arrays for JAX: (n_structures, n_samples, n_strategies)
    n_samples = all_nb_arrays[0].shape[0]
    n_strategies = all_nb_arrays[0].shape[1]
    stacked = np.zeros((len(all_nb_arrays), n_samples, n_strategies), dtype=np.float64)
    for i, arr in enumerate(all_nb_arrays):
        stacked[i] = arr

    result = _compute_structural_evpi(stacked, prob_arr)

    # Population scaling
    per_decision_sevpi = float(result)
    if population is not None and time_horizon is not None:
        current_dr = discount_rate if discount_rate is not None else 0.0
        if current_dr == 0:
            annuity_factor = time_horizon
        else:
            annuity_factor = (1 - (1 + current_dr) ** (-time_horizon)) / current_dr
        return float(per_decision_sevpi * population * annuity_factor)

    return per_decision_sevpi


def structural_evppi_jit(
    all_nb_arrays: list[np.ndarray],
    structure_probabilities: np.ndarray | list[float],
    structures_of_interest: list[int],
    population: float | None = None,
    discount_rate: float | None = None,
    time_horizon: float | None = None,
) -> float:
    """JIT-compiled version of Structural EVPPI for improved performance.

    Args:
        all_nb_arrays (List[np.ndarray]):
            List of net benefit arrays, one per structure.
        structure_probabilities (Union[np.ndarray, List[float]]):
            Probabilities associated with each model structure.
        structures_of_interest (List[int]): Indices of structures to learn about.
        population (Optional[float]): Population size for scaling.
        discount_rate (Optional[float]): Discount rate for scaling.
        time_horizon (Optional[float]): Time horizon for scaling.

    Returns
    -------
        float: The calculated Structural EVPPI.
    """
    if not JAX_AVAILABLE:
        raise InputError("JAX is required for JIT-compiled structural EVPPI. Install JAX first.")

    if len(all_nb_arrays) != len(structure_probabilities):
        raise InputError("Number of net benefit arrays must match number of structure probabilities.")

    if not all_nb_arrays:
        return 0.0

    if not structures_of_interest:
        return 0.0

    prob_arr = np.asarray(structure_probabilities, dtype=float)
    if not np.isclose(np.sum(prob_arr), 1.0):
        raise InputError("Structure probabilities must sum to 1.")

    # If all structures are of interest, this becomes SEVPI
    if len(structures_of_interest) == len(all_nb_arrays):
        return structural_evpi_jit(all_nb_arrays, structure_probabilities,
                                   population, discount_rate, time_horizon)

    @jit
    def _compute_structural_evppi(nb_all, probs, known_mask):
        # Calculate expected max NB for each structure
        max_per_sample = jnp.max(nb_all, axis=2)
        expected_max_per_structure = jnp.mean(max_per_sample, axis=1)

        # Calculate mean NB for each structure and decision
        mean_nb_per_structure = jnp.mean(nb_all, axis=2)

        # Known structure probabilities using mask
        known_probs = jnp.where(known_mask, probs, 0.0)
        prob_known = jnp.sum(known_probs)

        # Avoid division by zero with safe division
        safe_prob_known = jnp.maximum(prob_known, 1e-10)
        known_probs_normalized = known_probs / safe_prob_known

        # Term 1: E_S_known[E_theta|S[max_d NB]]
        term1 = jnp.sum(known_probs_normalized * expected_max_per_structure)

        # Term 2: max_d E_S_known[E_theta|S[NB]]
        mean_nb_known = mean_nb_per_structure * known_mask[:, jnp.newaxis]
        weighted_avg_nb_known = jnp.sum(known_probs_normalized[:, jnp.newaxis] * mean_nb_known, axis=0)
        term2 = jnp.max(weighted_avg_nb_known)

        # Zero out result if no known structures
        result = jnp.maximum(0.0, term1 - term2)
        return jnp.where(prob_known > 0, result, 0.0)

    # Stack arrays
    n_samples = all_nb_arrays[0].shape[0]
    n_strategies = all_nb_arrays[0].shape[1]
    n_structures = len(all_nb_arrays)
    stacked = np.zeros((n_structures, n_samples, n_strategies), dtype=np.float64)
    for i, arr in enumerate(all_nb_arrays):
        stacked[i] = arr

    # Create boolean mask for known structures
    known_mask = np.zeros(n_structures, dtype=np.float64)
    for idx in structures_of_interest:
        known_mask[idx] = 1.0
    result = _compute_structural_evppi(stacked, prob_arr, known_mask)

    # Population scaling
    per_decision_sevppi = float(result)
    if population is not None and time_horizon is not None:
        current_dr = discount_rate if discount_rate is not None else 0.0
        if current_dr == 0:
            annuity_factor = time_horizon
        else:
            annuity_factor = (1 - (1 + current_dr) ** (-time_horizon)) / current_dr
        return float(per_decision_sevppi * population * annuity_factor)

    return per_decision_sevppi


if __name__ == "__main__":
    print("--- Testing structural.py ---")

    # Test with simple example
    # Create mock model structure evaluators
    def mock_evaluator1(psa_sample):
        # Simple evaluator that returns fixed net benefits
        values = np.array([[10, 5], [8, 7], [12, 3]])  # 3 samples, 2 strategies
        return NetBenefitArray.from_numpy(values, ["Strategy A", "Strategy B"])

    def mock_evaluator2(psa_sample):
        # Another simple evaluator
        values = np.array([[6, 9], [7, 8], [5, 10]])  # 3 samples, 2 strategies
        return NetBenefitArray.from_numpy(values, ["Strategy A", "Strategy B"])

    # Create mock PSA samples
    psa1 = PSASample.from_numpy_or_dict({"param1": np.array([1, 2, 3]), "param2": np.array([4, 5, 6])})
    psa2 = PSASample.from_numpy_or_dict({"param1": np.array([7, 8, 9]), "param2": np.array([10, 11, 12])})

    # Test structural_evpi
    try:
        result = structural_evpi(
            [mock_evaluator1, mock_evaluator2],
            [0.6, 0.4],
            [psa1, psa2]
        )
        print(f"structural_evpi result: {result}")
    except Exception as e:
        print(f"Error in structural_evpi: {e}")

    # Test structural_evppi
    try:
        result = structural_evppi(
            [mock_evaluator1, mock_evaluator2],
            [0.6, 0.4],
            [psa1, psa2],
            structures_of_interest=[0]  # Learning about structure 0
        )
        print(f"structural_evppi result: {result}")
    except Exception as e:
        print(f"Error in structural_evppi: {e}")

    print("--- structural.py tests completed ---")
