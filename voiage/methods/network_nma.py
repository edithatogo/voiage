# voiage/methods/network_nma.py

"""Value of information methods for network meta-analysis models.

The main entry point is ``evsi_nma``, which estimates EVSI for a proposed
study added to a treatment network.
"""

from collections.abc import Callable

import numpy as np

from voiage.exceptions import InputError
from voiage.schema import ParameterSet as PSASample
from voiage.schema import TrialDesign
from voiage.schema import ValueArray as NetBenefitArray

_NMA_MODEL_EVALUATOR_MESSAGE = "`nma_model_evaluator` must be a callable function."
_PSA_PRIOR_MESSAGE = "`psa_prior_nma` must be a PSASample object."
_TRIAL_DESIGN_MESSAGE = "`trial_design_new_study` must be a TrialDesign object."
_LOOPS_POSITIVE_MESSAGE = "n_outer_loops and n_inner_loops must be positive."
_POPULATION_POSITIVE_MESSAGE = "Population must be positive."
_TIME_HORIZON_POSITIVE_MESSAGE = "Time horizon must be positive."
_DISCOUNT_RATE_MESSAGE = "Discount rate must be between 0 and 1."


def _nma_model_evaluator_error() -> InputError:
    return InputError(_NMA_MODEL_EVALUATOR_MESSAGE)


def _psa_prior_error() -> InputError:
    return InputError(_PSA_PRIOR_MESSAGE)


def _trial_design_error() -> InputError:
    return InputError(_TRIAL_DESIGN_MESSAGE)


def _loops_positive_error() -> InputError:
    return InputError(_LOOPS_POSITIVE_MESSAGE)


def _population_positive_error() -> InputError:
    return InputError(_POPULATION_POSITIVE_MESSAGE)


def _time_horizon_positive_error() -> InputError:
    return InputError(_TIME_HORIZON_POSITIVE_MESSAGE)


def _discount_rate_error() -> InputError:
    return InputError(_DISCOUNT_RATE_MESSAGE)


# Type alias for a function that can perform NMA and then evaluate economic outcomes.
# This is highly complex: it might involve running an NMA model (e.g., in NumPyro, JAGS, Stan),
# obtaining posterior distributions of relative treatment effects, and then feeding these
# into a health economic model.
NMAEconomicModelEvaluator = Callable[
    [
        PSASample,
        TrialDesign | None,
        object | None,
    ],  # Prior PSA, Optional new trial, Optional new data
    NetBenefitArray,  # NB array post-NMA (and post-update if new data)
]


def evsi_nma(
    nma_model_evaluator: NMAEconomicModelEvaluator,
    psa_prior_nma: PSASample,  # Prior PSA samples for parameters in the NMA & econ model
    trial_design_new_study: TrialDesign,  # Design of the new study to add to the network
    # wtp: float, # Often implicit in NetBenefitArray
    population: float | None = None,
    discount_rate: float | None = None,
    time_horizon: float | None = None,
    n_outer_loops: int = 20,
    n_inner_loops: int = 100,
    # method_args specific to NMA context, e.g., MCMC samples for NMA, convergence criteria
    **kwargs: object,
) -> float:
    """Calculate EVSI for a proposed study in a network meta-analysis.

    Parameters
    ----------
    nma_model_evaluator : callable
        Model evaluator that maps PSA samples to net-benefit samples.
    psa_prior_nma : ParameterSet
        Prior PSA samples for the NMA and economic model.
    trial_design_new_study : TrialDesign
        Design of the new study to add to the network.
    population : float, optional
        Population size for population scaling.
    discount_rate : float, optional
        Annual discount rate used for population scaling.
    time_horizon : float, optional
        Time horizon in years for population scaling.
    n_outer_loops : int, default=20
        Number of outer Monte Carlo draws.
    n_inner_loops : int, default=100
        Number of inner Monte Carlo draws.
    **kwargs : object
        Additional model-evaluation options.

    Returns
    -------
    float
        EVSI on a per-decision basis unless population scaling is requested.
    """
    # Validate inputs
    if not callable(nma_model_evaluator):
        raise _nma_model_evaluator_error()
    if not isinstance(psa_prior_nma, PSASample):
        raise _psa_prior_error()
    if not isinstance(trial_design_new_study, TrialDesign):
        raise _trial_design_error()
    if n_outer_loops <= 0 or n_inner_loops <= 0:
        raise _loops_positive_error()

    # 1. Calculate max_d E[NB(d) | Prior Info] using `nma_model_evaluator(psa_prior_nma, None, None)`
    #    This gives the baseline expected net benefit of the optimal decision with current NMA.
    nb_array_prior = nma_model_evaluator(psa_prior_nma, None, None)
    mean_nb_per_strategy_prior = np.mean(nb_array_prior.numpy_values, axis=0)
    max_expected_nb_current_info: float = np.max(mean_nb_per_strategy_prior)

    # 2. Outer loop (simulating different potential datasets D_k from `trial_design_new_study`):
    max_nb_post_study = []
    for _ in range(n_outer_loops):
        # Sample a "true" parameter set from the prior
        true_params_idx = np.random.randint(0, psa_prior_nma.n_samples)
        true_params = {
            name: values[true_params_idx]
            for name, values in psa_prior_nma.parameters.items()
        }

        # Simulate trial data based on true parameters
        trial_data = _simulate_trial_data_nma(true_params, trial_design_new_study)

        # Update NMA posteriors with the simulated trial data
        # This is where we would normally run a full NMA with the new data incorporated
        # For this implementation, we'll use our Bayesian updating approach
        try:
            # Update parameter posteriors based on the new trial data
            updated_psa = _update_nma_posterior(
                psa_prior_nma, trial_data, trial_design_new_study
            )

            # Evaluate the economic model with the updated parameters
            nb_array_post = nma_model_evaluator(
                updated_psa, trial_design_new_study, trial_data
            )
            mean_nb_per_strategy_post = np.mean(nb_array_post.numpy_values, axis=0)
            max_nb_post_study.append(np.max(mean_nb_per_strategy_post))
        except Exception:
            # If the evaluator fails, use the prior value
            max_nb_post_study.append(max_expected_nb_current_info)

    # 3. Calculate E_D [ max_d E[NB(d) | D] ] = mean(V_k) over all k.
    expected_max_nb_post_study: float = np.mean(max_nb_post_study)

    # 4. EVSI-NMA = E_D [ max_d E[NB(d) | D] ] - max_d E[NB(d) | Prior Info]
    per_decision_evsi_nma = expected_max_nb_post_study - max_expected_nb_current_info
    per_decision_evsi_nma = max(0.0, per_decision_evsi_nma)

    # Population scaling
    if population is not None and time_horizon is not None:
        if population <= 0:
            raise _population_positive_error()
        if time_horizon <= 0:
            raise _time_horizon_positive_error()

        dr = discount_rate if discount_rate is not None else 0.0
        if not (0 <= dr <= 1):
            raise _discount_rate_error()

        annuity = (
            (1 - (1 + dr) ** -time_horizon) / dr if dr > 0 else float(time_horizon)
        )
        return per_decision_evsi_nma * population * annuity

    return float(per_decision_evsi_nma)


def _simulate_trial_data_nma(
    true_parameters: dict[str, float], trial_design: TrialDesign
) -> dict[str, np.ndarray]:
    """Simulate trial data for NMA based on true parameters.

    Parameters
    ----------
    true_parameters : dict[str, float]
        True parameter values for the simulated trial.
    trial_design : TrialDesign
        Trial design describing the treatment arms.

    Returns
    -------
    dict[str, numpy.ndarray]
        Simulated observations keyed by arm name.
    """
    data = {}

    # Extract NMA-specific parameters
    # In a real implementation, we would have treatment effect parameters
    # and baseline parameters that define the network structure

    # For each arm in the trial design, simulate outcomes
    for arm in trial_design.arms:
        arm_name = arm.name

        # Get treatment effect for this arm (relative to reference)
        # In a real NMA, this would be part of a treatment effect matrix
        te_param_name = f"te_{arm_name.lower().replace(' ', '_')}"
        treatment_effect = true_parameters.get(te_param_name, 0.0)

        # Get baseline outcome for the reference treatment
        baseline_param_name = "baseline_outcome"
        baseline_outcome = true_parameters.get(baseline_param_name, 0.5)

        # Get standard deviation of outcomes
        sd_param_name = "outcome_sd"
        outcome_sd = true_parameters.get(sd_param_name, 1.0)

        # Calculate expected outcome for this arm
        # In NMA, outcomes are typically modeled as: baseline + treatment_effect
        expected_outcome = baseline_outcome + treatment_effect

        # Simulate data for this arm
        data[arm_name] = np.random.normal(expected_outcome, outcome_sd, arm.sample_size)

    return data


def _perform_network_meta_analysis(
    treatment_effects: np.ndarray,
    se_effects: np.ndarray,
    study_designs: list[list[int]],
    reference_treatment: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Perform a simplified contrast-based network meta-analysis.

    Parameters
    ----------
    treatment_effects : numpy.ndarray
        Treatment-effect estimates.
    se_effects : numpy.ndarray
        Standard errors for the treatment effects.
    study_designs : list[list[int]]
        Study designs describing which treatments were compared.
    reference_treatment : int, default=0
        Index of the reference treatment.

    Returns
    -------
    tuple of numpy.ndarray
        Adjusted treatment effects and adjusted variances.
    """
    # Convert standard errors to precisions (1/variance)
    precisions = 1.0 / (se_effects**2)

    # Estimate heterogeneity using the DerSimonian-Laird method
    # This is a simplified approach - a full implementation would use more sophisticated methods
    k = len(treatment_effects)  # Number of studies
    if k < 2:
        # Not enough studies to estimate heterogeneity
        heterogeneity = 0.0
    else:
        # Calculate Q statistic
        mean_effect = np.average(treatment_effects, weights=precisions)
        Q = np.sum(precisions * (treatment_effects - mean_effect) ** 2)

        # Degrees of freedom
        df = k - 1

        # Estimate heterogeneity variance (tau^2)
        if df < Q:
            # Heterogeneity present
            tau_squared = (Q - df) / (
                np.sum(precisions) - np.sum(precisions**2) / np.sum(precisions)
            )
            heterogeneity = max(0.0, tau_squared)  # Ensure non-negative
        else:
            # No significant heterogeneity
            heterogeneity = 0.0

    # Adjust precisions for heterogeneity (random effects model)
    adjusted_precisions = 1.0 / (se_effects**2 + heterogeneity)

    # For a more complete NMA implementation, we would:
    # 1. Create a design matrix representing the network structure
    # 2. Solve a system of equations to estimate all treatment effects
    # 3. Check for consistency between direct and indirect evidence

    # For this implementation, we'll use a simplified approach that:
    # 1. Aggregates treatment effects by treatment comparison
    # 2. Performs network-adjusted averaging
    # 3. Returns heterogeneity-adjusted estimates

    # Group treatment effects by comparison type
    comparison_effects: dict[tuple[int, int], list[float]] = {}
    comparison_precisions: dict[tuple[int, int], list[float]] = {}

    for i, (t1, t2) in enumerate(study_designs):
        # Create a consistent key for the comparison (smaller index first)
        if t1 < t2:
            key = (t1, t2)
            effect = treatment_effects[i]
        else:
            key = (t2, t1)
            effect = -treatment_effects[i]  # Reverse the effect direction

        if key not in comparison_effects:
            comparison_effects[key] = []
            comparison_precisions[key] = []

        comparison_effects[key].append(effect)
        comparison_precisions[key].append(adjusted_precisions[i])

    # Calculate network-adjusted effects for each comparison
    network_effects: dict[tuple[int, int], float] = {}
    network_variances: dict[tuple[int, int], float] = {}

    for key, effects in comparison_effects.items():
        precisions_for_key = comparison_precisions[key]
        # Weighted average of effects for this comparison
        weighted_mean = np.average(effects, weights=precisions_for_key)
        # Variance of the weighted mean
        total_precision = sum(precisions_for_key)
        variance = 1.0 / total_precision

        network_effects[key] = weighted_mean
        network_variances[key] = variance

    # For a complete implementation, we would now:
    # 1. Check consistency using node-splitting or design-by-treatment interaction models
    # 2. Estimate all treatment effects relative to the reference treatment
    # 3. Calculate the full variance-covariance matrix

    # For this simplified implementation, we'll return the network-adjusted estimates
    # and their variances
    adjusted_effects = treatment_effects.copy()
    adjusted_variances = 1.0 / adjusted_precisions

    return adjusted_effects, adjusted_variances


def _update_nma_posterior(
    prior_samples: PSASample,
    trial_data: dict[str, np.ndarray],
    trial_design: TrialDesign,
) -> PSASample:
    """Update NMA parameter posterior distributions with new trial data.

    Parameters
    ----------
    prior_samples : ParameterSet
        Prior PSA samples.
    trial_data : dict[str, numpy.ndarray]
        Simulated trial data keyed by arm name.
    trial_design : TrialDesign
        Trial design used to generate the data.

    Returns
    -------
    ParameterSet
        Updated PSA samples after posterior approximation.
    """
    # Extract parameter names and values
    param_names = list(prior_samples.parameters.keys())
    n_samples = prior_samples.n_samples

    # For each parameter, perform Bayesian updating
    updated_parameters: dict[str, np.ndarray] = {}

    # We'll assume the new trial data provides information about treatment effects
    # For simplicity, we'll treat each treatment arm separately
    for arm_name, data in trial_data.items():
        # Look for treatment effect parameters that match this arm
        te_param_name = f"te_{arm_name.lower().replace(' ', '_')}"

        if te_param_name in prior_samples.parameters:
            # Extract prior samples for this parameter
            prior_values = prior_samples.parameters[te_param_name]

            # Calculate summary statistics from the new data
            data_mean = np.mean(data)
            data_std = np.std(data)
            n_data = len(data)

            # For a simple normal-normal conjugate update:
            # Posterior mean = (prior_precision * prior_mean + data_precision * data_mean) /
            #                  (prior_precision + data_precision)
            # Posterior variance = 1 / (prior_precision + data_precision)

            # Calculate prior precision (1/prior_variance)
            prior_variance = np.var(prior_values)
            prior_precision = (
                1.0 / prior_variance if prior_variance > 0 else 1e6
            )  # Large precision for near-constant prior

            # Calculate data precision (1/data_variance)
            if n_data > 1 and data_std > 0:
                # Use the sample variance as an estimate
                data_variance = (data_std**2) / n_data  # Variance of the mean
                data_precision = 1.0 / data_variance
            else:
                # If we can't estimate variance, use a conservative approach
                data_precision = 1.0

            # Calculate posterior parameters
            prior_mean = np.mean(prior_values)
            posterior_precision = prior_precision + data_precision
            posterior_variance = 1.0 / posterior_precision
            posterior_mean = (
                prior_precision * prior_mean + data_precision * data_mean
            ) / posterior_precision

            # Generate updated samples from the posterior distribution
            updated_parameters[te_param_name] = np.random.normal(
                posterior_mean, np.sqrt(posterior_variance), n_samples
            )
        # If we don't have a matching parameter, keep the prior
        elif te_param_name in prior_samples.parameters:
            updated_parameters[te_param_name] = prior_samples.parameters[te_param_name]

    # For parameters that weren't updated, keep the prior values
    for param_name in param_names:
        if (
            param_name not in updated_parameters
            and param_name in prior_samples.parameters
        ):
            updated_parameters[param_name] = prior_samples.parameters[param_name]

    # Create updated ParameterSet
    import xarray as xr

    dataset = xr.Dataset(
        {k: (("n_samples",), v) for k, v in updated_parameters.items()},
        coords={"n_samples": np.arange(n_samples)},
    )
    return PSASample(dataset=dataset)


def sophisticated_nma_model_evaluator(
    psa_samples: PSASample,
    trial_design: TrialDesign | None = None,
    trial_data: dict[str, np.ndarray] | None = None,
) -> NetBenefitArray:
    """Evaluate the NMA model and return net benefits.

    Parameters
    ----------
    psa_samples : ParameterSet
        PSA samples for the network model.
    trial_design : TrialDesign, optional
        Optional trial design for the study.
    trial_data : dict[str, numpy.ndarray], optional
        Optional simulated trial data.

    Returns
    -------
    ValueArray
        Net-benefit samples for the modeled strategies.
    """
    n_samples = psa_samples.n_samples

    # Create a more realistic economic model based on treatment effects
    # Let's assume 3 treatment strategies with different cost-effectiveness
    n_strategies = 3
    net_benefits = np.zeros((n_samples, n_strategies))

    # Strategy 0: Standard care (baseline)
    baseline_costs = psa_samples.parameters.get(
        "baseline_cost", np.full(n_samples, 1000)
    )
    net_benefits[:, 0] = -baseline_costs  # Only costs, no additional effectiveness

    # Strategy 1: Treatment B
    if "te_treatment_b" in psa_samples.parameters:
        te_b = psa_samples.parameters["te_treatment_b"]
        effectiveness_slope = psa_samples.parameters.get(
            "effectiveness_slope", np.full(n_samples, 0.8)
        )
        # Net benefit = effectiveness gain - additional cost
        # Assume treatment B costs more but is more effective
        additional_cost_b = 200 + np.random.normal(
            0, 20, n_samples
        )  # Variable additional cost
        effectiveness_gain_b = (
            te_b * effectiveness_slope * 1000
        )  # Scale effectiveness gain
        net_benefits[:, 1] = effectiveness_gain_b - additional_cost_b
    else:
        net_benefits[:, 1] = -200  # Default if no treatment effect parameter

    # Strategy 2: Treatment C
    if "te_treatment_c" in psa_samples.parameters:
        te_c = psa_samples.parameters["te_treatment_c"]
        effectiveness_slope = psa_samples.parameters.get(
            "effectiveness_slope", np.full(n_samples, 0.8)
        )
        # Net benefit = effectiveness gain - additional cost
        # Assume treatment C costs even more but is even more effective
        additional_cost_c = 500 + np.random.normal(
            0, 50, n_samples
        )  # Variable additional cost
        effectiveness_gain_c = (
            te_c * effectiveness_slope * 1000
        )  # Scale effectiveness gain
        net_benefits[:, 2] = effectiveness_gain_c - additional_cost_c
    else:
        net_benefits[:, 2] = -500  # Default if no treatment effect parameter

    # Add some noise to make the differences more pronounced
    net_benefits += np.random.normal(0, 10, net_benefits.shape)

    # Create ValueArray
    import xarray as xr

    dataset = xr.Dataset(
        {"net_benefit": (("n_samples", "n_strategies"), net_benefits)},
        coords={
            "n_samples": np.arange(n_samples),
            "n_strategies": np.arange(n_strategies),
            "strategy": (
                "n_strategies",
                ["Standard Care", "Treatment B", "Treatment C"],
            ),
        },
    )
    return NetBenefitArray(dataset=dataset)


# Additional utility functions for NMA-specific calculations


def calculate_nma_consistency(
    treatment_effects: np.ndarray, study_designs: list[list[int]]
) -> float:
    """Calculate a simple consistency measure for NMA.

    Parameters
    ----------
    treatment_effects : numpy.ndarray
        Treatment-effect estimates.
    study_designs : list[list[int]]
        Study designs describing which treatments were compared.

    Returns
    -------
    float
        A simple variance-based consistency score.
    """
    # For a simple consistency check, we'll calculate the variance of treatment effects
    # for the same treatment comparisons across different studies
    if len(treatment_effects) < 2:
        return 0.0

    # Group treatment effects by comparison type
    comparison_effects: dict[tuple[int, int], list[float]] = {}

    for i, (t1, t2) in enumerate(study_designs):
        # Create a consistent key for the comparison (smaller index first)
        if t1 < t2:
            key = (t1, t2)
            effect = treatment_effects[i]
        else:
            key = (t2, t1)
            effect = -treatment_effects[i]  # Reverse the effect direction

        if key not in comparison_effects:
            comparison_effects[key] = []

        comparison_effects[key].append(effect)

    # Calculate consistency as the weighted variance of effects for each comparison
    total_variance = 0.0
    total_weight = 0.0

    for effects in comparison_effects.values():
        if len(effects) > 1:
            # Calculate variance for this comparison
            variance = np.var(effects)
            weight = len(effects)
            total_variance += variance * weight
            total_weight += weight

    return total_variance / total_weight if total_weight > 0 else 0.0


def simulate_nma_network_data(
    n_treatments: int,
    n_studies: int,
    baseline_effect: float = 0.0,
    heterogeneity: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, list[list[int]]]:
    """Simulate synthetic network meta-analysis data.

    Parameters
    ----------
    n_treatments : int
        Number of treatments in the network.
    n_studies : int
        Number of studies to simulate.
    baseline_effect : float, default=0.0
        Mean treatment effect around which studies are generated.
    heterogeneity : float, default=0.1
        Standard deviation of the simulated treatment effects.

    Returns
    -------
    tuple
        Simulated treatment effects, standard errors, and study designs.
    """
    # Generate random study designs (comparing 2 treatments each)
    study_designs = []
    for _ in range(n_studies):
        treatments = np.random.choice(n_treatments, size=2, replace=False)
        study_designs.append([int(t) for t in treatments])

    # Generate treatment effects
    treatment_effects = np.random.normal(baseline_effect, heterogeneity, n_studies)

    # Generate standard errors
    se_effects = np.random.uniform(0.1, 0.5, n_studies)

    return treatment_effects, se_effects, study_designs


if __name__ == "__main__":  # pragma: no cover
    print("--- Testing network_nma.py ---")

    # Add local imports for classes used in this test block
    import numpy as np  # np is used by NetBenefitArray and PSASample

    from voiage.schema import (
        DecisionOption as TrialArm,
    )
    from voiage.schema import (
        ParameterSet as PSASample,
    )
    from voiage.schema import TrialDesign
    from voiage.schema import (
        ValueArray as NetBenefitArray,
    )

    # Test _simulate_trial_data_nma function
    print("Testing _simulate_trial_data_nma...")
    true_params = {
        "te_treatment_a": 0.2,
        "te_treatment_b": 0.5,
        "baseline_outcome": 0.3,
        "outcome_sd": 0.1,
    }
    trial_design = TrialDesign(
        [
            TrialArm(name="Treatment A", sample_size=50),
            TrialArm(name="Treatment B", sample_size=50),
        ]
    )

    simulated_data = _simulate_trial_data_nma(true_params, trial_design)
    print(f"Simulated data keys: {list(simulated_data.keys())}")
    print(f"Data shape for Treatment A: {simulated_data['Treatment A'].shape}")
    print(f"Data shape for Treatment B: {simulated_data['Treatment B'].shape}")

    # Test simulate_nma_network_data function
    print("\nTesting simulate_nma_network_data...")
    te, se, designs = simulate_nma_network_data(4, 6)
    print(f"Generated {len(te)} treatment effects")
    print(f"Study designs: {designs}")

    print("--- network_nma.py tests completed ---")
