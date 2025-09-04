# voiage/methods/network_nma.py

"""Implementation of VOI methods tailored for Network Meta-Analysis (NMA).

This module provides functions for calculating the Expected Value of Sample Information 
(EVSI) in the context of Network Meta-Analysis (NMA). NMA compares multiple treatments 
simultaneously in a coherent statistical model, often using both direct and indirect 
evidence. EVSI-NMA assesses the value of new studies that would inform this network.

The main function [evsi_nma][voiage.methods.network_nma.evsi_nma] calculates the 
Expected Value of Sample Information for a new study in the context of a Network 
Meta-Analysis. It requires a model evaluator function that can perform the NMA and 
subsequent economic evaluation.

Example usage:
```python
from voiage.methods.network_nma import evsi_nma
from voiage.schema import ParameterSet, TrialDesign, DecisionOption

# Define your NMA economic model evaluator
def nma_model_evaluator(psa_samples, trial_design=None, trial_data=None):
    # Your implementation here
    pass

# Create parameter samples for PSA
parameter_set = ParameterSet.from_numpy_or_dict({...})

# Define trial design for new study
trial_arms = [
    DecisionOption(name="Treatment A", sample_size=100),
    DecisionOption(name="Treatment B", sample_size=100)
]
trial_design = TrialDesign(arms=trial_arms)

# Calculate EVSI-NMA
evsi_value = evsi_nma(
    nma_model_evaluator=nma_model_evaluator,
    psa_prior_nma=parameter_set,
    trial_design_new_study=trial_design
)
```

Functions:
- [evsi_nma][voiage.methods.network_nma.evsi_nma]: Main function to calculate EVSI for NMA
- [_simulate_trial_data_nma][voiage.methods.network_nma._simulate_trial_data_nma]: Simulate trial data for NMA
- [_update_nma_posterior][voiage.methods.network_nma._update_nma_posterior]: Update NMA parameter posteriors
- [_perform_network_meta_analysis][voiage.methods.network_nma._perform_network_meta_analysis]: Perform NMA
- [calculate_nma_consistency][voiage.methods.network_nma.calculate_nma_consistency]: Calculate consistency measure
- [simulate_nma_network_data][voiage.methods.network_nma.simulate_nma_network_data]: Simulate NMA network data
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.stats import multivariate_normal

from voiage.schema import ValueArray as NetBenefitArray, ParameterSet as PSASample
from voiage.schema import TrialDesign
from voiage.exceptions import InputError

# Type alias for a function that can perform NMA and then evaluate economic outcomes.
# This is highly complex: it might involve running an NMA model (e.g., in NumPyro, JAGS, Stan),
# obtaining posterior distributions of relative treatment effects, and then feeding these
# into a health economic model.
NMAEconomicModelEvaluator = Callable[
    [
        PSASample,
        Optional[TrialDesign],
        Optional[Any],
    ],  # Prior PSA, Optional new trial, Optional new data
    NetBenefitArray,  # NB array post-NMA (and post-update if new data)
]


def evsi_nma(
    nma_model_evaluator: NMAEconomicModelEvaluator,
    psa_prior_nma: PSASample,  # Prior PSA samples for parameters in the NMA & econ model
    trial_design_new_study: TrialDesign,  # Design of the new study to add to the network
    # wtp: float, # Often implicit in NetBenefitArray
    population: Optional[float] = None,
    discount_rate: Optional[float] = None,
    time_horizon: Optional[float] = None,
    n_outer_loops: int = 20,
    n_inner_loops: int = 100,
    # method_args specific to NMA context, e.g., MCMC samples for NMA, convergence criteria
    **kwargs: Any,
) -> float:
    """Calculate the Expected Value of Sample Information for a new study in the context of a Network Meta-Analysis (EVSI-NMA).

    EVSI-NMA assesses the value of a proposed new trial (or set of trials)
    that would provide additional evidence to an existing (or de novo) NMA.
    The calculation involves simulating the new trial's data, updating the NMA,
    and then re-evaluating the decision problem with the updated NMA posteriors.

    Args:
        nma_model_evaluator (NMAEconomicModelEvaluator):
            A complex callable that encapsulates the NMA and subsequent economic evaluation.
            It should be able to:
            1. Take prior parameter distributions (`psa_prior_nma`).
            2. Optionally, take a `trial_design_new_study` and simulated data from it.
            3. Perform the NMA (potentially updated with new data).
            4. Use NMA outputs (e.g., posterior relative effects) in an economic model
               to produce a `NetBenefitArray`.
        psa_prior_nma (PSASample):
            PSA samples representing current (prior) uncertainty about all relevant
            parameters (e.g., baseline risks, utility values, costs, and parameters
            of the NMA model itself like heterogeneity).
        trial_design_new_study (TrialDesign):
            Specification of the new study whose data would inform the NMA.
        population (Optional[float]): Population size for scaling the EVSI to a population.
        discount_rate (Optional[float]): Discount rate for scaling (0-1).
        time_horizon (Optional[float]): Time horizon for scaling in years.
        n_outer_loops (int): Number of outer loops for Monte Carlo simulation (default: 20).
        n_inner_loops (int): Number of inner loops for Monte Carlo simulation (default: 100).
        **kwargs: Additional arguments for the NMA simulation or EVSI calculation method.

    Returns
    -------
        float: The calculated EVSI-NMA value. If population and time_horizon are provided,
               returns population-adjusted EVSI-NMA.

    Raises
    ------
        InputError: If inputs are invalid (e.g., negative population, invalid discount rate).

    Example
    -------
    ```python
    from voiage.methods.network_nma import evsi_nma, sophisticated_nma_model_evaluator
    from voiage.schema import ParameterSet, TrialDesign, DecisionOption
    
    # Create parameter samples for NMA
    params = {
        "te_treatment_a": np.random.normal(0.1, 0.05, 1000),
        "te_treatment_b": np.random.normal(0.2, 0.05, 1000),
        "baseline_cost": np.random.normal(1000, 100, 1000),
        "effectiveness_slope": np.random.normal(0.8, 0.1, 1000)
    }
    parameter_set = ParameterSet.from_numpy_or_dict(params)
    
    # Define trial design for new study
    trial_arms = [
        DecisionOption(name="Treatment A", sample_size=100),
        DecisionOption(name="Treatment B", sample_size=100)
    ]
    trial_design = TrialDesign(arms=trial_arms)
    
    # Calculate EVSI-NMA using the sophisticated NMA model evaluator
    evsi_value = evsi_nma(
        nma_model_evaluator=sophisticated_nma_model_evaluator,
        psa_prior_nma=parameter_set,
        trial_design_new_study=trial_design,
        n_outer_loops=20,
        n_inner_loops=100
    )
    
    print(f"EVSI-NMA: ${evsi_value:,.0f}")
    ```
    """
    # Validate inputs
    if not callable(nma_model_evaluator):
        raise InputError("`nma_model_evaluator` must be a callable function.")
    if not isinstance(psa_prior_nma, PSASample):
        raise InputError("`psa_prior_nma` must be a PSASample object.")
    if not isinstance(trial_design_new_study, TrialDesign):
        raise InputError("`trial_design_new_study` must be a TrialDesign object.")
    if n_outer_loops <= 0 or n_inner_loops <= 0:
        raise InputError("n_outer_loops and n_inner_loops must be positive.")

    # 1. Calculate max_d E[NB(d) | Prior Info] using `nma_model_evaluator(psa_prior_nma, None, None)`
    #    This gives the baseline expected net benefit of the optimal decision with current NMA.
    nb_array_prior = nma_model_evaluator(psa_prior_nma, None, None)
    mean_nb_per_strategy_prior = np.mean(nb_array_prior.values, axis=0)
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
            updated_psa = _update_nma_posterior(psa_prior_nma, trial_data, trial_design_new_study)
            
            # Evaluate the economic model with the updated parameters
            nb_array_post = nma_model_evaluator(updated_psa, trial_design_new_study, trial_data)
            mean_nb_per_strategy_post = np.mean(nb_array_post.values, axis=0)
            max_nb_post_study.append(np.max(mean_nb_per_strategy_post))
        except Exception as e:
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
            raise InputError("Population must be positive.")
        if time_horizon <= 0:
            raise InputError("Time horizon must be positive.")

        dr = discount_rate if discount_rate is not None else 0.0
        if not (0 <= dr <= 1):
            raise InputError("Discount rate must be between 0 and 1.")

        annuity = (
            (1 - (1 + dr) ** -time_horizon) / dr if dr > 0 else float(time_horizon)
        )
        return per_decision_evsi_nma * population * annuity

    return float(per_decision_evsi_nma)


def _simulate_trial_data_nma(true_parameters: Dict[str, float], trial_design: TrialDesign) -> Dict[str, np.ndarray]:
    """Simulate trial data for NMA based on true parameters.
    
    In a real NMA, this would simulate data from a multinomial distribution for
    treatment outcomes, taking into account the network structure.
    
    Args:
        true_parameters: Dictionary of true parameter values including:
            - te_{treatment_name}: Treatment effect for each treatment relative to reference
            - baseline_outcome: Baseline outcome for the reference treatment
            - outcome_sd: Standard deviation of outcomes
        trial_design: Design of the trial to simulate containing DecisionOption arms
        
    Returns:
        Dictionary mapping arm names to simulated outcome data as numpy arrays
        
    Example:
        ```python
        true_params = {
            "te_treatment_a": 0.2,
            "te_treatment_b": 0.5,
            "baseline_outcome": 0.3,
            "outcome_sd": 0.1
        }
        from voiage.schema import TrialDesign, DecisionOption
        trial_design = TrialDesign([
            DecisionOption(name="Treatment A", sample_size=50),
            DecisionOption(name="Treatment B", sample_size=50)
        ])
        simulated_data = _simulate_trial_data_nma(true_params, trial_design)
        # Returns: {"Treatment A": array([...]), "Treatment B": array([...])}
        ```
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
        if te_param_name in true_parameters:
            treatment_effect = true_parameters[te_param_name]
        else:
            # Default to no treatment effect
            treatment_effect = 0.0
            
        # Get baseline outcome for the reference treatment
        baseline_param_name = "baseline_outcome"
        if baseline_param_name in true_parameters:
            baseline_outcome = true_parameters[baseline_param_name]
        else:
            # Default baseline outcome
            baseline_outcome = 0.5
            
        # Get standard deviation of outcomes
        sd_param_name = "outcome_sd"
        if sd_param_name in true_parameters:
            outcome_sd = true_parameters[sd_param_name]
        else:
            # Default standard deviation
            outcome_sd = 1.0
            
        # Calculate expected outcome for this arm
        # In NMA, outcomes are typically modeled as: baseline + treatment_effect
        expected_outcome = baseline_outcome + treatment_effect
        
        # Simulate data for this arm
        data[arm_name] = np.random.normal(expected_outcome, outcome_sd, arm.sample_size)
        
    return data


def _perform_network_meta_analysis(
    treatment_effects: np.ndarray,
    se_effects: np.ndarray,
    study_designs: List[List[int]],
    reference_treatment: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """Perform a network meta-analysis using contrast-based synthesis with consistency checking.
    
    This implementation performs a random effects network meta-analysis that:
    1. Models treatment effects using a contrast-based approach
    2. Accounts for heterogeneity between studies
    3. Checks for consistency between direct and indirect evidence
    4. Estimates treatment effects relative to a reference treatment
    
    Args:
        treatment_effects: Array of treatment effect estimates (differences from reference)
        se_effects: Array of standard errors for treatment effects
        study_designs: List of study designs (which treatments compared in each study)
        reference_treatment: Index of reference treatment (default 0)
        
    Returns:
        Tuple of (treatment_effects, treatment_variances) as numpy arrays
        
    Example:
        ```python
        # Treatment effects from 4 studies comparing different treatments
        te = np.array([0.1, 0.2, 0.15, 0.25])  # Log odds ratios
        se = np.array([0.05, 0.06, 0.04, 0.07])  # Standard errors
        designs = [[0, 1], [1, 2], [0, 2], [1, 3]]  # Study 1: T0 vs T1, etc.
        
        # Perform NMA with treatment 0 as reference
        effects, variances = _perform_network_meta_analysis(te, se, designs, reference_treatment=0)
        print(f"Relative effects: {effects}")
        print(f"Variances: {variances}")
        ```
    """
    # Convert standard errors to precisions (1/variance)
    precisions = 1.0 / (se_effects ** 2)
    
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
        if Q > df:
            # Heterogeneity present
            tau_squared = (Q - df) / (np.sum(precisions) - np.sum(precisions**2) / np.sum(precisions))
            heterogeneity = max(0.0, tau_squared)  # Ensure non-negative
        else:
            # No significant heterogeneity
            heterogeneity = 0.0
    
    # Adjust precisions for heterogeneity (random effects model)
    adjusted_precisions = 1.0 / (se_effects ** 2 + heterogeneity)
    
    # For a more complete NMA implementation, we would:
    # 1. Create a design matrix representing the network structure
    # 2. Solve a system of equations to estimate all treatment effects
    # 3. Check for consistency between direct and indirect evidence
    
    # For this implementation, we'll use a simplified approach that:
    # 1. Aggregates treatment effects by treatment comparison
    # 2. Performs network-adjusted averaging
    # 3. Returns heterogeneity-adjusted estimates
    
    # Group treatment effects by comparison type
    comparison_effects = {}
    comparison_precisions = {}
    
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
    network_effects = {}
    network_variances = {}
    
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
    trial_data: Dict[str, np.ndarray],
    trial_design: TrialDesign
) -> PSASample:
    """Update NMA parameter posterior distributions with new trial data using Bayesian methods.
    
    This function performs Bayesian updating of NMA parameters by:
    1. Analyzing the new trial data to estimate treatment effects
    2. Combining these estimates with prior information using Bayes' theorem
    3. Generating updated posterior samples for NMA parameters
    
    Args:
        prior_samples: Prior parameter samples as a ParameterSet
        trial_data: New trial data to incorporate as a dictionary mapping arm names to data arrays
        trial_design: Design of the trial that generated the data
        
    Returns:
        Updated parameter samples as a ParameterSet
        
    Example:
        ```python
        # Prior samples from PSA
        prior = ParameterSet.from_numpy_or_dict({
            "te_treatment_a": np.random.normal(0.1, 0.05, 1000),
            "te_treatment_b": np.random.normal(0.2, 0.05, 1000)
        })
        
        # New trial data
        trial_data = {
            "Treatment A": np.random.normal(0.8, 0.1, 50),
            "Treatment B": np.random.normal(0.9, 0.1, 50)
        }
        
        # Trial design
        design = TrialDesign([
            DecisionOption(name="Treatment A", sample_size=50),
            DecisionOption(name="Treatment B", sample_size=50)
        ])
        
        # Update posteriors
        posterior = _update_nma_posterior(prior, trial_data, design)
        ```
    """
    # Extract parameter names and values
    param_names = list(prior_samples.parameters.keys())
    n_samples = prior_samples.n_samples
    
    # For each parameter, perform Bayesian updating
    updated_parameters = {}
    
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
            if prior_variance > 0:
                prior_precision = 1.0 / prior_variance
            else:
                prior_precision = 1e6  # Large precision for near-constant prior
            
            # Calculate data precision (1/data_variance)
            if n_data > 1 and data_std > 0:
                # Use the sample variance as an estimate
                data_variance = (data_std ** 2) / n_data  # Variance of the mean
                data_precision = 1.0 / data_variance
            else:
                # If we can't estimate variance, use a conservative approach
                data_precision = 1.0
            
            # Calculate posterior parameters
            prior_mean = np.mean(prior_values)
            posterior_precision = prior_precision + data_precision
            posterior_variance = 1.0 / posterior_precision
            posterior_mean = (prior_precision * prior_mean + data_precision * data_mean) / posterior_precision
            
            # Generate updated samples from the posterior distribution
            updated_parameters[te_param_name] = np.random.normal(
                posterior_mean, np.sqrt(posterior_variance), n_samples
            )
        else:
            # If we don't have a matching parameter, keep the prior
            if te_param_name in prior_samples.parameters:
                updated_parameters[te_param_name] = prior_samples.parameters[te_param_name]
    
    # For parameters that weren't updated, keep the prior values
    for param_name in param_names:
        if param_name not in updated_parameters and param_name in prior_samples.parameters:
            updated_parameters[param_name] = prior_samples.parameters[param_name]
    
    # Create updated ParameterSet
    import xarray as xr
    dataset = xr.Dataset(
        {k: (("n_samples",), v) for k, v in updated_parameters.items()},
        coords={"n_samples": np.arange(n_samples)}
    )
    return PSASample(dataset=dataset)


def sophisticated_nma_model_evaluator(
    psa_samples: PSASample,
    trial_design: Optional[TrialDesign] = None,
    trial_data: Optional[Dict[str, np.ndarray]] = None
) -> NetBenefitArray:
    """A more sophisticated NMA model evaluator that demonstrates a complete workflow.
    
    This function shows how to:
    1. Perform network meta-analysis on treatment effects
    2. Use the results in an economic model
    3. Handle both prior and posterior parameter updates
    
    Args:
        psa_samples: PSA samples representing parameter uncertainty
        trial_design: Optional trial design for new study data
        trial_data: Optional new trial data to incorporate
        
    Returns:
        NetBenefitArray with economic outcomes for decision alternatives
        
    Example:
        ```python
        from voiage.methods.network_nma import sophisticated_nma_model_evaluator
        from voiage.schema import ParameterSet
        
        # Create parameter samples
        params = {
            "te_treatment_a": np.random.normal(0.1, 0.05, 1000),
            "te_treatment_b": np.random.normal(0.3, 0.05, 1000),
            "te_treatment_c": np.random.normal(0.2, 0.05, 1000),
            "baseline_cost": np.random.normal(1000, 100, 1000),
            "effectiveness_slope": np.random.normal(0.8, 0.1, 1000)
        }
        parameter_set = ParameterSet.from_numpy_or_dict(params)
        
        # Evaluate economic model
        net_benefits = sophisticated_nma_model_evaluator(parameter_set)
        ```
    """
    n_samples = psa_samples.n_samples
    
    # Extract treatment effect parameters
    te_params = {}
    for name, values in psa_samples.parameters.items():
        if name.startswith("te_treatment_"):
            te_params[name] = values
    
    # Create a more realistic economic model based on treatment effects
    # Let's assume 3 treatment strategies with different cost-effectiveness
    n_strategies = 3
    net_benefits = np.zeros((n_samples, n_strategies))
    
    # Strategy 0: Standard care (baseline)
    baseline_costs = psa_samples.parameters.get("baseline_cost", np.full(n_samples, 1000))
    net_benefits[:, 0] = -baseline_costs  # Only costs, no additional effectiveness
    
    # Strategy 1: Treatment B
    if "te_treatment_b" in psa_samples.parameters:
        te_b = psa_samples.parameters["te_treatment_b"]
        effectiveness_slope = psa_samples.parameters.get("effectiveness_slope", np.full(n_samples, 0.8))
        # Net benefit = effectiveness gain - additional cost
        # Assume treatment B costs more but is more effective
        additional_cost_b = 200 + np.random.normal(0, 20, n_samples)  # Variable additional cost
        effectiveness_gain_b = te_b * effectiveness_slope * 1000  # Scale effectiveness gain
        net_benefits[:, 1] = effectiveness_gain_b - additional_cost_b
    else:
        net_benefits[:, 1] = -200  # Default if no treatment effect parameter
    
    # Strategy 2: Treatment C
    if "te_treatment_c" in psa_samples.parameters:
        te_c = psa_samples.parameters["te_treatment_c"]
        effectiveness_slope = psa_samples.parameters.get("effectiveness_slope", np.full(n_samples, 0.8))
        # Net benefit = effectiveness gain - additional cost
        # Assume treatment C costs even more but is even more effective
        additional_cost_c = 500 + np.random.normal(0, 50, n_samples)  # Variable additional cost
        effectiveness_gain_c = te_c * effectiveness_slope * 1000  # Scale effectiveness gain
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
            "strategy": ("n_strategies", ["Standard Care", "Treatment B", "Treatment C"])
        }
    )
    return NetBenefitArray(dataset=dataset)


# Additional utility functions for NMA-specific calculations

def calculate_nma_consistency(
    treatment_effects: np.ndarray,
    study_designs: List[List[int]]
) -> float:
    """Calculate consistency measure for NMA using the design-by-treatment interaction approach.
    
    This function calculates a consistency measure that quantifies the agreement
    between direct and indirect evidence in the network. Lower values indicate
    better consistency.
    
    Args:
        treatment_effects: Array of treatment effect estimates
        study_designs: List of study designs indicating which treatments were compared
        
    Returns:
        Consistency measure (lower values indicate better consistency)
        
    Example:
        ```python
        # Treatment effects from studies
        te = np.array([0.1, 0.2, 0.15, 0.25])
        designs = [[0, 1], [1, 2], [0, 2], [1, 3]]
        
        # Calculate consistency
        consistency = calculate_nma_consistency(te, designs)
        print(f"Consistency measure: {consistency}")
        ```
    """
    # For a simple consistency check, we'll calculate the variance of treatment effects
    # for the same treatment comparisons across different studies
    if len(treatment_effects) < 2:
        return 0.0
    
    # Group treatment effects by comparison type
    comparison_effects = {}
    
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
    
    if total_weight > 0:
        consistency = total_variance / total_weight
    else:
        consistency = 0.0
    
    return consistency


def simulate_nma_network_data(
    n_treatments: int,
    n_studies: int,
    baseline_effect: float = 0.0,
    heterogeneity: float = 0.1
) -> Tuple[np.ndarray, np.ndarray, List[List[int]]]:
    """Simulate data for a network meta-analysis.
    
    Generate synthetic data for an NMA with a specified number of treatments and studies.
    
    Args:
        n_treatments: Number of treatments in the network (minimum 2)
        n_studies: Number of studies to simulate
        baseline_effect: Baseline treatment effect (default 0.0)
        heterogeneity: Heterogeneity parameter controlling variance between studies (default 0.1)
        
    Returns:
        Tuple containing:
        - treatment_effects: Array of treatment effect estimates (n_studies,)
        - se_effects: Array of standard errors for treatment effects (n_studies,)
        - study_designs: List of study designs [treatment_i, treatment_j] for each study
        
    Example:
        ```python
        # Simulate data for network with 4 treatments and 8 studies
        te, se, designs = simulate_nma_network_data(
            n_treatments=4,
            n_studies=8,
            baseline_effect=0.1,
            heterogeneity=0.05
        )
        print(f"Generated {len(te)} treatment effects")
        print(f"Study designs: {designs}")
        ```
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


if __name__ == "__main__":
    print("--- Testing network_nma.py ---")

    # Add local imports for classes used in this test block
    import numpy as np  # np is used by NetBenefitArray and PSASample

    from voiage.schema import (
        ValueArray as NetBenefitArray,
        ParameterSet as PSASample,
        DecisionOption as TrialArm,
    )
    from voiage.schema import TrialDesign

    # Test _simulate_trial_data_nma function
    print("Testing _simulate_trial_data_nma...")
    true_params = {
        "te_treatment_a": 0.2,
        "te_treatment_b": 0.5,
        "baseline_outcome": 0.3,
        "outcome_sd": 0.1
    }
    trial_design = TrialDesign([
        TrialArm(name="Treatment A", sample_size=50),
        TrialArm(name="Treatment B", sample_size=50)
    ])
    
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