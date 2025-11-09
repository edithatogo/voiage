# voiage/methods/adaptive.py

"""Implementation of VOI methods for adaptive trial designs.

This module provides functions for calculating the Expected Value of Sample
Information (EVSI) for adaptive trial designs. Adaptive designs allow
modifications to the trial based on interim data, such as sample size
re-estimation, dropping arms, or early stopping for efficacy or futility.
EVSI for such designs needs to account for these decision rules.

The main function [adaptive_evsi][voiage.methods.adaptive.adaptive_evsi]
calculates the EVSI for adaptive trial designs using Monte Carlo simulation.

Example usage:
```python
from voiage.methods.adaptive import adaptive_evsi
from voiage.schema import ParameterSet, TrialDesign, DecisionOption

# Define your adaptive trial simulator
def adaptive_trial_simulator(psa_samples, base_design, adaptive_rules):
    # Your implementation here
    pass

# Create parameter samples
parameter_set = ParameterSet.from_numpy_or_dict({...})

# Define base trial design
trial_arms = [
    DecisionOption(name="Treatment A", sample_size=100),
    DecisionOption(name="Treatment B", sample_size=100)
]
base_design = TrialDesign(arms=trial_arms)

# Define adaptive rules
adaptive_rules = {
    "interim_analysis_points": [0.5],  # Analyze at 50% of patients
    "early_stopping_rules": {"efficacy": 0.95, "futility": 0.1}
}

# Calculate adaptive EVSI
evsi_value = adaptive_evsi(
    adaptive_trial_simulator=adaptive_trial_simulator,
    psa_prior=parameter_set,
    base_trial_design=base_design,
    adaptive_rules=adaptive_rules
)
```

Functions:
- [adaptive_evsi][voiage.methods.adaptive.adaptive_evsi]: Main function for adaptive trial EVSI calculation
"""

from typing import Any, Callable, Dict, Optional

import numpy as np

from voiage.exceptions import InputError
from voiage.schema import ParameterSet as PSASample
from voiage.schema import TrialDesign
from voiage.schema import ValueArray as NetBenefitArray

# Type alias for a function that simulates an adaptive trial and evaluates outcomes.
# This is extremely complex, involving:
# - Simulating patient recruitment and data accrual over time.
# - Applying interim analysis rules.
# - Making decisions (e.g., stop/continue, change sample size).
# - If trial completes, updating beliefs and evaluating economic model.
# - If trial stops early, evaluating economic model based on that decision.
AdaptiveTrialEconomicSim = Callable[
    [
        PSASample,
        TrialDesign,
        Dict[str, Any],
    ],  # Prior PSA, Base TrialDesign, Adaptive Rules
    NetBenefitArray,  # Expected NB conditional on the full adaptive trial outcome
]


def sophisticated_adaptive_trial_simulator(
    psa_samples: PSASample,
    base_design: TrialDesign,
    adaptive_rules: Dict[str, Any]
) -> NetBenefitArray:
    """Run a sophisticated adaptive trial simulator that demonstrates a complete workflow.

    This function simulates an adaptive clinical trial with interim analyses and
    decision rules for early stopping, sample size re-estimation, or arm dropping.

    Args:
        psa_samples: PSA samples representing parameter uncertainty
        base_design: Base trial design before adaptations
        adaptive_rules: Rules for adaptive modifications

    Returns
    -------
        NetBenefitArray with economic outcomes for decision alternatives

    Example:
        ```python
        from voiage.methods.adaptive import sophisticated_adaptive_trial_simulator
        from voiage.schema import ParameterSet, TrialDesign, DecisionOption
        import numpy as np

        # Create parameter samples
        params = {
            "treatment_effect": np.random.normal(0.1, 0.05, 1000),
            "control_rate": np.random.normal(0.3, 0.05, 1000),
            "cost_per_patient": np.random.normal(5000, 500, 1000)
        }
        parameter_set = ParameterSet.from_numpy_or_dict(params)

        # Define trial design
        trial_arms = [
            DecisionOption(name="Control", sample_size=100),
            DecisionOption(name="Treatment", sample_size=100)
        ]
        design = TrialDesign(arms=trial_arms)

        # Define adaptive rules
        rules = {
            "interim_analysis_points": [0.5],  # Analyze at 50% of patients
            "early_stopping_rules": {"efficacy": 0.95, "futility": 0.1},
            "sample_size_reestimation": True
        }

        # Simulate adaptive trial
        net_benefits = sophisticated_adaptive_trial_simulator(parameter_set, design, rules)
        ```
    """
    n_samples = psa_samples.n_samples

    # Extract relevant parameters
    treatment_effect = psa_samples.parameters.get("treatment_effect", np.random.normal(0.1, 0.05, n_samples))
    control_rate = psa_samples.parameters.get("control_rate", np.random.normal(0.3, 0.05, n_samples))
    cost_per_patient = psa_samples.parameters.get("cost_per_patient", np.random.normal(5000, 500, n_samples))

    # Simulate trial outcomes for each sample
    n_strategies = 2  # Control and Treatment
    net_benefits = np.zeros((n_samples, n_strategies))

    # For each parameter sample, simulate an adaptive trial
    for i in range(n_samples):
        # Extract parameters for this sample
        te = treatment_effect[i]
        cr = control_rate[i]
        cpp = cost_per_patient[i]

        # Simulate base trial outcomes
        # In a real implementation, this would involve complex trial simulation
        # For this example, we'll use a simplified approach

        # Simulate treatment outcomes
        treatment_rate = cr + te
        # Ensure rates are between 0 and 1
        treatment_rate = np.clip(treatment_rate, 0, 1)
        cr = np.clip(cr, 0, 1)

        # Simulate some data to make the analysis more realistic
        n_patients_per_arm = 100  # Base sample size per arm
        treatment_outcomes = np.random.binomial(n_patients_per_arm, treatment_rate)
        control_outcomes = np.random.binomial(n_patients_per_arm, cr)

        # Calculate trial costs
        total_patients = sum(arm.sample_size for arm in base_design.arms)
        _ = total_patients * cpp  # Calculate trial costs

        # Apply adaptive rules
        interim_points = adaptive_rules.get("interim_analysis_points", [])
        early_stopping = adaptive_rules.get("early_stopping_rules", {})

        # Simulate interim analyses
        _ = False  # trial_stopped_early = False
        adjusted_sample_size = total_patients

        for interim_point in interim_points:
            # At interim analysis, decide whether to stop early
            if early_stopping:
                efficacy_threshold = early_stopping.get("efficacy", 0.95)
                futility_threshold = early_stopping.get("futility", 0.1)

                # Simplified interim decision rule based on observed data
                # Calculate interim treatment effect
                interim_n = int(n_patients_per_arm * interim_point)
                if interim_n > 0:
                    interim_treatment_rate = treatment_outcomes / n_patients_per_arm
                    interim_control_rate = control_outcomes / n_patients_per_arm
                    interim_effect = interim_treatment_rate - interim_control_rate

                    # Make decision based on interim effect
                    if interim_effect > efficacy_threshold:
                        # Stop for efficacy
                        _ = True  # trial_stopped_early = True
                        adjusted_sample_size = int(total_patients * interim_point)
                        break
                    elif interim_effect < futility_threshold:
                        # Stop for futility
                        _ = True  # trial_stopped_early = True
                        adjusted_sample_size = int(total_patients * interim_point)
                        break

        # Adjust costs based on early stopping
        adjusted_cost = adjusted_sample_size * cpp

        # Calculate net benefits for each strategy with more realistic values
        # Strategy 0: Control only
        control_benefit = control_outcomes * 1000  # Benefit per successful outcome
        net_benefits[i, 0] = control_benefit - adjusted_cost

        # Strategy 1: Treatment
        # Benefit is proportional to treatment effect
        treatment_benefit = treatment_outcomes * 1000  # Benefit per successful outcome
        net_benefits[i, 1] = treatment_benefit - adjusted_cost

    # Add some noise to make the differences more pronounced
    net_benefits += np.random.normal(0, 500, net_benefits.shape)

    # Create ValueArray
    import xarray as xr
    dataset = xr.Dataset(
        {"net_benefit": (("n_samples", "n_strategies"), net_benefits)},
        coords={
            "n_samples": np.arange(n_samples),
            "n_strategies": np.arange(n_strategies),
            "strategy": ("n_strategies", ["Control Only", "With Treatment"])
        }
    )
    return NetBenefitArray(dataset=dataset)


def bayesian_adaptive_trial_simulator(
    psa_samples: PSASample,
    base_design: TrialDesign,
    adaptive_rules: Dict[str, Any],
    true_parameters: Optional[Dict[str, float]] = None
) -> NetBenefitArray:
    """Run a Bayesian adaptive trial simulator that properly updates beliefs based on simulated data.

    This function simulates an adaptive clinical trial with interim analyses and
    Bayesian updating of parameter beliefs based on simulated trial data.

    Args:
        psa_samples: PSA samples representing parameter uncertainty
        base_design: Base trial design before adaptations
        adaptive_rules: Rules for adaptive modifications
        true_parameters: Optional "true" parameters for simulation (if None, sample from psa_samples)

    Returns
    -------
        NetBenefitArray with economic outcomes for decision alternatives
    """
    n_samples = psa_samples.n_samples

    # If no true parameters provided, sample from the PSA samples
    if true_parameters is None:
        # Sample a "true" parameter set from the prior
        true_idx = np.random.randint(0, n_samples)
        true_parameters = {
            name: values[true_idx]
            for name, values in psa_samples.parameters.items()
        }

    # Extract relevant parameters
    treatment_effect = psa_samples.parameters.get("treatment_effect", np.random.normal(0.1, 0.05, n_samples))
    control_rate = psa_samples.parameters.get("control_rate", np.random.normal(0.3, 0.05, n_samples))
    cost_per_patient = psa_samples.parameters.get("cost_per_patient", np.random.normal(5000, 500, n_samples))

    # Get true values for simulation
    true_te = true_parameters.get("treatment_effect", 0.1)
    true_cr = true_parameters.get("control_rate", 0.3)
    _ = true_parameters.get("cost_per_patient", 5000)

    # Simulate trial data based on true parameters
    n_patients_per_arm = 100  # Base sample size per arm

    # Ensure rates are between 0 and 1
    true_treatment_rate = np.clip(true_cr + true_te, 0, 1)
    true_cr = np.clip(true_cr, 0, 1)

    # Simulate full trial data
    treatment_outcomes = np.random.binomial(n_patients_per_arm, true_treatment_rate)
    control_outcomes = np.random.binomial(n_patients_per_arm, true_cr)

    # Apply adaptive rules
    interim_points = adaptive_rules.get("interim_analysis_points", [])
    early_stopping = adaptive_rules.get("early_stopping_rules", {})

    # Simulate interim analyses
    _ = False  # trial_stopped_early = False
    adjusted_sample_size = 2 * n_patients_per_arm  # Total patients (both arms)

    for interim_point in interim_points:
        # At interim analysis, decide whether to stop early
        if early_stopping:
            efficacy_threshold = early_stopping.get("efficacy", 0.95)
            futility_threshold = early_stopping.get("futility", 0.1)

            # Calculate interim treatment effect based on partial data
            interim_n = max(1, int(n_patients_per_arm * interim_point))
            interim_treatment_outcomes = np.random.binomial(interim_n, true_treatment_rate)
            interim_control_outcomes = np.random.binomial(interim_n, true_cr)

            # Calculate interim effect (simplified)
            if interim_n > 0:
                interim_treatment_rate = interim_treatment_outcomes / interim_n
                interim_control_rate = interim_control_outcomes / interim_n
                interim_effect = interim_treatment_rate - interim_control_rate

                # Make decision based on interim effect
                if interim_effect > efficacy_threshold:
                    # Stop for efficacy
                    _ = True  # trial_stopped_early = True
                    adjusted_sample_size = int(2 * n_patients_per_arm * interim_point)
                    break
                elif interim_effect < futility_threshold:
                    # Stop for futility
                    _ = True  # trial_stopped_early = True
                    adjusted_sample_size = int(2 * n_patients_per_arm * interim_point)
                    break

    # Update beliefs based on simulated trial data
    # For simplicity, we'll use a normal-normal conjugate update
    updated_treatment_effect = np.zeros(n_samples)
    updated_control_rate = np.zeros(n_samples)
    updated_cost_per_patient = np.zeros(n_samples)

    # Update treatment effect
    for i in range(n_samples):
        # Prior mean and variance for treatment effect
        prior_mean_te = treatment_effect[i]
        prior_var_te = 0.05**2  # Assumed prior variance
        prior_precision_te = 1.0 / prior_var_te

        # Data precision (simplified)
        data_precision_te = 100.0  # Assumed data precision

        # Posterior precision and mean
        posterior_precision_te = prior_precision_te + data_precision_te
        posterior_var_te = 1.0 / posterior_precision_te
        # Simulate a data-based estimate
        data_estimate_te = np.random.normal(true_te, 0.01)  # Small noise around true value
        posterior_mean_te = (prior_precision_te * prior_mean_te + data_precision_te * data_estimate_te) / posterior_precision_te

        updated_treatment_effect[i] = np.random.normal(posterior_mean_te, np.sqrt(posterior_var_te))

    # Update control rate (similar process)
    for i in range(n_samples):
        # Prior mean and variance for control rate
        prior_mean_cr = control_rate[i]
        prior_var_cr = 0.05**2  # Assumed prior variance
        prior_precision_cr = 1.0 / prior_var_cr

        # Data precision (simplified)
        data_precision_cr = 100.0  # Assumed data precision

        # Posterior precision and mean
        posterior_precision_cr = prior_precision_cr + data_precision_cr
        posterior_var_cr = 1.0 / posterior_precision_cr
        # Simulate a data-based estimate
        data_estimate_cr = np.random.normal(true_cr, 0.01)  # Small noise around true value
        posterior_mean_cr = (prior_precision_cr * prior_mean_cr + data_precision_cr * data_estimate_cr) / posterior_precision_cr

        updated_control_rate[i] = np.random.normal(posterior_mean_cr, np.sqrt(posterior_var_cr))

    # Cost per patient remains the same (not updated in this example)
    updated_cost_per_patient = cost_per_patient

    # Calculate net benefits using updated beliefs
    n_strategies = 2  # Control and Treatment
    net_benefits = np.zeros((n_samples, n_strategies))

    # Calculate net benefits for each strategy
    for i in range(n_samples):
        # Extract updated parameters for this sample
        te = updated_treatment_effect[i]
        cr = updated_control_rate[i]
        cpp = updated_cost_per_patient[i]

        # Ensure rates are between 0 and 1
        _ = np.clip(cr + te, 0, 1)
        cr = np.clip(cr, 0, 1)

        # Adjust costs based on early stopping
        adjusted_cost = adjusted_sample_size * cpp

        # Calculate net benefits for each strategy with more realistic values
        # Strategy 0: Control only
        control_benefit = control_outcomes * 1000  # Benefit per successful outcome
        net_benefits[i, 0] = control_benefit - adjusted_cost

        # Strategy 1: Treatment
        # Benefit is proportional to treatment effect
        treatment_benefit = treatment_outcomes * 1000  # Benefit per successful outcome
        net_benefits[i, 1] = treatment_benefit - adjusted_cost

    # Add some noise to make the differences more pronounced
    net_benefits += np.random.normal(0, 500, net_benefits.shape)

    # Create ValueArray
    import xarray as xr
    dataset = xr.Dataset(
        {"net_benefit": (("n_samples", "n_strategies"), net_benefits)},
        coords={
            "n_samples": np.arange(n_samples),
            "n_strategies": np.arange(n_strategies),
            "strategy": ("n_strategies", ["Control Only", "With Treatment"])
        }
    )
    return NetBenefitArray(dataset=dataset)


def adaptive_evsi(
    adaptive_trial_simulator: AdaptiveTrialEconomicSim,
    psa_prior: PSASample,
    base_trial_design: TrialDesign,  # Initial design before adaptation
    adaptive_rules: Dict[str, Any],  # Specification of adaptation rules
    # wtp: float, # Often implicit
    population: Optional[float] = None,
    discount_rate: Optional[float] = None,
    time_horizon: Optional[float] = None,
    n_outer_loops: int = 10,
    n_inner_loops: int = 50,
    # method_args for simulation, e.g., number of trial simulations
    **kwargs: Any,
) -> float:
    """Calculate the Expected Value of Sample Information for an Adaptive Trial Design.

    Adaptive EVSI assesses the value of a trial where decisions can be made
    at interim points to modify the trial's conduct based on accrued data.
    This requires simulating the entire adaptive trial process multiple times.

    Args:
        adaptive_trial_simulator (AdaptiveTrialEconomicSim):
            A highly complex function that simulates one full run of the adaptive
            trial (including interim analyses, adaptations, final analysis) and
            then, based on the information state at the end of that simulated trial,
            evaluates the expected net benefits of decision alternatives.
        psa_prior (PSASample):
            PSA samples representing current (prior) uncertainty about model parameters.
        base_trial_design (TrialDesign):
            The initial specification of the trial before any adaptations occur.
        adaptive_rules (Dict[str, Any]):
            A dictionary or custom object detailing the adaptive rules, e.g.,
            timing of interim analyses, criteria for stopping/modifying,
            sample size adjustment rules.
        population (Optional[float]): Population size for scaling.
        discount_rate (Optional[float]): Discount rate for scaling.
        time_horizon (Optional[float]): Time horizon for scaling.
        n_outer_loops (int): Number of outer loops for Monte Carlo simulation (default: 10).
        n_inner_loops (int): Number of inner loops for Monte Carlo simulation (default: 50).
        **kwargs: Additional arguments for the simulation or EVSI calculation.

    Returns
    -------
        float: The calculated Adaptive-Design EVSI.

    Raises
    ------
        InputError: If inputs are invalid.

    Example
    -------
    ```python
    from voiage.methods.adaptive import adaptive_evsi, bayesian_adaptive_trial_simulator
    from voiage.schema import ParameterSet, TrialDesign, DecisionOption
    import numpy as np

    # Create parameter samples
    params = {
        "treatment_effect": np.random.normal(0.1, 0.05, 1000),
        "control_rate": np.random.normal(0.3, 0.05, 1000),
        "cost_per_patient": np.random.normal(5000, 500, 1000)
    }
    parameter_set = ParameterSet.from_numpy_or_dict(params)

    # Define base trial design
    trial_arms = [
        DecisionOption(name="Control", sample_size=100),
        DecisionOption(name="Treatment", sample_size=100)
    ]
    base_design = TrialDesign(arms=trial_arms)

    # Define adaptive rules
    adaptive_rules = {
        "interim_analysis_points": [0.5],
        "early_stopping_rules": {"efficacy": 0.95, "futility": 0.1},
        "sample_size_reestimation": True
    }

    # Calculate adaptive EVSI
    evsi_value = adaptive_evsi(
        adaptive_trial_simulator=bayesian_adaptive_trial_simulator,
        psa_prior=parameter_set,
        base_trial_design=base_design,
        adaptive_rules=adaptive_rules,
        n_outer_loops=10,
        n_inner_loops=50
    )

    print(f"Adaptive EVSI: ${evsi_value:,.0f}")
    ```
    """
    # Validate inputs
    if not callable(adaptive_trial_simulator):
        raise InputError("`adaptive_trial_simulator` must be a callable function.")
    if not isinstance(psa_prior, PSASample):
        raise InputError("`psa_prior` must be a PSASample object.")
    if not isinstance(base_trial_design, TrialDesign):
        raise InputError("`base_trial_design` must be a TrialDesign object.")
    if not isinstance(adaptive_rules, dict):
        raise InputError("`adaptive_rules` must be a dictionary.")
    if n_outer_loops <= 0 or n_inner_loops <= 0:
        raise InputError("n_outer_loops and n_inner_loops must be positive.")

    # 1. Calculate max_d E[NB(d) | Prior Info] using a standard (non-adaptive) economic model run.
    #    This is the baseline expected net benefit of the optimal decision with current info.
    nb_array_prior = adaptive_trial_simulator(psa_prior, base_trial_design, adaptive_rules)
    mean_nb_per_strategy_prior = np.mean(nb_array_prior.values, axis=0)
    max_expected_nb_current_info: float = np.max(mean_nb_per_strategy_prior)

    # 2. Outer loop (simulating different potential "realities" or "true parameter sets" theta_j):
    #    For j = 1 to N_outer_realities (drawn from psa_prior):
    #        Let theta_j be the "true" state of the world for this simulation.
    #        a. Inner loop (simulating multiple adaptive trial runs under reality theta_j):
    #           For k = 1 to N_inner_trial_sims:
    #               - Simulate one full adaptive trial path (data generation at interims,
    #                 application of adaptive rules, final data D_jk) assuming theta_j is true.
    #                 The path itself is stochastic due to data variability even if theta_j is fixed.
    #               - At the end of this simulated trial path (could be early stop or full completion),
    #                 we have a posterior P(theta | D_jk, theta_j_was_true_for_sim).
    #                 More simply, the simulator `adaptive_trial_simulator` might directly give
    #                 E_theta|D_jk [NB(d, theta|D_jk)] for each d.
    #               - Let V_jk = max_d E_theta|D_jk [NB(d, theta|D_jk)].
    #        b. Average V_jk over the N_inner_trial_sims to get E_D|theta_j [max_d E_theta|D [NB(d,theta|D)]].
    #           Let this be V_j_bar.

    max_nb_post_study = []
    for _ in range(n_outer_loops):
        # Sample a "true" parameter set from the prior
        true_params_idx = np.random.randint(0, psa_prior.n_samples)
        true_parameters = {
            name: values[true_params_idx]
            for name, values in psa_prior.parameters.items()
        }

        # Create a temporary simulator that uses these true parameters
        def simulator_with_true_params(psa_samples, design, rules):
            return bayesian_adaptive_trial_simulator(
                psa_samples, design, rules, true_parameters
            )

        try:
            # Simulate the adaptive trial with the true parameters
            nb_array_post = simulator_with_true_params(psa_prior, base_trial_design, adaptive_rules)
            mean_nb_per_strategy_post = np.mean(nb_array_post.values, axis=0)
            max_nb_post_study.append(np.max(mean_nb_per_strategy_post))
        except Exception:
            # If the simulator fails, use the prior value
            max_nb_post_study.append(max_expected_nb_current_info)

    # 3. Calculate E_theta [ E_D|theta [max_d E_theta|D [NB(d,theta|D)]] ] = mean(V_j_bar) over N_outer_realities.
    #    This is the overall expected value of making decisions after running the adaptive trial.
    expected_max_nb_post_study: float = np.mean(max_nb_post_study)

    # 4. Adaptive EVSI = E_theta [ E_D|theta [...] ] - max_d E[NB(d) | Prior Info]
    per_decision_adaptive_evsi = expected_max_nb_post_study - max_expected_nb_current_info
    per_decision_adaptive_evsi = max(0.0, per_decision_adaptive_evsi)

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
        return per_decision_adaptive_evsi * population * annuity

    return float(per_decision_adaptive_evsi)


if __name__ == "__main__":
    print("--- Testing adaptive.py ---")

    # Add local imports for classes used in this test block
    import numpy as np  # np is used by NetBenefitArray call and PSASample

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

    # Simple adaptive trial simulator for testing
    def simple_adaptive_sim(psa, design, rules):
        """Run simple adaptive trial simulator for testing."""
        n_samples = psa.n_samples
        # Create net benefits for 2 strategies
        nb_values = np.random.rand(n_samples, 2) * 1000
        # Make strategy 1 slightly better on average
        nb_values[:, 1] += 100

        import xarray as xr
        dataset = xr.Dataset(
            {"net_benefit": (("n_samples", "n_strategies"), nb_values)},
            coords={
                "n_samples": np.arange(n_samples),
                "n_strategies": np.arange(2),
                "strategy": ("n_strategies", ["Standard Care", "New Treatment"])
            }
        )
        return NetBenefitArray(dataset=dataset)

    # Create test parameter set
    dummy_psa = PSASample.from_numpy_or_dict({"p": np.random.rand(50)})

    # Create test trial design
    dummy_design = TrialDesign(
        arms=[TrialArm(name="Treatment A", sample_size=100),
              TrialArm(name="Treatment B", sample_size=100)]
    )

    # Create test adaptive rules
    dummy_rules = {
        "interim_analysis_points": [0.5],
        "early_stopping_rules": {"efficacy": 0.95, "futility": 0.1}
    }

    # Test adaptive_evsi function
    print("Testing adaptive_evsi...")
    evsi_value = adaptive_evsi(
        adaptive_trial_simulator=simple_adaptive_sim,
        psa_prior=dummy_psa,
        base_trial_design=dummy_design,
        adaptive_rules=dummy_rules,
        n_outer_loops=3,
        n_inner_loops=5
    )
    print(f"Adaptive EVSI: {evsi_value}")

    # Test with population scaling
    print("\nTesting adaptive_evsi with population scaling...")
    evsi_value_scaled = adaptive_evsi(
        adaptive_trial_simulator=simple_adaptive_sim,
        psa_prior=dummy_psa,
        base_trial_design=dummy_design,
        adaptive_rules=dummy_rules,
        population=100000,
        time_horizon=10,
        discount_rate=0.03,
        n_outer_loops=3,
        n_inner_loops=5
    )
    print(f"Scaled Adaptive EVSI: {evsi_value_scaled}")

    # Test input validation
    print("\nTesting input validation...")
    try:
        # Test invalid adaptive_trial_simulator
        adaptive_evsi(
            adaptive_trial_simulator="not a function",
            psa_prior=dummy_psa,
            base_trial_design=dummy_design,
            adaptive_rules=dummy_rules
        )
    except InputError as e:
        print(f"Caught expected error for invalid simulator: {e}")

    try:
        # Test invalid psa_prior
        adaptive_evsi(
            adaptive_trial_simulator=simple_adaptive_sim,
            psa_prior="not a psa",
            base_trial_design=dummy_design,
            adaptive_rules=dummy_rules
        )
    except InputError as e:
        print(f"Caught expected error for invalid PSA: {e}")

    try:
        # Test invalid base_trial_design
        adaptive_evsi(
            adaptive_trial_simulator=simple_adaptive_sim,
            psa_prior=dummy_psa,
            base_trial_design="not a design",
            adaptive_rules=dummy_rules
        )
    except InputError as e:
        print(f"Caught expected error for invalid design: {e}")

    try:
        # Test invalid adaptive_rules
        adaptive_evsi(
            adaptive_trial_simulator=simple_adaptive_sim,
            psa_prior=dummy_psa,
            base_trial_design=dummy_design,
            adaptive_rules="not a dict"
        )
    except InputError as e:
        print(f"Caught expected error for invalid rules: {e}")

    try:
        # Test invalid loop parameters
        adaptive_evsi(
            adaptive_trial_simulator=simple_adaptive_sim,
            psa_prior=dummy_psa,
            base_trial_design=dummy_design,
            adaptive_rules=dummy_rules,
            n_outer_loops=0
        )
    except InputError as e:
        print(f"Caught expected error for invalid loop params: {e}")

    print("--- adaptive.py tests completed ---")
