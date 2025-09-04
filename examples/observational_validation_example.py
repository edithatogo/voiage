# examples/observational_validation_example.py

"""Validation example for observational study VOI methods.

This example demonstrates the use of the voi_observational function with a more
realistic economic model for observational studies.
"""

import sys
import os
import numpy as np
import xarray as xr

# Add the parent directory to the path to import voiage
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from voiage.methods.observational import voi_observational
from voiage.schema import ParameterSet, ValueArray


def realistic_obs_modeler(psa_samples, study_design, bias_models):
    """A more realistic observational study modeler.
    
    This modeler simulates an observational study where we're trying to estimate
    treatment effects while accounting for potential biases.
    """
    n_samples = psa_samples.n_samples
    
    # Extract parameters from PSA samples
    effectiveness_params = psa_samples.parameters.get("effectiveness", np.random.normal(0.7, 0.1, n_samples))
    cost_params = psa_samples.parameters.get("cost", np.random.normal(5000, 500, n_samples))
    
    # Get bias parameters
    confounding_strength = bias_models.get("confounding", {}).get("strength", 0.0)
    selection_bias_prob = bias_models.get("selection_bias", {}).get("probability", 0.0)
    
    # Simulate the effect of biases on our estimates
    # Confounding tends to bias estimates toward the null (underestimate true effect)
    biased_effectiveness = effectiveness_params * (1 - confounding_strength) + np.random.normal(0, 0.02, n_samples)
    
    # Selection bias can affect both treatment assignment and outcomes
    # We'll simulate this by adding some noise to our estimates
    final_effectiveness = biased_effectiveness + np.random.normal(0, selection_bias_prob * 0.1, n_samples)
    
    # Create net benefits for 2 strategies (Standard Care vs New Treatment)
    # Strategy 0: Standard Care
    # Strategy 1: New Treatment
    
    nb_standard_care = np.random.normal(50000, 5000, n_samples)  # Base net benefit
    
    # New treatment benefit is based on (biased) effectiveness parameter
    treatment_benefit = final_effectiveness * 20000  # Convert effectiveness to monetary benefit
    treatment_cost = cost_params * 0.5  # Treatment cost component
    
    nb_new_treatment = nb_standard_care + treatment_benefit - treatment_cost
    
    # Ensure there's a meaningful difference between strategies
    nb_new_treatment += 5000  # Make treatment more favorable on average
    
    # Create ValueArray
    dataset = xr.Dataset(
        {"net_benefit": (("n_samples", "n_strategies"), 
                        np.column_stack([nb_standard_care, nb_new_treatment]))},
        coords={
            "n_samples": np.arange(n_samples),
            "n_strategies": np.arange(2),
            "strategy": ("n_strategies", ["Standard Care", "New Treatment"])
        }
    )
    
    return ValueArray(dataset=dataset)


def sophisticated_observational_modeler(psa_samples, study_design, bias_models):
    """A sophisticated observational study modeler that demonstrates bias adjustment.
    
    This function simulates a more sophisticated approach where we attempt to
    adjust for biases using statistical methods.
    """
    import xarray as xr
    from voiage.schema import ValueArray
    
    n_samples = psa_samples.n_samples
    
    # Extract key parameters
    effectiveness = psa_samples.parameters.get("effectiveness", np.random.normal(0.7, 0.1, n_samples))
    cost = psa_samples.parameters.get("cost", np.random.normal(5000, 500, n_samples))
    utility = psa_samples.parameters.get("utility", np.random.normal(0.8, 0.05, n_samples))
    
    # Get bias parameters
    confounding_strength = bias_models.get("confounding", {}).get("strength", 0.0)
    selection_bias_prob = bias_models.get("selection_bias", {}).get("probability", 0.0)
    measurement_error = bias_models.get("measurement_error", {}).get("std_dev", 0.05)
    
    # Simulate the observational study process
    # 1. Generate "observed" data with biases
    observed_effectiveness = effectiveness * (1 - confounding_strength) + np.random.normal(0, measurement_error, n_samples)
    
    # 2. Apply bias adjustment (simplified - in reality this would involve complex statistical methods)
    # We'll simulate adjustment by partially recovering the true effect
    adjustment_factor = 0.7  # How much of the bias we can adjust for
    adjusted_effectiveness = observed_effectiveness * (1 + adjustment_factor * confounding_strength) + np.random.normal(0, 0.01, n_samples)
    
    # 3. Create net benefits with adjusted parameters
    # Base case costs and outcomes
    base_cost_standard = 5000
    base_cost_treatment = 8000
    base_qaly_standard = 7.5
    base_qaly_treatment = 8.2
    
    # Apply parameter variations
    # Standard care costs
    cost_standard = base_cost_standard + cost * 0.1
    qaly_standard = base_qaly_standard + utility * 0.5
    
    # Treatment costs and benefits (using adjusted effectiveness)
    cost_treatment = base_cost_treatment + cost * 0.3
    qaly_treatment = base_qaly_treatment + adjusted_effectiveness * 1.0 + utility * 0.3
    
    # Calculate net benefits (assuming WTP of $50,000/QALY)
    wtp = 50000
    nb_standard = (qaly_standard * wtp) - cost_standard
    nb_treatment = (qaly_treatment * wtp) - cost_treatment
    
    # Add some noise to make it more realistic
    nb_standard += np.random.normal(0, 1000, n_samples)
    nb_treatment += np.random.normal(0, 1500, n_samples)
    
    # Ensure there's some meaningful difference between strategies
    # This helps ensure we get a positive VOI
    nb_treatment += 5000  # Make treatment more favorable on average
    
    # Create ValueArray
    dataset = xr.Dataset(
        {"net_benefit": (("n_samples", "n_strategies"), 
                        np.column_stack([nb_standard, nb_treatment]))},
        coords={
            "n_samples": np.arange(n_samples),
            "n_strategies": np.arange(2),
            "strategy": ("n_strategies", ["Standard Care", "New Treatment"])
        }
    )
    
    return ValueArray(dataset=dataset)


def main():
    """Run the observational study VOI validation example."""
    print("=== Observational Study VOI Validation Example ===\n")
    
    # Create parameter samples representing prior uncertainty
    np.random.seed(42)  # For reproducible results
    n_samples = 100
    
    params = {
        "effectiveness": np.random.normal(0.7, 0.1, n_samples),  # Treatment effectiveness
        "cost": np.random.normal(5000, 500, n_samples),          # Treatment cost
        "utility": np.random.normal(0.8, 0.05, n_samples),       # Utility weights
        "sd_outcome": np.full(n_samples, 0.15),                  # Outcome standard deviation
    }
    
    parameter_set = ParameterSet.from_numpy_or_dict(params)
    
    # Define observational study design
    observational_study_design = {
        "study_type": "cohort",
        "sample_size": 2000,
        "variables_collected": ["treatment", "outcome", "confounders", "demographics"],
        "duration": 5,  # years
        "follow_up_rate": 0.90  # Expected follow-up rate
    }
    
    # Define bias models
    bias_models = {
        "confounding": {"strength": 0.3},      # Moderate confounding
        "selection_bias": {"probability": 0.1}, # 10% selection bias
        "measurement_error": {"std_dev": 0.05}  # Measurement error
    }
    
    # Calculate VOI for observational study without population scaling
    print("Calculating VOI for observational study...")
    voi_value = voi_observational(
        obs_study_modeler=realistic_obs_modeler,
        psa_prior=parameter_set,
        observational_study_design=observational_study_design,
        bias_models=bias_models,
        n_outer_loops=20
    )
    
    print(f"Observational Study VOI (no scaling): ${voi_value:,.2f}")
    
    # Calculate VOI for observational study with population scaling
    print("\nCalculating VOI for observational study with population scaling...")
    population = 100000
    time_horizon = 10
    discount_rate = 0.03
    
    # Let's manually calculate the annuity factor to debug
    dr = discount_rate if discount_rate is not None else 0.0
    annuity = (
        (1 - (1 + dr) ** -time_horizon) / dr if dr > 0 else float(time_horizon)
    )
    print(f"Annuity factor: {annuity}")
    print(f"Population: {population}")
    print(f"Time horizon: {time_horizon}")
    print(f"Discount rate: {discount_rate}")
    
    voi_value_scaled = voi_observational(
        obs_study_modeler=realistic_obs_modeler,
        psa_prior=parameter_set,
        observational_study_design=observational_study_design,
        bias_models=bias_models,
        population=population,       # 100,000 patients
        time_horizon=time_horizon,         # 10-year time horizon
        discount_rate=discount_rate,      # 3% discount rate
        n_outer_loops=20
    )
    
    print(f"Observational Study VOI (scaled): ${voi_value_scaled:,.2f}")
    
    # Show the impact of different bias levels
    print("\n=== Comparing Different Bias Levels ===")
    
    # Low bias scenario
    low_bias_models = {
        "confounding": {"strength": 0.1},      # Low confounding
        "selection_bias": {"probability": 0.05}, # 5% selection bias
        "measurement_error": {"std_dev": 0.02}  # Low measurement error
    }
    
    voi_low_bias = voi_observational(
        obs_study_modeler=realistic_obs_modeler,
        psa_prior=parameter_set,
        observational_study_design=observational_study_design,
        bias_models=low_bias_models,
        n_outer_loops=20
    )
    
    print(f"Low bias VOI: ${voi_low_bias:,.2f}")
    
    # High bias scenario
    high_bias_models = {
        "confounding": {"strength": 0.5},      # High confounding
        "selection_bias": {"probability": 0.2}, # 20% selection bias
        "measurement_error": {"std_dev": 0.1}   # High measurement error
    }
    
    voi_high_bias = voi_observational(
        obs_study_modeler=realistic_obs_modeler,
        psa_prior=parameter_set,
        observational_study_design=observational_study_design,
        bias_models=high_bias_models,
        n_outer_loops=20
    )
    
    print(f"High bias VOI: ${voi_high_bias:,.2f}")
    
    # Demonstrate the sophisticated observational modeler
    print("\n=== Using Sophisticated Observational Modeler ===")
    
    voi_sophisticated = voi_observational(
        obs_study_modeler=sophisticated_observational_modeler,
        psa_prior=parameter_set,
        observational_study_design=observational_study_design,
        bias_models=bias_models,
        n_outer_loops=20
    )
    
    print(f"Sophisticated modeler VOI: ${voi_sophisticated:,.2f}")
    
    print("\n=== Example Complete ===")


if __name__ == "__main__":
    main()