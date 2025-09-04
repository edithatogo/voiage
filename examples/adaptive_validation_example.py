# examples/adaptive_validation_example.py

"""Validation example for adaptive trial EVSI methods.

This example demonstrates the use of the adaptive_evsi function with a more
realistic economic model for adaptive clinical trials.
"""

import sys
import os
import numpy as np
import xarray as xr

# Add the parent directory to the path to import voiage
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from voiage.methods.adaptive import adaptive_evsi, sophisticated_adaptive_trial_simulator
from voiage.schema import ParameterSet, TrialDesign, DecisionOption


def main():
    """Run the adaptive trial EVSI validation example."""
    print("=== Adaptive Trial EVSI Validation Example ===\n")
    
    # Create parameter samples representing prior uncertainty
    np.random.seed(42)  # For reproducible results
    n_samples = 100
    
    params = {
        "treatment_effect": np.random.normal(0.1, 0.05, n_samples),  # Treatment effect
        "control_rate": np.random.normal(0.3, 0.05, n_samples),      # Control event rate
        "cost_per_patient": np.random.normal(5000, 500, n_samples),  # Cost per patient
        "utility_weight": np.random.normal(0.8, 0.05, n_samples),    # Utility weight
    }
    
    parameter_set = ParameterSet.from_numpy_or_dict(params)
    
    # Define base trial design
    trial_arms = [
        DecisionOption(name="Control", sample_size=150),
        DecisionOption(name="Treatment", sample_size=150)
    ]
    base_design = TrialDesign(arms=trial_arms)
    
    # Define adaptive rules
    adaptive_rules = {
        "interim_analysis_points": [0.5],  # Analyze at 50% of patients
        "early_stopping_rules": {
            "efficacy": 0.95,   # Stop for efficacy if posterior prob > 95%
            "futility": 0.10    # Stop for futility if posterior prob < 10%
        },
        "sample_size_reestimation": True  # Allow sample size adjustment
    }
    
    # Calculate adaptive EVSI without population scaling
    print("Calculating adaptive EVSI...")
    evsi_value = adaptive_evsi(
        adaptive_trial_simulator=sophisticated_adaptive_trial_simulator,
        psa_prior=parameter_set,
        base_trial_design=base_design,
        adaptive_rules=adaptive_rules,
        n_outer_loops=10,
        n_inner_loops=20
    )
    
    print(f"Adaptive EVSI (no scaling): ${evsi_value:,.2f}")
    
    # Calculate adaptive EVSI with population scaling
    print("\nCalculating adaptive EVSI with population scaling...")
    evsi_value_scaled = adaptive_evsi(
        adaptive_trial_simulator=sophisticated_adaptive_trial_simulator,
        psa_prior=parameter_set,
        base_trial_design=base_design,
        adaptive_rules=adaptive_rules,
        population=100000,       # 100,000 patients
        time_horizon=10,         # 10-year time horizon
        discount_rate=0.03,      # 3% discount rate
        n_outer_loops=10,
        n_inner_loops=20
    )
    
    print(f"Adaptive EVSI (scaled): ${evsi_value_scaled:,.2f}")
    
    # Compare with a non-adaptive design
    print("\n=== Comparing Adaptive vs Non-Adaptive Designs ===")
    
    # Non-adaptive rules (no interim analyses)
    non_adaptive_rules = {
        "interim_analysis_points": [],  # No interim analyses
        "early_stopping_rules": {},
        "sample_size_reestimation": False
    }
    
    evsi_non_adaptive = adaptive_evsi(
        adaptive_trial_simulator=sophisticated_adaptive_trial_simulator,
        psa_prior=parameter_set,
        base_trial_design=base_design,
        adaptive_rules=non_adaptive_rules,
        n_outer_loops=10,
        n_inner_loops=20
    )
    
    print(f"Non-adaptive EVSI: ${evsi_non_adaptive:,.2f}")
    print(f"Value of adaptivity: ${evsi_value - evsi_non_adaptive:,.2f}")
    
    # Compare with different adaptive rules
    print("\n=== Comparing Different Adaptive Rules ===")
    
    # More aggressive stopping rules
    aggressive_rules = {
        "interim_analysis_points": [0.3, 0.6],  # Earlier and more frequent analyses
        "early_stopping_rules": {
            "efficacy": 0.90,   # Less stringent efficacy threshold
            "futility": 0.15    # Less stringent futility threshold
        },
        "sample_size_reestimation": True
    }
    
    evsi_aggressive = adaptive_evsi(
        adaptive_trial_simulator=sophisticated_adaptive_trial_simulator,
        psa_prior=parameter_set,
        base_trial_design=base_design,
        adaptive_rules=aggressive_rules,
        n_outer_loops=10,
        n_inner_loops=20
    )
    
    print(f"Aggressive adaptive EVSI: ${evsi_aggressive:,.2f}")
    
    # Conservative stopping rules
    conservative_rules = {
        "interim_analysis_points": [0.5],  # Same analysis point
        "early_stopping_rules": {
            "efficacy": 0.99,   # More stringent efficacy threshold
            "futility": 0.05    # More stringent futility threshold
        },
        "sample_size_reestimation": True
    }
    
    evsi_conservative = adaptive_evsi(
        adaptive_trial_simulator=sophisticated_adaptive_trial_simulator,
        psa_prior=parameter_set,
        base_trial_design=base_design,
        adaptive_rules=conservative_rules,
        n_outer_loops=10,
        n_inner_loops=20
    )
    
    print(f"Conservative adaptive EVSI: ${evsi_conservative:,.2f}")
    
    print("\n=== Example Complete ===")


if __name__ == "__main__":
    main()