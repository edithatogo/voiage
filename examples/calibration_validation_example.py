# examples/calibration_validation_example.py

"""Validation example for calibration VOI methods.

This example demonstrates the use of the voi_calibration function with a more
realistic economic model that requires calibration.
"""

import sys
import os
import numpy as np
import xarray as xr

# Add the parent directory to the path to import voiage
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from voiage.methods.calibration import voi_calibration, sophisticated_calibration_modeler
from voiage.schema import ParameterSet, ValueArray


def realistic_cal_modeler(psa_samples, study_design, process_spec):
    """A more realistic calibration study modeler.
    
    This modeler simulates a calibration process where we're trying to calibrate
    a health economic model to match observed clinical trial data.
    """
    n_samples = psa_samples.n_samples
    
    # Extract parameters from PSA samples
    # In a real model, these would be used to run the economic model
    effectiveness_params = psa_samples.parameters.get("effectiveness", np.random.normal(0.7, 0.1, n_samples))
    cost_params = psa_samples.parameters.get("cost", np.random.normal(5000, 500, n_samples))
    
    # Simulate the calibration process
    # In a real implementation, this would involve:
    # 1. Running the model with current parameters
    # 2. Comparing outputs to target values
    # 3. Adjusting parameters to improve fit
    
    # For this example, we'll simulate the effect of calibration by slightly
    # improving the parameter estimates
    calibrated_effectiveness = effectiveness_params + np.random.normal(0.01, 0.005, n_samples)
    calibrated_cost = cost_params - np.random.normal(100, 50, n_samples)
    
    # Create net benefits for 2 strategies (Standard Care vs New Treatment)
    # Strategy 0: Standard Care
    # Strategy 1: New Treatment (assumed to be more effective but more costly)
    nb_standard_care = np.random.normal(50000, 5000, n_samples)  # Base net benefit
    
    # New treatment benefit is based on effectiveness parameter
    treatment_benefit = calibrated_effectiveness * 20000  # Convert effectiveness to monetary benefit
    treatment_cost = calibrated_cost * 0.5  # Treatment cost component
    
    nb_new_treatment = nb_standard_care + treatment_benefit - treatment_cost
    
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


def main():
    """Run the calibration VOI validation example."""
    print("=== Calibration VOI Validation Example ===\n")
    
    # Create parameter samples representing prior uncertainty
    np.random.seed(42)  # For reproducible results
    n_samples = 100
    
    params = {
        "effectiveness": np.random.normal(0.7, 0.1, n_samples),  # Treatment effectiveness
        "cost": np.random.normal(5000, 500, n_samples),          # Treatment cost
        "sd_outcome": np.full(n_samples, 0.15),                  # Outcome standard deviation
    }
    
    parameter_set = ParameterSet.from_numpy_or_dict(params)
    
    # Define calibration study design
    calibration_study_design = {
        "experiment_type": "clinical_trial",
        "sample_size": 200,
        "variables_measured": ["treatment_effect", "cost_per_patient"],
        "duration": 2,  # years
        "follow_up_rate": 0.95  # Expected follow-up rate
    }
    
    # Define calibration process specification
    calibration_process_spec = {
        "method": "bayesian",
        "likelihood_function": "normal",
        "calibration_targets": {
            "target_effectiveness": 0.75,
            "target_cost": 4800
        },
        "convergence_criteria": 0.01
    }
    
    # Calculate VOI for calibration without population scaling
    print("Calculating VOI for calibration study...")
    voi_value = voi_calibration(
        cal_study_modeler=realistic_cal_modeler,
        psa_prior=parameter_set,
        calibration_study_design=calibration_study_design,
        calibration_process_spec=calibration_process_spec,
        n_outer_loops=20
    )
    
    print(f"Calibration VOI (no scaling): ${voi_value:,.2f}")
    
    # Calculate VOI for calibration with population scaling
    print("\nCalculating VOI for calibration study with population scaling...")
    voi_value_scaled = voi_calibration(
        cal_study_modeler=realistic_cal_modeler,
        psa_prior=parameter_set,
        calibration_study_design=calibration_study_design,
        calibration_process_spec=calibration_process_spec,
        population=100000,       # 100,000 patients
        time_horizon=10,         # 10-year time horizon
        discount_rate=0.03,      # 3% discount rate
        n_outer_loops=20
    )
    
    print(f"Calibration VOI (scaled): ${voi_value_scaled:,.2f}")
    
    # Show the impact of different study designs
    print("\n=== Comparing Different Calibration Study Designs ===")
    
    # Smaller study
    small_study_design = {
        "experiment_type": "clinical_trial",
        "sample_size": 100,  # Smaller sample size
        "variables_measured": ["treatment_effect"],
        "duration": 1,
        "follow_up_rate": 0.90
    }
    
    voi_small_study = voi_calibration(
        cal_study_modeler=realistic_cal_modeler,
        psa_prior=parameter_set,
        calibration_study_design=small_study_design,
        calibration_process_spec=calibration_process_spec,
        n_outer_loops=20
    )
    
    print(f"Small study VOI: ${voi_small_study:,.2f}")
    
    # Larger study
    large_study_design = {
        "experiment_type": "clinical_trial",
        "sample_size": 500,  # Larger sample size
        "variables_measured": ["treatment_effect", "cost_per_patient", "quality_of_life"],
        "duration": 3,
        "follow_up_rate": 0.98
    }
    
    voi_large_study = voi_calibration(
        cal_study_modeler=realistic_cal_modeler,
        psa_prior=parameter_set,
        calibration_study_design=large_study_design,
        calibration_process_spec=calibration_process_spec,
        n_outer_loops=20
    )
    
    print(f"Large study VOI: ${voi_large_study:,.2f}")
    
    # Demonstrate the sophisticated calibration modeler
    print("\n=== Using Sophisticated Calibration Modeler ===")
    
    voi_sophisticated = voi_calibration(
        cal_study_modeler=sophisticated_calibration_modeler,
        psa_prior=parameter_set,
        calibration_study_design=calibration_study_design,
        calibration_process_spec=calibration_process_spec,
        n_outer_loops=20
    )
    
    print(f"Sophisticated modeler VOI: ${voi_sophisticated:,.2f}")
    
    print("\n=== Example Complete ===")


if __name__ == "__main__":
    main()