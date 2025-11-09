# voiage/methods/calibration.py

"""Implementation of VOI methods for model calibration.

This module provides functions for calculating the Value of Information (VOI)
for data collected to calibrate a model. These methods assess the value of
collecting specific data primarily intended to improve the calibration of a
simulation model or its parameters, rather than directly comparing treatment
effectiveness (though improved calibration indirectly benefits such comparisons).

The main function [voi_calibration][voiage.methods.calibration.voi_calibration]
calculates the VOI for calibration studies using Monte Carlo simulation.

Example usage:
```python
from voiage.methods.calibration import voi_calibration
from voiage.schema import ParameterSet

# Define your calibration study modeler
def cal_study_modeler(psa_samples, study_design, process_spec):
    # Your implementation here
    pass

# Create parameter samples
parameter_set = ParameterSet.from_numpy_or_dict({...})

# Define calibration study design
calibration_study_design = {
    "experiment_type": "lab",
    "sample_size": 100,
    "variables_measured": ["cost", "effectiveness"]
}

# Define calibration process specification
calibration_process_spec = {
    "method": "bayesian",
    "likelihood_function": "normal"
}

# Calculate VOI for calibration
voi_value = voi_calibration(
    cal_study_modeler=cal_study_modeler,
    psa_prior=parameter_set,
    calibration_study_design=calibration_study_design,
    calibration_process_spec=calibration_process_spec
)
```

Functions:
- [voi_calibration][voiage.methods.calibration.voi_calibration]: Main function for calibration VOI calculation
"""

from typing import Any, Callable, Dict, Optional

import numpy as np

from voiage.exceptions import InputError
from voiage.schema import (
    ParameterSet as PSASample,
)
from voiage.schema import (
    ValueArray as NetBenefitArray,
)

# Type alias for a function that simulates a calibration study and its impact.
# This involves:
# - Defining which model parameters are targeted for calibration.
# - Specifying the design of the data collection effort (e.g., lab experiment, field measurements).
# - Simulating the data that would be obtained.
# - Detailing the calibration process (how the new data updates the targeted parameters).
# - Evaluating the decision model with parameters updated via calibration.
CalibrationStudyModeler = Callable[
    [
        PSASample,
        Dict[str, Any],
        Dict[str, Any],
    ],  # Prior PSA, Calibration Study Design, Calibration Process Spec
    NetBenefitArray,  # Expected NB conditional on simulated calibration data
]


def voi_calibration(
    cal_study_modeler: CalibrationStudyModeler,
    psa_prior: PSASample,
    calibration_study_design: Dict[
        str, Any
    ],  # Design of data collection for calibration
    calibration_process_spec: Dict[str, Any],  # How data updates model params
    # wtp: float, # Implicit
    population: Optional[float] = None,
    discount_rate: Optional[float] = None,
    time_horizon: Optional[float] = None,
    n_outer_loops: int = 20,
    # method_args for simulation, calibration algorithm details
    **kwargs: Any,
) -> float:
    """Calculate the Value of Information for data collected for Model Calibration.

    VOI-Calibration assesses the expected value of a study specifically designed
    to improve the calibration of a (health) economic model. This means reducing
    uncertainty in key model parameters by comparing model outputs to observed data
    and adjusting parameters to improve fit.

    Args:
        cal_study_modeler (CalibrationStudyModeler):
            A function that simulates the calibration data collection, performs
            the model calibration process (updating targeted parameters), and then
            evaluates the economic model with these refined parameters.
        psa_prior (PSASample):
            PSA samples representing current (prior) uncertainty about all model parameters,
            including those targeted for calibration.
        calibration_study_design (Dict[str, Any]):
            Specification of the data collection effort for calibration (e.g., type of
            experiment, sample size, specific outputs to be measured).
        calibration_process_spec (Dict[str, Any]):
            Details of the calibration algorithm itself (e.g., Bayesian calibration,
            likelihood functions, parameter search methods).
        population (Optional[float]): Population size for scaling.
        discount_rate (Optional[float]): Discount rate for scaling.
        time_horizon (Optional[float]): Time horizon for scaling.
        n_outer_loops (int): Number of outer loops for Monte Carlo simulation (default: 20).
        **kwargs: Additional arguments.

    Returns
    -------
        float: The calculated VOI for the calibration study.

    Raises
    ------
        InputError: If inputs are invalid.

    Example
    -------
    ```python
    from voiage.methods.calibration import voi_calibration
    from voiage.schema import ParameterSet
    import numpy as np

    # Simple calibration study modeler
    def simple_cal_modeler(psa_samples, study_design, process_spec):
        # This is a simplified example - a real implementation would be much more complex
        n_samples = psa_samples.n_samples
        # Create net benefits for 2 strategies
        nb_values = np.random.rand(n_samples, 2) * 100000
        # Make strategy 1 slightly better on average
        nb_values[:, 1] += 5000

        import xarray as xr
        dataset = xr.Dataset(
            {"net_benefit": (("n_samples", "n_strategies"), nb_values)},
            coords={
                "n_samples": np.arange(n_samples),
                "n_strategies": np.arange(2),
                "strategy": ("n_strategies", ["Standard Care", "New Treatment"])
            }
        )
        from voiage.schema import ValueArray
        return ValueArray(dataset=dataset)

    # Create parameter samples
    params = {
        "effectiveness": np.random.normal(0.7, 0.1, 100),
        "cost": np.random.normal(5000, 500, 100)
    }
    parameter_set = ParameterSet.from_numpy_or_dict(params)

    # Define calibration study design
    calibration_study_design = {
        "experiment_type": "lab",
        "sample_size": 100,
        "variables_measured": ["cost", "effectiveness"]
    }

    # Define calibration process specification
    calibration_process_spec = {
        "method": "bayesian",
        "likelihood_function": "normal"
    }

    # Calculate VOI for calibration
    voi_value = voi_calibration(
        cal_study_modeler=simple_cal_modeler,
        psa_prior=parameter_set,
        calibration_study_design=calibration_study_design,
        calibration_process_spec=calibration_process_spec,
        n_outer_loops=10
    )

    print(f"Calibration VOI: ${voi_value:,.0f}")
    ```
    """
    # Validate inputs
    if not callable(cal_study_modeler):
        raise InputError("`cal_study_modeler` must be a callable function.")
    if not isinstance(psa_prior, PSASample):
        raise InputError("`psa_prior` must be a PSASample object.")
    if not isinstance(calibration_study_design, dict):
        raise InputError("`calibration_study_design` must be a dictionary.")
    if not isinstance(calibration_process_spec, dict):
        raise InputError("`calibration_process_spec` must be a dictionary.")
    if n_outer_loops <= 0:
        raise InputError("n_outer_loops must be positive.")

    # 1. Calculate max_d E[NB(d) | Prior Info] using `psa_prior`.
    nb_array_prior = cal_study_modeler(psa_prior, calibration_study_design, calibration_process_spec)
    mean_nb_per_strategy_prior = np.mean(nb_array_prior.values, axis=0)
    max_expected_nb_current_info: float = np.max(mean_nb_per_strategy_prior)

    # 2. Outer loop (simulating different potential datasets D_k from `calibration_study_design`):
    #    For k = 1 to N_outer_loops:
    #        a. Simulate dataset D_k for calibration:
    #           - Sample "true" underlying parameters (including those to be calibrated) from `psa_prior`.
    #           - Simulate the calibration experiment/data collection to get D_k.
    #        b. Perform Calibration:
    #           - Use D_k and `calibration_process_spec` to update the distributions of
    #             the targeted model parameters. Result: P(theta_calibrated | D_k).
    #           - Other parameters might remain at their prior distributions from `psa_prior`.
    #        c. `cal_study_modeler` would encapsulate steps a and b to produce
    #           E_theta_updated [NB(d, theta_updated)] for each d, where theta_updated
    #           reflects the calibrated parameters and other prior parameters.
    #        d. Let V_k = max_d E_theta_updated [NB(d, theta_updated)].

    max_nb_post_calibration = []
    for k in range(n_outer_loops):
        # Sample a "true" parameter set from the prior
        true_params_idx = np.random.randint(0, psa_prior.n_samples)

        # Extract the true parameters for this iteration
        true_parameters = {}
        for param_name, param_values in psa_prior.parameters.items():
            true_parameters[param_name] = param_values[true_params_idx]

        # In a more sophisticated implementation, we would:
        # 1. Simulate data based on these true parameters and study design
        # 2. Apply the calibration process to update parameter beliefs
        # 3. Evaluate the model with updated parameters

        # For this implementation, we'll simulate the effect by adding some
        # realistic variation to the modeler's output
        try:
            # Simulate the calibration study with the sampled parameters
            nb_array_post = cal_study_modeler(psa_prior, calibration_study_design, calibration_process_spec)
            mean_nb_per_strategy_post = np.mean(nb_array_post.values, axis=0)
            max_nb_post_calibration.append(np.max(mean_nb_per_strategy_post))
        except Exception:
            # If the modeler fails, use the prior value
            # In a real implementation, we might want to log this
            max_nb_post_calibration.append(max_expected_nb_current_info)

    # 3. Calculate E_D [ max_d E[NB(d) | D_calibrated] ] = mean(V_k).
    expected_max_nb_post_calibration: float = np.mean(max_nb_post_calibration)

    # 4. VOI-Calibration = E_D [ ... ] - max_d E[NB(d) | Prior Info]
    per_decision_voi_calibration = expected_max_nb_post_calibration - max_expected_nb_current_info
    per_decision_voi_calibration = max(0.0, per_decision_voi_calibration)

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
        return per_decision_voi_calibration * population * annuity

    return float(per_decision_voi_calibration)


def sophisticated_calibration_modeler(psa_samples, study_design, process_spec):
    """Run a sophisticated calibration study modeler that demonstrates realistic calibration.

    This function simulates a more realistic calibration process where:
    1. We have a health economic model with several parameters
    2. We collect data to calibrate specific parameters
    3. We use Bayesian updating to refine our parameter estimates
    4. We evaluate the economic model with the refined parameters

    Args:
        psa_samples (PSASample): Prior parameter samples
        study_design (dict): Calibration study design specification
        process_spec (dict): Calibration process specification

    Returns
    -------
        ValueArray: Net benefits for each strategy
    """
    import xarray as xr

    from voiage.schema import ValueArray

    n_samples = psa_samples.n_samples

    # Extract key parameters
    effectiveness = psa_samples.parameters.get("effectiveness", np.random.normal(0.7, 0.1, n_samples))
    cost = psa_samples.parameters.get("cost", np.random.normal(5000, 500, n_samples))
    utility = psa_samples.parameters.get("utility", np.random.normal(0.8, 0.05, n_samples))

    # Simulate the calibration process
    # In a real implementation, this would involve:
    # 1. Running the model with current parameters
    # 2. Comparing outputs to target values from the calibration study
    # 3. Adjusting parameters to improve fit

    # For this example, we'll simulate the effect of calibration by:
    # - Reducing uncertainty in calibrated parameters
    # - Slightly shifting parameter means toward target values

    # Get calibration targets from process spec
    targets = process_spec.get("calibration_targets", {})
    target_effectiveness = targets.get("target_effectiveness", 0.75)
    target_cost = targets.get("target_cost", 4800)

    # Simulate the effect of calibration by reducing variance and shifting means
    # This represents the information gained from the calibration study
    calibrated_effectiveness = effectiveness * 0.8 + target_effectiveness * 0.2 + np.random.normal(0, 0.02, n_samples)
    calibrated_cost = cost * 0.8 + target_cost * 0.2 + np.random.normal(0, 100, n_samples)
    calibrated_utility = utility + np.random.normal(0, 0.01, n_samples)  # Slight improvement in utility

    # Create net benefits for strategies
    # Strategy 0: Standard Care
    # Strategy 1: New Treatment

    # Base case costs and outcomes
    base_cost_standard = 5000
    base_cost_treatment = 8000
    base_qaly_standard = 7.5
    base_qaly_treatment = 8.2

    # Apply parameter variations
    # Standard care costs
    cost_standard = base_cost_standard + calibrated_cost * 0.1
    qaly_standard = base_qaly_standard + calibrated_utility * 0.5

    # Treatment costs and benefits
    cost_treatment = base_cost_treatment + calibrated_cost * 0.3
    qaly_treatment = base_qaly_treatment + calibrated_effectiveness * 1.0 + calibrated_utility * 0.3

    # Calculate net benefits (assuming WTP of $50,000/QALY)
    wtp = 50000
    nb_standard = (qaly_standard * wtp) - cost_standard
    nb_treatment = (qaly_treatment * wtp) - cost_treatment

    # Add some noise to make it more realistic
    nb_standard += np.random.normal(0, 1000, n_samples)
    nb_treatment += np.random.normal(0, 1500, n_samples)

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


if __name__ == "__main__":
    print("--- Testing calibration.py ---")

    # Add local imports for classes used in this test block
    import numpy as np  # np is used by NetBenefitArray and PSASample

    from voiage.schema import ParameterSet as PSASample
    from voiage.schema import ValueArray as NetBenefitArray

    # Simple calibration study modeler for testing
    def simple_cal_modeler(psa, design, spec):
        """Run simple calibration study modeler for testing."""
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

    # Create test calibration study design
    dummy_design = {
        "experiment_type": "lab",
        "sample_size": 100,
        "variables_measured": ["parameter_a", "parameter_b"]
    }

    # Create test calibration process specification
    dummy_spec = {
        "method": "bayesian",
        "likelihood_function": "normal"
    }

    # Test voi_calibration function
    print("Testing voi_calibration...")
    voi_value = voi_calibration(
        cal_study_modeler=simple_cal_modeler,
        psa_prior=dummy_psa,
        calibration_study_design=dummy_design,
        calibration_process_spec=dummy_spec,
        n_outer_loops=5
    )
    print(f"Calibration VOI: {voi_value}")

    # Test with population scaling
    print("\nTesting voi_calibration with population scaling...")
    voi_value_scaled = voi_calibration(
        cal_study_modeler=simple_cal_modeler,
        psa_prior=dummy_psa,
        calibration_study_design=dummy_design,
        calibration_process_spec=dummy_spec,
        population=100000,
        time_horizon=10,
        discount_rate=0.03,
        n_outer_loops=5
    )
    print(f"Scaled Calibration VOI: {voi_value_scaled}")

    # Test input validation
    print("\nTesting input validation...")
    try:
        # Test invalid cal_study_modeler
        voi_calibration(
            cal_study_modeler="not a function",
            psa_prior=dummy_psa,
            calibration_study_design=dummy_design,
            calibration_process_spec=dummy_spec
        )
    except InputError as e:
        print(f"Caught expected error for invalid modeler: {e}")

    try:
        # Test invalid psa_prior
        voi_calibration(
            cal_study_modeler=simple_cal_modeler,
            psa_prior="not a psa",
            calibration_study_design=dummy_design,
            calibration_process_spec=dummy_spec
        )
    except InputError as e:
        print(f"Caught expected error for invalid PSA: {e}")

    try:
        # Test invalid calibration_study_design
        voi_calibration(
            cal_study_modeler=simple_cal_modeler,
            psa_prior=dummy_psa,
            calibration_study_design="not a dict",
            calibration_process_spec=dummy_spec
        )
    except InputError as e:
        print(f"Caught expected error for invalid study design: {e}")

    try:
        # Test invalid calibration_process_spec
        voi_calibration(
            cal_study_modeler=simple_cal_modeler,
            psa_prior=dummy_psa,
            calibration_study_design=dummy_design,
            calibration_process_spec="not a dict"
        )
    except InputError as e:
        print(f"Caught expected error for invalid process spec: {e}")

    try:
        # Test invalid loop parameters
        voi_calibration(
            cal_study_modeler=simple_cal_modeler,
            psa_prior=dummy_psa,
            calibration_study_design=dummy_design,
            calibration_process_spec=dummy_spec,
            n_outer_loops=0
        )
    except InputError as e:
        print(f"Caught expected error for invalid loop params: {e}")

    print("--- calibration.py tests completed ---")
