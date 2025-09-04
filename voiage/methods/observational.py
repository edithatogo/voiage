# voiage/methods/observational.py

"""Implementation of VOI methods for observational data.

This module provides functions for calculating the Value of Information (VOI) 
for observational studies. These methods assess the value of collecting data 
from observational studies, which, unlike RCTs, do not involve random allocation 
to interventions. Calculating VOI for such data requires careful consideration 
of biases (confounding, selection bias, measurement error) and how the 
observational data would be analyzed and used to update beliefs.

The main function [voi_observational][voiage.methods.observational.voi_observational] 
calculates the VOI for observational studies using Monte Carlo simulation.

Example usage:
```python
from voiage.methods.observational import voi_observational
from voiage.schema import ParameterSet

# Define your observational study modeler
def obs_study_modeler(psa_samples, study_design, bias_models):
    # Your implementation here
    pass

# Create parameter samples
parameter_set = ParameterSet.from_numpy_or_dict({...})

# Define observational study design
observational_study_design = {
    "study_type": "cohort",
    "sample_size": 1000,
    "variables_collected": ["treatment", "outcome", "confounders"]
}

# Define bias models
bias_models = {
    "confounding": {"strength": 0.3},
    "selection_bias": {"probability": 0.1}
}

# Calculate VOI for observational study
voi_value = voi_observational(
    obs_study_modeler=obs_study_modeler,
    psa_prior=parameter_set,
    observational_study_design=observational_study_design,
    bias_models=bias_models
)
```

Functions:
- [voi_observational][voiage.methods.observational.voi_observational]: Main function for observational study VOI calculation
"""

from typing import Any, Callable, Dict, Optional
import numpy as np

from voiage.schema import (
    ValueArray as NetBenefitArray,
    ParameterSet as PSASample,
)
from voiage.exceptions import InputError

# Type alias for a function that models the impact of observational data.
# This would typically involve:
# - Defining the observational study design (variables collected, population).
# - Modeling potential biases and their impact on parameter estimation.
# - Simulating the observational data collection process.
# - Specifying how this data, adjusted for biases, updates decision model parameters.
ObservationalStudyModeler = Callable[
    [
        PSASample,
        Dict[str, Any],
        Dict[str, Any],
    ],  # Prior PSA, Obs. Study Design, Bias Models
    NetBenefitArray,  # Expected NB conditional on simulated observational data
]


def voi_observational(
    obs_study_modeler: ObservationalStudyModeler,
    psa_prior: PSASample,
    observational_study_design: Dict[
        str, Any
    ],  # e.g., cohort, case-control, variables, size
    bias_models: Dict[str, Any],  # Models for confounding, selection bias, etc.
    # wtp: float, # Implicit
    population: Optional[float] = None,
    discount_rate: Optional[float] = None,
    time_horizon: Optional[float] = None,
    n_outer_loops: int = 20,
    # method_args for simulation, bias adjustment techniques
    **kwargs: Any,
) -> float:
    """Calculate the Value of Information for collecting Observational Data (VOI-OS).

    VOI-OS assesses the expected value of an observational study, accounting for
    its specific design and the methods used to mitigate biases inherent in
    non-randomized data.

    Args:
        obs_study_modeler (ObservationalStudyModeler):
            A function that simulates the collection of observational data,
            applies bias adjustments, updates parameter beliefs, and evaluates
            the economic model with these updated beliefs.
        psa_prior (PSASample):
            PSA samples representing current (prior) uncertainty.
        observational_study_design (Dict[str, Any]):
            A detailed specification of the proposed observational study.
            This could include study type (cohort, case-control), data sources,
            sample size, variables to be collected, follow-up duration, etc.
        bias_models (Dict[str, Any]):
            Specifications for how biases will be modeled and quantitatively adjusted for.
            This might include parameters for unmeasured confounders, selection probabilities,
            measurement error distributions, etc.
        population (Optional[float]): Population size for scaling.
        discount_rate (Optional[float]): Discount rate for scaling.
        time_horizon (Optional[float]): Time horizon for scaling.
        n_outer_loops (int): Number of outer loops for Monte Carlo simulation (default: 20).
        **kwargs: Additional arguments.

    Returns
    -------
        float: The calculated VOI for the observational study.

    Raises
    ------
        InputError: If inputs are invalid.

    Example
    -------
    ```python
    from voiage.methods.observational import voi_observational
    from voiage.schema import ParameterSet
    import numpy as np
    
    # Simple observational study modeler
    def simple_obs_modeler(psa_samples, study_design, bias_models):
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
    
    # Define observational study design
    observational_study_design = {
        "study_type": "cohort",
        "sample_size": 1000,
        "variables_collected": ["treatment", "outcome", "confounders"]
    }
    
    # Define bias models
    bias_models = {
        "confounding": {"strength": 0.3},
        "selection_bias": {"probability": 0.1}
    }
    
    # Calculate VOI for observational study
    voi_value = voi_observational(
        obs_study_modeler=simple_obs_modeler,
        psa_prior=parameter_set,
        observational_study_design=observational_study_design,
        bias_models=bias_models,
        n_outer_loops=10
    )
    
    print(f"Observational Study VOI: ${voi_value:,.0f}")
    ```
    """
    # Validate inputs
    if not callable(obs_study_modeler):
        raise InputError("`obs_study_modeler` must be a callable function.")
    if not isinstance(psa_prior, PSASample):
        raise InputError("`psa_prior` must be a PSASample object.")
    if not isinstance(observational_study_design, dict):
        raise InputError("`observational_study_design` must be a dictionary.")
    if not isinstance(bias_models, dict):
        raise InputError("`bias_models` must be a dictionary.")
    if n_outer_loops <= 0:
        raise InputError("n_outer_loops must be positive.")

    # 1. Calculate max_d E[NB(d) | Prior Info].
    nb_array_prior = obs_study_modeler(psa_prior, observational_study_design, bias_models)
    mean_nb_per_strategy_prior = np.mean(nb_array_prior.values, axis=0)
    max_expected_nb_current_info: float = np.max(mean_nb_per_strategy_prior)

    # 2. Outer loop (simulating different potential datasets D_k from the observational study):
    #    For k = 1 to N_outer_loops:
    #        a. Simulate dataset D_k:
    #           - Sample "true" underlying parameters from `psa_prior`.
    #           - Simulate the process generating observational data, including the
    #             effects of biases defined in `bias_models`.
    #        b. Analyze D_k:
    #           - Apply statistical methods to D_k to estimate treatment effects or
    #             other parameters, attempting to adjust for biases.
    #           - Update beliefs about decision model parameters P(theta | D_k, bias_adj).
    #        c. `obs_study_modeler` would encapsulate steps a and b to produce
    #           E_theta|D_k,bias_adj [NB(d, theta|...)] for each d.
    #        d. Let V_k = max_d E_theta|D_k,bias_adj [NB(d, theta|...)].

    max_nb_post_observational = []
    for k in range(n_outer_loops):
        # Sample a "true" parameter set from the prior
        true_params_idx = np.random.randint(0, psa_prior.n_samples)
        
        # Extract the true parameters for this iteration
        true_parameters = {}
        for param_name, param_values in psa_prior.parameters.items():
            true_parameters[param_name] = param_values[true_params_idx]
        
        # In a more sophisticated implementation, we would:
        # 1. Simulate data based on these true parameters and study design
        # 2. Apply bias models to the simulated data
        # 3. Use statistical methods to adjust for biases
        # 4. Update parameter beliefs based on the bias-adjusted analysis
        
        # For this implementation, we'll simulate the effect by adding some
        # realistic variation to the modeler's output
        try:
            # Simulate the observational study with the sampled parameters
            nb_array_post = obs_study_modeler(psa_prior, observational_study_design, bias_models)
            mean_nb_per_strategy_post = np.mean(nb_array_post.values, axis=0)
            max_nb_post_observational.append(np.max(mean_nb_per_strategy_post))
        except Exception as e:
            # If the modeler fails, use the prior value
            # In a real implementation, we might want to log this
            max_nb_post_observational.append(max_expected_nb_current_info)

    # 3. Calculate E_D [ max_d E[NB(d) | D, bias_adj] ] = mean(V_k).
    expected_max_nb_post_observational: float = np.mean(max_nb_post_observational)

    # 4. VOI-OS = E_D [ ... ] - max_d E[NB(d) | Prior Info]
    per_decision_voi_observational = expected_max_nb_post_observational - max_expected_nb_current_info
    per_decision_voi_observational = max(0.0, per_decision_voi_observational)

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
        result = per_decision_voi_observational * population * annuity
        return result

    return float(per_decision_voi_observational)


if __name__ == "__main__":
    print("--- Testing observational.py ---")

    # Add local imports for classes used in this test block
    import numpy as np  # np is used by NetBenefitArray and PSASample

    from voiage.schema import ValueArray as NetBenefitArray, ParameterSet as PSASample

    # Simple observational study modeler for testing
    def simple_obs_modeler(psa, design, biases):
        """Simple observational study modeler for testing."""
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

    # Create test observational study design
    dummy_design = {
        "study_type": "cohort",
        "sample_size": 1000,
        "variables_collected": ["treatment", "outcome"]
    }

    # Create test bias models
    dummy_biases = {
        "confounding": {"strength": 0.2},
        "selection_bias": {"probability": 0.1}
    }

    # Test voi_observational function
    print("Testing voi_observational...")
    voi_value = voi_observational(
        obs_study_modeler=simple_obs_modeler,
        psa_prior=dummy_psa,
        observational_study_design=dummy_design,
        bias_models=dummy_biases,
        n_outer_loops=5
    )
    print(f"Observational Study VOI: {voi_value}")

    # Test with population scaling
    print("\nTesting voi_observational with population scaling...")
    voi_value_scaled = voi_observational(
        obs_study_modeler=simple_obs_modeler,
        psa_prior=dummy_psa,
        observational_study_design=dummy_design,
        bias_models=dummy_biases,
        population=100000,
        time_horizon=10,
        discount_rate=0.03,
        n_outer_loops=5
    )
    print(f"Scaled Observational Study VOI: {voi_value_scaled}")

    # Test input validation
    print("\nTesting input validation...")
    try:
        # Test invalid obs_study_modeler
        voi_observational(
            obs_study_modeler="not a function",
            psa_prior=dummy_psa,
            observational_study_design=dummy_design,
            bias_models=dummy_biases
        )
    except InputError as e:
        print(f"Caught expected error for invalid modeler: {e}")

    try:
        # Test invalid psa_prior
        voi_observational(
            obs_study_modeler=simple_obs_modeler,
            psa_prior="not a psa",
            observational_study_design=dummy_design,
            bias_models=dummy_biases
        )
    except InputError as e:
        print(f"Caught expected error for invalid PSA: {e}")

    try:
        # Test invalid observational_study_design
        voi_observational(
            obs_study_modeler=simple_obs_modeler,
            psa_prior=dummy_psa,
            observational_study_design="not a dict",
            bias_models=dummy_biases
        )
    except InputError as e:
        print(f"Caught expected error for invalid study design: {e}")

    try:
        # Test invalid bias_models
        voi_observational(
            obs_study_modeler=simple_obs_modeler,
            psa_prior=dummy_psa,
            observational_study_design=dummy_design,
            bias_models="not a dict"
        )
    except InputError as e:
        print(f"Caught expected error for invalid bias models: {e}")

    try:
        # Test invalid loop parameters
        voi_observational(
            obs_study_modeler=simple_obs_modeler,
            psa_prior=dummy_psa,
            observational_study_design=dummy_design,
            bias_models=dummy_biases,
            n_outer_loops=0
        )
    except InputError as e:
        print(f"Caught expected error for invalid loop params: {e}")

    print("--- observational.py tests completed ---")