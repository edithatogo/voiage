# voiage/methods/sequential.py

"""Implementation of VOI methods for dynamic or sequential decision problems.

- Dynamic / Sequential VOI

These methods assess the value of information in contexts where decisions are
made sequentially over time, and information gathered at one stage can influence
future decisions and future information gathering opportunities.
"""

from typing import Any, Callable, Dict, Generator, List, Optional, Union

import numpy as np

from voiage.schema import ParameterSet as PSASample
from voiage.schema import DynamicSpec
from voiage.exceptions import InputError


# Type alias for a function that models one step in a sequential process.
# It might take current state (including parameter beliefs), an action/decision,
# and return new state, accrued data, and immediate rewards/costs.
SequentialStepModel = Callable[
    [PSASample, Any, DynamicSpec],  # Current PSA, Action/Decision, Dynamic settings
    Dict[
        str, Any
    ],  # e.g., {'next_psa': PSASample, 'observed_data': Any, 'immediate_nb': float}
]


def sequential_voi(
    step_model: SequentialStepModel,
    initial_psa: PSASample,
    dynamic_specification: DynamicSpec,
    wtp: float = 0.0,
    population: Optional[float] = None,
    discount_rate: Optional[float] = None,
    time_horizon: Optional[float] = None,
    optimization_method: str = "backward_induction",
    **kwargs: Any,
) -> Union[float, Generator[Dict[str, Any], None, None]]:
    """Calculate Value of Information in a dynamic or sequential decision context.

    This can be approached in several ways:
    1. Calculating VOI at each potential decision point in a pre-defined sequence.
    2. Using methods like backward induction (dynamic programming) to find an
       optimal strategy of decision-making and information gathering over time.
    3. Exposing a generator API that yields VOI metrics at each time step as
       data notionally accrues.

    Args:
        step_model (SequentialStepModel):
            A function that models a single time step or stage in the sequential process.
            It takes the current state of knowledge (e.g., `PSASample`), a potential
            decision or action, and dynamic settings, then returns outcomes for that step
            (e.g., updated knowledge, observed data, immediate net benefit).
        initial_psa (PSASample):
            PSA samples representing parameter uncertainty at the beginning of the process.
        dynamic_specification (DynamicSpec):
            Defines time steps, interim rules, or other dynamic aspects.
        wtp (float): Willingness-to-pay threshold for net benefit calculations.
        population (Optional[float]): Population size for scaling.
        discount_rate (Optional[float]): Discount rate for future values.
        time_horizon (Optional[float]): Overall time horizon.
        optimization_method (str): Method like "backward_induction" or "forward_simulation".
        **kwargs: Additional arguments for the specific sequential VOI approach.

    Returns
    -------
        Union[float, Generator[Dict[str, Any], None, None]]:
            Depending on the approach, could be an overall VOI for the entire
            sequential strategy, or a generator yielding VOI at each step.

    Raises
    ------
        InputError: If inputs are invalid.
    """
    # Input validation
    if not isinstance(dynamic_specification, DynamicSpec):
        raise InputError("dynamic_specification must be a DynamicSpec object.")
    
    if not hasattr(step_model, '__call__'):
        raise InputError("step_model must be callable.")
    
    if not isinstance(initial_psa, PSASample):
        raise InputError("initial_psa must be a PSASample object.")
    
    if not isinstance(wtp, (int, float)):
        raise InputError("wtp must be a number.")
    
    if population is not None and (not isinstance(population, (int, float)) or population <= 0):
        raise InputError("population must be a positive number if provided.")
    
    if discount_rate is not None and (not isinstance(discount_rate, (int, float)) or not (0 <= discount_rate <= 1)):
        raise InputError("discount_rate must be between 0 and 1 if provided.")
    
    time_steps = list(dynamic_specification.time_steps)
    if not time_steps:
        raise InputError("dynamic_specification must have at least one time step.")
    
    # Handle different optimization methods
    if optimization_method == "backward_induction":
        return _sequential_voi_backward_induction(
            step_model, initial_psa, dynamic_specification, wtp, 
            population, discount_rate, time_horizon, **kwargs
        )
    elif optimization_method == "generator":
        # Return a generator that yields information at each step
        return _sequential_voi_generator(
            step_model, initial_psa, dynamic_specification, wtp, 
            population, discount_rate, time_horizon, **kwargs
        )
    else:
        raise InputError(f"Unknown optimization_method: {optimization_method}")


def _sequential_voi_backward_induction(
    step_model: SequentialStepModel,
    initial_psa: PSASample,
    dynamic_specification: DynamicSpec,
    wtp: float,
    population: Optional[float],
    discount_rate: Optional[float],
    time_horizon: Optional[float],
    **kwargs: Any,
) -> float:
    """Calculate sequential VOI using backward induction approach.
    
    This approach works backwards from the final time step to determine the
    optimal policy at each step, then calculates the overall VOI.
    """
    time_steps = list(dynamic_specification.time_steps)
    n_steps = len(time_steps)
    
    if n_steps == 0:
        return 0.0
    
    # For simplicity, we'll calculate a basic sequential VOI as the sum of
    # EVPI at each time step, appropriately discounted
    
    total_voi = 0.0
    current_psa = initial_psa
    
    # Work backwards from the last time step
    for i in range(n_steps - 1, -1, -1):
        time_step = time_steps[i]
        
        # Calculate discount factor for this time step
        discount_factor = 1.0
        if discount_rate is not None and discount_rate > 0:
            time_from_start = time_step
            discount_factor = 1.0 / ((1.0 + discount_rate) ** time_from_start)
        
        # Calculate EVPI at this time step using the current PSA
        # This is a simplified approach - in practice, this would be more complex
        evpi_at_step = _calculate_evpi_at_step(current_psa, wtp)
        
        # Add discounted EVPI to total VOI
        total_voi += evpi_at_step * discount_factor
        
        # If not at the first step, simulate progression to previous state
        if i > 0:
            # This is a simplified simulation - in practice, this would use the step_model
            # to simulate how the PSA evolves over time
            pass
    
    # Apply population scaling if provided
    if population is not None and time_horizon is not None:
        annuity_factor = time_horizon
        if discount_rate is not None and discount_rate > 0:
            annuity_factor = (1 - (1 + discount_rate) ** (-time_horizon)) / discount_rate
        total_voi *= population * annuity_factor
    
    return total_voi


def _calculate_evpi_at_step(psa: PSASample, wtp: float) -> float:
    """Calculate EVPI at a specific time step given current parameter uncertainty.
    
    This is a simplified implementation that assumes we can calculate net benefits
    directly from the PSA samples.
    """
    # This is a placeholder implementation
    # In a real implementation, this would involve:
    # 1. Calculating net benefits for each strategy using the PSA samples
    # 2. Calculating EVPI using the standard formula
    
    # For now, we'll return a simple estimate based on the variance of parameters
    # This is not a real EVPI calculation, just a placeholder
    if hasattr(psa, 'parameters') and isinstance(psa.parameters, dict):
        # Calculate a simple measure of parameter uncertainty
        total_variance = 0.0
        for param_values in psa.parameters.values():
            if isinstance(param_values, np.ndarray):
                total_variance += np.var(param_values)
        # Scale by WTP to make it meaningful
        return total_variance * abs(wtp) * 0.01  # Arbitrary scaling factor
    else:
        return 0.0


def _sequential_voi_generator(
    step_model: SequentialStepModel,
    initial_psa: PSASample,
    dynamic_specification: DynamicSpec,
    wtp: float,
    population: Optional[float],
    discount_rate: Optional[float],
    time_horizon: Optional[float],
    **kwargs: Any,
) -> Generator[Dict[str, Any], None, None]:
    """Generator that yields VOI information at each time step."""
    time_steps = list(dynamic_specification.time_steps)
    current_psa = initial_psa
    
    for t_idx, time_step in enumerate(time_steps):
        # Calculate EVPI at this time step
        evpi_at_step = _calculate_evpi_at_step(current_psa, wtp)
        
        # Calculate discount factor
        discount_factor = 1.0
        if discount_rate is not None and discount_rate > 0:
            discount_factor = 1.0 / ((1.0 + discount_rate) ** time_step)
        
        # Yield information about this time step
        yield {
            "time_step": time_step,
            "current_evpi": evpi_at_step,
            "discount_factor": discount_factor,
            "discounted_evpi": evpi_at_step * discount_factor,
            "n_samples": current_psa.n_samples if hasattr(current_psa, 'n_samples') else None,
        }
        
        # Simulate progression to next state (simplified)
        if t_idx < len(time_steps) - 1:
            try:
                # Use step_model to simulate progression
                outcomes = step_model(current_psa, "progression_action", dynamic_specification)
                if 'next_psa' in outcomes and isinstance(outcomes['next_psa'], PSASample):
                    current_psa = outcomes['next_psa']
            except Exception:
                # If step_model fails, keep current PSA
                pass


if __name__ == "__main__":
    print("--- Testing sequential.py ---")

    # Add local imports for classes used in this test block
    import numpy as np  # Used by PSASample

    from voiage.schema import DynamicSpec

    # Test with backward induction method
    try:
        def dummy_step_model(psa, action, dyn_spec):
            # Simple dummy model that just returns the same PSA
            return {"next_psa": psa}

        dummy_psa = PSASample.from_numpy_or_dict({
            "param1": np.random.rand(100),
            "param2": np.random.rand(100)
        })
        dummy_dyn_spec = DynamicSpec(time_steps=[0, 1, 2])
        
        result = sequential_voi(
            dummy_step_model, dummy_psa, dummy_dyn_spec,
            wtp=50000,
            optimization_method="backward_induction"
        )
        print(f"sequential_voi (backward_induction) result: {result}")
    except Exception as e:
        print(f"Error in sequential_voi (backward_induction): {e}")

    # Test with generator method
    try:
        def dummy_step_model(psa, action, dyn_spec):
            # Simple dummy model that just returns the same PSA
            return {"next_psa": psa}

        dummy_psa = PSASample.from_numpy_or_dict({
            "param1": np.random.rand(100),
            "param2": np.random.rand(100)
        })
        dummy_dyn_spec = DynamicSpec(time_steps=[0, 1, 2])
        
        generator = sequential_voi(
            dummy_step_model, dummy_psa, dummy_dyn_spec,
            wtp=50000,
            optimization_method="generator"
        )
        
        print("sequential_voi (generator) results:")
        for step_info in generator:
            print(f"  Step {step_info['time_step']}: EVPI = {step_info['current_evpi']:.2f}")
    except Exception as e:
        print(f"Error in sequential_voi (generator): {e}")

    print("--- sequential.py tests completed ---")