# voiage/methods/sequential.py

"""Value of information methods for dynamic and sequential decision problems."""

from collections.abc import Callable, Generator
from typing import cast

import numpy as np

from voiage.exceptions import raise_input_error
from voiage.schema import DynamicSpec
from voiage.schema import ParameterSet as PSASample

# Type alias for a function that models one step in a sequential process.
# It might take current state (including parameter beliefs), an action/decision,
# and return new state, accrued data, and immediate rewards/costs.
SequentialStepModel = Callable[
    [PSASample, object, DynamicSpec],  # Current PSA, Action/Decision, Dynamic settings
    dict[
        str, object
    ],  # e.g., {'next_psa': PSASample, 'observed_data': object, 'immediate_nb': float}
]


def _advance_psa(
    step_model: SequentialStepModel,
    current_psa: PSASample,
    dynamic_specification: DynamicSpec,
) -> PSASample:
    """Advance the PSA state when the step model yields a valid next state."""
    outcomes = step_model(current_psa, "progression_action", dynamic_specification)
    if isinstance(outcomes, dict):
        next_psa = outcomes.get("next_psa")
        if isinstance(next_psa, PSASample):
            return next_psa
    return current_psa


def sequential_voi(
    step_model: SequentialStepModel,
    initial_psa: PSASample,
    dynamic_specification: DynamicSpec,
    wtp: float = 0.0,
    population: float | None = None,
    discount_rate: float | None = None,
    time_horizon: float | None = None,
    optimization_method: str = "backward_induction",
    **kwargs: object,
) -> float | Generator[dict[str, object], None, None]:
    """Calculate value of information for a sequential decision problem."""
    # Input validation
    if not isinstance(dynamic_specification, DynamicSpec):
        raise_input_error("dynamic_specification must be a DynamicSpec object.")

    if not callable(step_model):
        raise_input_error("step_model must be callable.")

    if not isinstance(initial_psa, PSASample):
        raise_input_error("initial_psa must be a PSASample object.")

    if not isinstance(wtp, (int, float)):
        raise_input_error("wtp must be a number.")

    if population is not None and (
        not isinstance(population, (int, float)) or population <= 0
    ):
        raise_input_error("population must be a positive number if provided.")

    if discount_rate is not None and (
        not isinstance(discount_rate, (int, float)) or not (0 <= discount_rate <= 1)
    ):
        raise_input_error("discount_rate must be between 0 and 1 if provided.")

    time_steps = list(dynamic_specification.time_steps)
    if not time_steps:
        raise_input_error("dynamic_specification must have at least one time step.")

    # Handle different optimization methods
    if optimization_method == "backward_induction":
        return _sequential_voi_backward_induction(
            step_model,
            initial_psa,
            dynamic_specification,
            wtp,
            population,
            discount_rate,
            time_horizon,
            **kwargs,
        )
    if optimization_method == "generator":
        # Return a generator that yields information at each step
        return _sequential_voi_generator(
            step_model,
            initial_psa,
            dynamic_specification,
            wtp,
            population,
            discount_rate,
            time_horizon,
            **kwargs,
        )
    return raise_input_error(f"Unknown optimization_method: {optimization_method}")


def _sequential_voi_backward_induction(
    step_model: SequentialStepModel,
    initial_psa: PSASample,
    dynamic_specification: DynamicSpec,
    wtp: float,
    population: float | None,
    discount_rate: float | None,
    time_horizon: float | None,
    **kwargs: object,
) -> float:
    """Calculate sequential VOI using backward induction approach.

    This approach works backwards from the final time step to determine the
    optimal policy at each step, then calculates the overall VOI.
    """
    time_steps = list(dynamic_specification.time_steps)
    n_steps = len(time_steps)

    if n_steps == 0:
        return 0.0

    total_voi = 0.0
    current_psa = initial_psa

    # Evaluate the decision horizon in chronological order so each step can
    # update the PSA state before the next time point is evaluated.
    for time_step in time_steps:
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

        # Simulate the next state when the model provides a valid transition.
        current_psa = _advance_psa(step_model, current_psa, dynamic_specification)

    # Apply population scaling if provided
    if population is not None and time_horizon is not None:
        annuity_factor = time_horizon
        if discount_rate is not None and discount_rate > 0:
            annuity_factor = (
                1 - (1 + discount_rate) ** (-time_horizon)
            ) / discount_rate
        total_voi *= population * annuity_factor

    return total_voi


def _calculate_evpi_at_step(psa: PSASample, wtp: float) -> float:
    """Calculate EVPI at a specific time step given current parameter uncertainty.

    The calculation uses the standard EVPI formula:
    E[max_d NB_d(theta)] - max_d E[NB_d(theta)].

    The PSA must include strategy payoff samples. Supported representations are:
    a 2D ``net_benefits`` variable, multiple ``net_benefit_*``/``nb_*``
    variables, or matched ``effect_*``/``cost_*`` strategy pairs. If no payoff
    surface is present, EVPI is not identifiable from parameter uncertainty
    alone and the function returns zero.
    """
    if not hasattr(psa, "parameters") or not isinstance(psa.parameters, dict):
        return 0.0

    net_benefits = _extract_step_net_benefits(psa.parameters, wtp)
    if net_benefits is None:
        return 0.0

    if net_benefits.ndim != 2 or net_benefits.shape[1] <= 1:
        return 0.0

    expected_max_net_benefit = np.mean(np.max(net_benefits, axis=1))
    max_expected_net_benefit = np.max(np.mean(net_benefits, axis=0))
    evpi = expected_max_net_benefit - max_expected_net_benefit
    return float(max(0.0, evpi))


def _extract_step_net_benefits(
    parameters: dict[str, np.ndarray], wtp: float
) -> np.ndarray | None:
    """Extract strategy net-benefit samples from supported PSA representations."""
    direct_net_benefits = _direct_net_benefits(parameters)
    if direct_net_benefits is not None:
        return direct_net_benefits

    named_net_benefits = _stack_prefixed_strategy_arrays(
        parameters, prefixes=("net_benefit_", "nb_")
    )
    if named_net_benefits is not None:
        return named_net_benefits

    return _cost_effect_net_benefits(parameters, wtp)


def _direct_net_benefits(parameters: dict[str, np.ndarray]) -> np.ndarray | None:
    """Return a 2D net-benefit matrix from a direct ``net_benefits`` variable."""
    if "net_benefits" not in parameters:
        return None

    values = np.asarray(parameters["net_benefits"], dtype=float)
    if values.ndim == 1:
        return values.reshape(-1, 1)
    if values.ndim == 2:
        return values
    return None


def _stack_prefixed_strategy_arrays(
    parameters: dict[str, np.ndarray], prefixes: tuple[str, ...]
) -> np.ndarray | None:
    """Stack strategy arrays that use one of the provided name prefixes."""
    strategy_arrays = [
        np.asarray(values, dtype=float)
        for name, values in sorted(parameters.items())
        if any(name.startswith(prefix) for prefix in prefixes)
    ]
    if not strategy_arrays:
        return None
    if any(values.ndim != 1 for values in strategy_arrays):
        return None
    return np.column_stack(strategy_arrays)


def _cost_effect_net_benefits(
    parameters: dict[str, np.ndarray], wtp: float
) -> np.ndarray | None:
    """Derive net benefits from matched strategy cost and effect samples."""
    effect_prefixes = ("effect_", "effectiveness_", "qaly_", "qalys_")
    cost_prefixes = ("cost_", "costs_")
    effect_by_strategy: dict[str, np.ndarray] = {}
    cost_by_strategy: dict[str, np.ndarray] = {}

    for name, values in parameters.items():
        for prefix in effect_prefixes:
            if name.startswith(prefix):
                effect_by_strategy[name.removeprefix(prefix)] = np.asarray(
                    values, dtype=float
                )
                break
        for prefix in cost_prefixes:
            if name.startswith(prefix):
                cost_by_strategy[name.removeprefix(prefix)] = np.asarray(
                    values, dtype=float
                )
                break

    strategy_names = sorted(set(effect_by_strategy) & set(cost_by_strategy))
    if not strategy_names:
        return None

    net_benefit_arrays = []
    for strategy_name in strategy_names:
        effect = effect_by_strategy[strategy_name]
        cost = cost_by_strategy[strategy_name]
        if effect.ndim != 1 or cost.ndim != 1 or len(effect) != len(cost):
            return None
        net_benefit_arrays.append(wtp * effect - cost)

    return np.column_stack(net_benefit_arrays)


def _sequential_voi_generator(
    step_model: SequentialStepModel,
    initial_psa: PSASample,
    dynamic_specification: DynamicSpec,
    wtp: float,
    population: float | None,
    discount_rate: float | None,
    time_horizon: float | None,
    **kwargs: object,
) -> Generator[dict[str, object], None, None]:
    """Yield VOI information at each time step."""
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
            "n_samples": current_psa.n_samples
            if hasattr(current_psa, "n_samples")
            else None,
        }

        # Simulate progression to next state (simplified)
        if t_idx < len(time_steps) - 1:
            current_psa = _advance_psa(step_model, current_psa, dynamic_specification)


if __name__ == "__main__":  # pragma: no cover
    print("--- Testing sequential.py ---")

    # Add local imports for classes used in this test block
    import numpy as np  # Used by PSASample

    from voiage.schema import DynamicSpec

    # Test with backward induction method
    def dummy_step_model(
        psa: object, action: object, dyn_spec: object
    ) -> dict[str, object]:
        """Return the input PSA unchanged for smoke testing."""
        return {"next_psa": psa}

    dummy_psa = PSASample.from_numpy_or_dict(
        {"param1": np.random.rand(100), "param2": np.random.rand(100)}
    )
    dummy_dyn_spec = DynamicSpec(time_steps=[0, 1, 2])

    result = sequential_voi(
        dummy_step_model,
        dummy_psa,
        dummy_dyn_spec,
        wtp=50000,
        optimization_method="backward_induction",
    )
    print(f"sequential_voi (backward_induction) result: {result}")

    # Test with generator method
    def dummy_step_model(
        psa: object, action: object, dyn_spec: object
    ) -> dict[str, object]:
        """Return the input PSA unchanged for smoke testing."""
        return {"next_psa": psa}

    dummy_psa = PSASample.from_numpy_or_dict(
        {"param1": np.random.rand(100), "param2": np.random.rand(100)}
    )
    dummy_dyn_spec = DynamicSpec(time_steps=[0, 1, 2])

    generator = cast(
        "Generator[dict[str, object], None, None]",
        sequential_voi(
            dummy_step_model,
            dummy_psa,
            dummy_dyn_spec,
            wtp=50000,
            optimization_method="generator",
        ),
    )

    print("sequential_voi (generator) results:")
    for step_info in generator:
        print(
            f"  Step {step_info['time_step']}: EVPI = {step_info['current_evpi']:.2f}"
        )

    print("--- sequential.py tests completed ---")
