# voiage/methods/sequential.py

"""Implementation of VOI methods for dynamic or sequential decision problems.

- Dynamic / Sequential VOI

These methods assess the value of information in contexts where decisions are
made sequentially over time, and information gathered at one stage can influence
future decisions and future information gathering opportunities.
"""

from typing import Any, Callable, Dict, Generator, Union

from voiage.schema import DynamicSpec, ParameterSet
from voiage.exceptions import VoiageNotImplementedError

# Type alias for a function that models one step in a sequential process.
# It might take current state (including parameter beliefs), an action/decision,
# and return new state, accrued data, and immediate rewards/costs.
SequentialStepModel = Callable[
    [ParameterSet, Any, DynamicSpec],  # Current PSA, Action/Decision, Dynamic settings
    Dict[
        str, Any
    ],  # e.g., {'next_psa': ParameterSet, 'observed_data': Any, 'immediate_nb': float}
]


def sequential_voi(
    step_model: SequentialStepModel,
    initial_psa: ParameterSet,
    dynamic_specification: DynamicSpec,
    **kwargs: Any,
) -> Union[float, Generator[Dict[str, Any], None, None]]:
    raise VoiageNotImplementedError(
        "Dynamic / Sequential VOI is a highly advanced topic, often requiring "
        "custom modeling (e.g., Markov Decision Processes, Reinforcement Learning approaches, "
        "or bespoke simulations). Not implemented in v0.1.",
    )

    # Conceptual flow for a generator-based API (simplified):
    # current_psa_state = initial_psa
    # for t_idx, time_step in enumerate(dynamic_specification.time_steps):
    #     # 1. At this time_step, what is the value of immediate perfect info (EVPI_t)?
    #     #    This would use current_psa_state.
    #     evpi_at_step_t = calculate_evpi_at_step(current_psa_state, step_model, time_step, **kwargs)
    #
    #     # 2. What is the value of sample info for a study conducted *before* next decision (EVSI_t)?
    #     #    This would be more complex, involving simulating that study.
    #     evsi_at_step_t = calculate_evsi_at_step(current_psa_state, step_model, time_step, **kwargs)
    #
    #     yield {
    #         "time_step": time_step,
    #         "current_evpi": evpi_at_step_t,
    #         "next_study_evsi": evsi_at_step_t,
    #         # other relevant metrics
    #     }
    #
    #     # 3. Simulate progression to the next state (if not final step)
    #     # This might involve assuming an optimal action is taken, or simulating data accrual.
    #     if t_idx < len(dynamic_specification.time_steps) - 1:
    #         # outcomes = step_model(current_psa_state, "optimal_action_or_data_accrual", dynamic_specification)
    #         # current_psa_state = outcomes['next_psa']
    #         pass # Placeholder for state transition logic

