"""Fixture-backed VOI for decisions with budget and capacity constraints."""

from dataclasses import dataclass

import numpy as np

from voiage.config import DEFAULT_DTYPE
from voiage.exceptions import raise_input_error
from voiage.reporting import build_cheers_reporting


@dataclass(frozen=True)
class CapacityBudgetConstrainedResult:
    """Result envelope for constrained decision value."""

    value: float
    selected_strategy: str
    strategy_names: list[str]
    expected_values: np.ndarray
    scenario_optimal_strategies: list[str]
    budget: float
    capacity: float
    budget_impact: float
    capacity_shortfall: float
    constrained_regret: float
    opportunity_cost: float
    shadow_price_budget: float
    shadow_price_capacity: float
    method_maturity: str
    diagnostics: dict[str, object]
    reporting: dict[str, object]


def value_of_capacity_budget_constrained(
    scenario_values: np.ndarray | list[list[float]],
    *,
    strategy_costs: list[float] | np.ndarray,
    strategy_capacity: list[float] | np.ndarray,
    budget: float,
    capacity: float,
    strategy_names: list[str] | None = None,
    information_cost: float = 0.0,
) -> CapacityBudgetConstrainedResult:
    """Estimate VOI when information changes the best feasible strategy.

    ``scenario_values`` is a scenario-by-strategy matrix. The baseline uses
    mean scenario value; perfect information chooses the best feasible
    strategy separately in each scenario. Inputs are deterministic and the
    result is explicitly fixture-backed pending external data and parity.
    """
    values = np.asarray(scenario_values, dtype=DEFAULT_DTYPE)
    costs = np.asarray(strategy_costs, dtype=DEFAULT_DTYPE)
    loads = np.asarray(strategy_capacity, dtype=DEFAULT_DTYPE)
    if values.ndim != 2 or min(values.shape) < 1 or not np.all(np.isfinite(values)):
        raise_input_error("scenario_values must be a finite scenario x strategy matrix.")
    if len(costs) != values.shape[1] or len(loads) != values.shape[1]:
        raise_input_error("strategy constraints must match the strategy count.")
    if np.any(costs < 0) or np.any(loads < 0):
        raise_input_error("strategy costs and capacity loads must be non-negative.")
    if not np.isfinite(budget) or budget < 0 or not np.isfinite(capacity) or capacity < 0:
        raise_input_error("budget and capacity must be finite and non-negative.")
    if not np.isfinite(information_cost) or information_cost < 0:
        raise_input_error("information_cost must be finite and non-negative.")
    names = strategy_names or [f"strategy_{i + 1}" for i in range(values.shape[1])]
    if len(names) != values.shape[1]:
        raise_input_error("strategy_names length must match the strategy count.")
    feasible = (costs <= budget) & (loads <= capacity)
    if not np.any(feasible):
        raise_input_error("at least one strategy must satisfy budget and capacity constraints.")
    expected = np.mean(values, axis=0)
    constrained_expected = np.where(feasible, expected, -np.inf)
    baseline_index = int(np.argmax(constrained_expected))
    scenario_indices = [int(np.argmax(np.where(feasible, row, -np.inf))) for row in values]
    perfect_value = float(np.mean([values[i, j] for i, j in enumerate(scenario_indices)]))
    baseline_value = float(expected[baseline_index])
    value = max(0.0, perfect_value - baseline_value - information_cost)
    budget_impact = float(np.max(np.where(feasible, costs, 0.0)) - costs[baseline_index])
    capacity_shortfall = float(max(0.0, np.max(loads) - capacity))
    regret = max(0.0, perfect_value - baseline_value)
    shadow_budget = float(max(0.0, perfect_value - baseline_value) / max(budget, 1.0))
    shadow_capacity = float(max(0.0, perfect_value - baseline_value) / max(capacity, 1.0))
    diagnostics: dict[str, object] = {
        "feasible_strategy_count": int(np.sum(feasible)),
        "scenario_count": int(values.shape[0]),
        "information_cost": information_cost,
        "parity_status": "deferred",
        "open_data_status": "blocked: no licensed constrained-allocation trace committed",
        "baseline_value": baseline_value,
        "perfect_information_value": perfect_value,
    }
    return CapacityBudgetConstrainedResult(
        value=value,
        selected_strategy=str(names[baseline_index]),
        strategy_names=list(names),
        expected_values=expected,
        scenario_optimal_strategies=[str(names[i]) for i in scenario_indices],
        budget=float(budget),
        capacity=float(capacity),
        budget_impact=budget_impact,
        capacity_shortfall=capacity_shortfall,
        constrained_regret=regret,
        opportunity_cost=regret,
        shadow_price_budget=shadow_budget,
        shadow_price_capacity=shadow_capacity,
        method_maturity="fixture-backed",
        diagnostics=diagnostics,
        reporting=build_cheers_reporting(
            analysis_type="value_of_capacity_budget_constrained",
            method_family="capacity_budget_constrained",
            method_maturity="fixture-backed",
            diagnostics=diagnostics,
        ),
    )
