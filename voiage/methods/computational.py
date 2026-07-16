"""Computational-budget and model-refinement value of information."""

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from voiage.config import DEFAULT_DTYPE
from voiage.exceptions import raise_input_error


@dataclass(frozen=True)
class ComputationalResult:
    """Fixture-backed result envelope for computational refinement VOI."""

    value: float
    compute_value: float
    approximation_error_value: float
    refinement_value: float
    compute_budget_ids: list[str]
    strategy_names: list[str]
    expected_net_benefits: np.ndarray
    optimal_strategy_by_compute_budget: dict[str, str]
    compute_cost_matrix: np.ndarray
    approximation_error_matrix: np.ndarray
    refinement_weight_matrix: np.ndarray
    consensus_strategy: str
    robust_strategy: str
    pareto_strategies: list[str]
    reference_compute_budget: str
    method_maturity: str = "fixture-backed"
    diagnostics: dict[str, object] = field(default_factory=dict)
    reporting: dict[str, object] = field(default_factory=dict)


def value_of_computational_refinement(
    net_benefits: np.ndarray,
    compute_budget_ids: Sequence[str],
    strategy_names: Sequence[str],
    compute_costs: np.ndarray,
    approximation_errors: np.ndarray,
    refinement_weights: np.ndarray,
    *,
    analysis_id: str = "computational-analysis",
    decision_problem_id: str = "unspecified",
    reference_compute_budget: str | None = None,
) -> ComputationalResult:
    """Value model refinement across budgets, error, and compute cost.

    ``net_benefits`` has shape ``(samples, strategies, compute_budgets)``.
    """
    values = np.asarray(net_benefits, dtype=DEFAULT_DTYPE)
    budgets = [str(item) for item in compute_budget_ids]
    strategies = [str(item) for item in strategy_names]
    costs = np.asarray(compute_costs, dtype=DEFAULT_DTYPE)
    errors = np.asarray(approximation_errors, dtype=DEFAULT_DTYPE)
    weights = np.asarray(refinement_weights, dtype=DEFAULT_DTYPE)
    if values.ndim != 3 or min(values.shape) < 1:
        raise_input_error(
            "net_benefits must be a non-empty 3D array (samples, strategies, budgets)."
        )
    if values.shape[1:] != (len(strategies), len(budgets)):
        raise_input_error(
            "net_benefits dimensions must match strategy and budget names."
        )
    expected_shape = (len(budgets), len(strategies))
    if any(matrix.shape != expected_shape for matrix in (costs, errors, weights)):
        raise_input_error("Computational matrices must have budget x strategy shape.")
    if len(set(budgets)) != len(budgets) or len(set(strategies)) != len(strategies):
        raise_input_error("Budget and strategy names must be unique.")
    if not all(
        np.all(np.isfinite(matrix)) for matrix in (values, costs, errors, weights)
    ):
        raise_input_error("Inputs must contain only finite values.")
    if (
        np.any(costs < 0)
        or np.any(errors < 0)
        or np.any(weights < 0)
        or np.any(weights > 1)
    ):
        raise_input_error(
            "Costs and approximation errors must be non-negative; refinement weights must be in [0, 1]."
        )
    reference_compute_budget = reference_compute_budget or budgets[0]
    if reference_compute_budget not in budgets:
        raise_input_error("reference_compute_budget must identify a compute budget.")

    raw = np.mean(values, axis=0).T
    adjusted = raw * weights - costs - errors
    optimal = np.argmax(adjusted, axis=1)
    reference_index = budgets.index(reference_compute_budget)
    reference_strategy = int(optimal[reference_index])
    reference_value = float(adjusted[reference_index, reference_strategy])
    flexible_value = float(np.mean(np.max(adjusted, axis=1)))
    value = max(0.0, flexible_value - reference_value)
    compute_value = max(0.0, float(np.mean(raw)) - float(np.mean(raw - costs)))
    error_value = max(
        0.0, float(np.mean(raw - costs)) - float(np.mean(raw - costs - errors))
    )
    refinement_value = max(0.0, float(np.mean(raw)) - float(np.mean(raw * weights)))
    pareto = [
        strategies[i]
        for i in range(len(strategies))
        if not any(
            i != j
            and np.all(adjusted[:, j] >= adjusted[:, i])
            and np.any(adjusted[:, j] > adjusted[:, i])
            for j in range(len(strategies))
        )
    ]
    consensus = strategies[int(np.argmax(np.mean(adjusted, axis=0)))]
    robust = strategies[int(np.argmax(np.min(adjusted, axis=0)))]
    return ComputationalResult(
        value=value,
        compute_value=compute_value,
        approximation_error_value=error_value,
        refinement_value=refinement_value,
        compute_budget_ids=budgets,
        strategy_names=strategies,
        expected_net_benefits=adjusted,
        optimal_strategy_by_compute_budget={
            budget: strategies[int(index)]
            for budget, index in zip(budgets, optimal, strict=True)
        },
        compute_cost_matrix=costs,
        approximation_error_matrix=errors,
        refinement_weight_matrix=weights,
        consensus_strategy=consensus,
        robust_strategy=robust,
        pareto_strategies=pareto,
        reference_compute_budget=reference_compute_budget,
        diagnostics={
            "analysis_id": analysis_id,
            "decision_problem_id": decision_problem_id,
            "n_budgets": len(budgets),
            "n_strategies": len(strategies),
        },
        reporting={
            "reporting_standard": "CHEERS-VOI",
            "analysis_type": "value_of_computational_refinement",
            "method_maturity": "fixture-backed",
        },
    )
