"""Implementation-strategy comparison value of information."""

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from voiage.config import DEFAULT_DTYPE
from voiage.exceptions import raise_input_error


@dataclass(frozen=True)
class ImplementationStrategyComparisonResult:
    """Fixture-backed result envelope for implementation strategy comparison."""

    value: float
    implementation_value: float
    strategy_names: list[str]
    expected_net_benefits: np.ndarray
    optimal_strategy_by_period: dict[str, str]
    uptake_matrix: np.ndarray
    adherence_matrix: np.ndarray
    coverage_matrix: np.ndarray
    implementation_delay_matrix: np.ndarray
    scale_up_cost_matrix: np.ndarray
    population_impact_matrix: np.ndarray
    implementation_multiplier_matrix: np.ndarray
    adoption_uncertainty_matrix: np.ndarray
    population_impact_by_strategy: dict[str, float]
    method_maturity: str = "fixture-backed"
    diagnostics: dict[str, object] = field(default_factory=dict)
    reporting: dict[str, object] = field(default_factory=dict)


def value_of_implementation_strategy_comparison(
    net_benefits: np.ndarray,
    strategy_names: Sequence[str],
    uptake: np.ndarray,
    adherence: np.ndarray,
    coverage: np.ndarray,
    implementation_delays: np.ndarray,
    scale_up_costs: np.ndarray,
    population_impacts: np.ndarray,
    *,
    discount_rate: float = 0.0,
    analysis_id: str = "implementation-strategy-analysis",
    decision_problem_id: str = "unspecified",
) -> ImplementationStrategyComparisonResult:
    """Compare implementation strategies across an adoption horizon.

    ``net_benefits`` has shape ``(samples, strategies, periods)``. All
    strategy matrices have shape ``(periods, strategies)``.
    """
    values = np.asarray(net_benefits, dtype=DEFAULT_DTYPE)
    strategies = [str(item) for item in strategy_names]
    matrices = tuple(
        np.asarray(item, dtype=DEFAULT_DTYPE)
        for item in (
            uptake,
            adherence,
            coverage,
            implementation_delays,
            scale_up_costs,
            population_impacts,
        )
    )
    uptake_values, adherence_values, coverage_values, delays, costs, impacts = matrices
    if values.ndim != 3 or min(values.shape) < 1:
        raise_input_error("net_benefits must be a non-empty 3D array.")
    periods = values.shape[2]
    expected_shape = (periods, len(strategies))
    if any(matrix.shape != expected_shape for matrix in matrices):
        raise_input_error("Implementation matrices must have period x strategy shape.")
    if not strategies or len(set(strategies)) != len(strategies):
        raise_input_error("Strategy names must be non-empty and unique.")
    if not all(np.all(np.isfinite(matrix)) for matrix in (values, *matrices)):
        raise_input_error("Inputs must contain only finite values.")
    if not np.isfinite(discount_rate) or discount_rate < 0:
        raise_input_error("discount_rate must be finite and non-negative.")
    if (
        np.any(uptake_values < 0)
        or np.any(uptake_values > 1)
        or np.any(adherence_values < 0)
        or np.any(adherence_values > 1)
        or np.any(coverage_values < 0)
        or np.any(coverage_values > 1)
        or np.any(delays < 0)
        or np.any(costs < 0)
        or np.any(impacts < 0)
    ):
        raise_input_error(
            "Uptake, adherence, and coverage must be in [0, 1]; delays, costs, and impacts must be non-negative."
        )

    period_discount = (1.0 + float(discount_rate)) ** -np.arange(periods)
    multipliers = uptake_values * adherence_values * coverage_values
    adoption_uncertainty = 1.0 - multipliers
    raw = np.mean(values, axis=0).T
    adjusted = raw.T * multipliers * period_discount[:, None] + impacts - costs
    optimal = np.argmax(adjusted, axis=1)
    baseline = float(np.max(np.mean(raw, axis=1)))
    implemented = float(np.mean(np.max(adjusted, axis=1)))
    implementation_value = max(0.0, implemented - baseline)
    value = max(0.0, implemented - float(np.mean(adjusted[:, 0])))
    population_impact_by_strategy = {
        strategy: float(np.sum(impacts[:, index]))
        for index, strategy in enumerate(strategies)
    }
    return ImplementationStrategyComparisonResult(
        value=value,
        implementation_value=implementation_value,
        strategy_names=strategies,
        expected_net_benefits=adjusted,
        optimal_strategy_by_period={
            str(period): strategies[int(index)] for period, index in enumerate(optimal)
        },
        uptake_matrix=uptake_values,
        adherence_matrix=adherence_values,
        coverage_matrix=coverage_values,
        implementation_delay_matrix=delays,
        scale_up_cost_matrix=costs,
        population_impact_matrix=impacts,
        implementation_multiplier_matrix=multipliers,
        adoption_uncertainty_matrix=adoption_uncertainty,
        population_impact_by_strategy=population_impact_by_strategy,
        diagnostics={
            "analysis_id": analysis_id,
            "decision_problem_id": decision_problem_id,
            "n_periods": periods,
            "n_strategies": len(strategies),
            "baseline_expected_net_benefit": baseline,
            "implemented_expected_net_benefit": implemented,
            "mean_adoption_uncertainty": float(np.mean(adoption_uncertainty)),
            "mean_implementation_delay": float(np.mean(delays)),
            "mean_scale_up_cost": float(np.mean(costs)),
        },
        reporting={
            "reporting_standard": "CHEERS-VOI",
            "analysis_type": "value_of_implementation_strategy_comparison",
            "method_maturity": "fixture-backed",
        },
    )
