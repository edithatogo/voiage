"""Value of resolving equity-relevant information uncertainty."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from voiage.config import DEFAULT_DTYPE
from voiage.exceptions import raise_input_error
from voiage.reporting import build_cheers_reporting
from voiage.schema import ValueArray


@dataclass(frozen=True)
class EquityInformationResult:
    """Structured result for equity-information VOI."""

    value: float
    baseline_optimal_strategy_index: int
    baseline_optimal_strategy_name: str
    baseline_social_welfare: float
    resolved_optimal_strategy_indices: np.ndarray
    resolved_optimal_strategy_names: list[str]
    resolved_social_welfare: np.ndarray
    equity_weights: np.ndarray
    subgroup_labels: list[str]
    subgroup_expected_net_benefits: np.ndarray
    scenario_probabilities: np.ndarray
    policy_strata: list[str]
    information_cost: float
    method_maturity: str
    diagnostics: dict[str, object]
    reporting: dict[str, object]


def _validate_inputs(
    value_array: ValueArray,
    subgroups: np.ndarray | list[Any],
    equity_weights: np.ndarray | list[float],
    resolved_equity_weights: np.ndarray,
    scenario_probabilities: np.ndarray | list[float] | None,
    strategy_names: list[str] | None,
    policy_strata: list[str] | None,
) -> tuple[
    np.ndarray, list[str], np.ndarray, np.ndarray, np.ndarray, list[str], list[str]
]:
    """Validate and normalize the equity-information contract."""
    if not isinstance(value_array, ValueArray):
        raise_input_error("`value_array` must be a ValueArray object.")
    values = np.asarray(value_array.numpy_values, dtype=DEFAULT_DTYPE)
    if values.ndim != 2 or min(values.shape) < 1:
        raise_input_error("Equity information VOI requires 2D net benefits.")
    if not np.all(np.isfinite(values)):
        raise_input_error("Net-benefit values must be finite.")

    subgroup_values = np.asarray(subgroups)
    if subgroup_values.ndim != 1 or len(subgroup_values) != values.shape[0]:
        raise_input_error("`subgroups` must match the number of samples.")
    labels = [str(label) for label in np.unique(subgroup_values)]
    subgroup_means = np.asarray(
        [np.mean(values[subgroup_values == label], axis=0) for label in labels],
        dtype=DEFAULT_DTYPE,
    )

    weights = np.asarray(equity_weights, dtype=DEFAULT_DTYPE)
    if weights.ndim != 1 or len(weights) != len(labels):
        raise_input_error("`equity_weights` must contain one value per subgroup.")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0) or np.sum(weights) <= 0:
        raise_input_error(
            "`equity_weights` must be finite, non-negative, and positive."
        )
    weights = weights / np.sum(weights)

    scenarios = np.asarray(resolved_equity_weights, dtype=DEFAULT_DTYPE)
    if (
        scenarios.ndim != 2
        or scenarios.shape[1] != len(labels)
        or scenarios.shape[0] < 1
    ):
        raise_input_error("`resolved_equity_weights` must be scenario x subgroup.")
    if not np.all(np.isfinite(scenarios)) or np.any(scenarios < 0):
        raise_input_error("Resolved equity weights must be finite and non-negative.")
    scenario_totals = np.sum(scenarios, axis=1)
    if np.any(scenario_totals <= 0):
        raise_input_error("Each resolved equity-weight scenario must be positive.")
    scenarios = scenarios / scenario_totals[:, None]

    probabilities = np.ones(scenarios.shape[0], dtype=DEFAULT_DTYPE)
    if scenario_probabilities is not None:
        probabilities = np.asarray(scenario_probabilities, dtype=DEFAULT_DTYPE)
        if probabilities.ndim != 1 or len(probabilities) != scenarios.shape[0]:
            raise_input_error("scenario probabilities must match scenario count.")
        if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0):
            raise_input_error("Scenario probabilities must be finite and non-negative.")
        if np.sum(probabilities) <= 0:
            raise_input_error("Scenario probabilities must sum to a positive value.")
        probabilities = probabilities / np.sum(probabilities)

    names = strategy_names or value_array.strategy_names
    if len(names) != values.shape[1]:
        raise_input_error("`strategy_names` length must match strategy count.")
    strata = labels if policy_strata is None else policy_strata
    if not strata:
        raise_input_error("At least one policy-relevant stratum is required.")
    return (
        values,
        labels,
        subgroup_means,
        weights,
        scenarios,
        probabilities,
        names,
        strata,
    )


def value_of_equity_information(
    value_array: ValueArray,
    subgroups: np.ndarray | list[Any],
    equity_weights: np.ndarray | list[float],
    resolved_equity_weights: np.ndarray | list[list[float]],
    scenario_probabilities: np.ndarray | list[float] | None = None,
    information_cost: float = 0.0,
    strategy_names: list[str] | None = None,
    policy_strata: list[str] | None = None,
) -> EquityInformationResult:
    r"""Calculate the value of resolving uncertainty in equity weights.

    ``resolved_equity_weights`` describes plausible equity-weight scenarios
    after subgroup data acquisition. The result is the expected gain from
    choosing the social-welfare-optimal strategy after resolution, less the
    information cost. This is intentionally fixture-backed until parity and
    real-data attribution gates are complete.
    """
    if not np.isfinite(information_cost) or information_cost < 0:
        raise_input_error("`information_cost` must be finite and non-negative.")
    (
        values,
        labels,
        subgroup_means,
        weights,
        scenarios,
        probabilities,
        names,
        strata,
    ) = _validate_inputs(
        value_array,
        subgroups,
        equity_weights,
        np.asarray(resolved_equity_weights, dtype=DEFAULT_DTYPE),
        scenario_probabilities,
        strategy_names,
        policy_strata,
    )
    baseline_values = weights @ subgroup_means
    baseline_idx = int(np.argmax(baseline_values))
    resolved_values = scenarios @ subgroup_means
    resolved_indices = np.argmax(resolved_values, axis=1).astype(int)
    resolved_welfare = resolved_values[
        np.arange(len(resolved_indices)), resolved_indices
    ]
    expected_resolved = float(probabilities @ resolved_welfare)
    baseline_welfare = float(baseline_values[baseline_idx])
    value = max(0.0, expected_resolved - baseline_welfare - information_cost)
    allocation = np.mean(np.abs(scenarios - weights[None, :]), axis=0)
    if np.sum(allocation) > 0:
        allocation = allocation / np.sum(allocation)

    diagnostics: dict[str, object] = {
        "n_samples": int(values.shape[0]),
        "n_strategies": int(values.shape[1]),
        "n_subgroups": len(labels),
        "n_equity_scenarios": int(scenarios.shape[0]),
        "subgroup_sample_allocation": allocation.tolist(),
        "equity_metric": "weighted social welfare over subgroup expected net benefits",
        "uncertainty_scope": "equity weights beyond subgroup heterogeneity",
        "parity_status": "deferred",
        "open_data_status": "blocked: no licensed individual-level equity snapshot committed",
        "policy_strata": strata,
    }
    return EquityInformationResult(
        value=value,
        baseline_optimal_strategy_index=baseline_idx,
        baseline_optimal_strategy_name=names[baseline_idx],
        baseline_social_welfare=baseline_welfare,
        resolved_optimal_strategy_indices=resolved_indices,
        resolved_optimal_strategy_names=[names[int(idx)] for idx in resolved_indices],
        resolved_social_welfare=resolved_welfare,
        equity_weights=weights,
        subgroup_labels=labels,
        subgroup_expected_net_benefits=subgroup_means,
        scenario_probabilities=probabilities,
        policy_strata=strata,
        information_cost=float(information_cost),
        method_maturity="fixture-backed",
        diagnostics=diagnostics,
        reporting=build_cheers_reporting(
            analysis_type="value_of_equity_information",
            method_family="value_of_equity_information",
            method_maturity="fixture-backed",
            diagnostics={
                "n_samples": int(values.shape[0]),
                "n_strategies": int(values.shape[1]),
                "n_subgroups": len(labels),
                "n_equity_scenarios": int(scenarios.shape[0]),
            },
        ),
    )
