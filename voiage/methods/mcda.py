"""Multi-Criteria Decision Analysis value of information."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from voiage.config import DEFAULT_DTYPE
from voiage.exceptions import raise_input_error


@dataclass(frozen=True)
class MCDAValueOfInformationResult:
    """Value of resolving uncertainty about MCDA criterion weights."""

    value: float
    baseline_value: float
    flexible_value: float
    baseline_strategy_index: int
    scenario_optimal_strategy_indices: np.ndarray


def value_of_mcda_information(
    criterion_scores: np.ndarray, weight_scenarios: np.ndarray
) -> MCDAValueOfInformationResult:
    """Calculate MCDA-VOI for strategy scores and alternative weight vectors."""
    scores = np.asarray(criterion_scores, dtype=DEFAULT_DTYPE)
    weights = np.asarray(weight_scenarios, dtype=DEFAULT_DTYPE)
    if scores.ndim != 2 or weights.ndim != 2 or scores.shape[1] != weights.shape[1]:
        raise_input_error("Scores and weight scenarios must be 2D with matching criteria.")
    if min(scores.shape) < 1 or weights.shape[0] < 1 or not np.all(np.isfinite(scores)) or not np.all(np.isfinite(weights)):
        raise_input_error("Scores and weights must be finite and non-empty.")
    if np.any(weights < 0) or np.any(weights.sum(axis=1) <= 0):
        raise_input_error("Each MCDA weight scenario must be non-negative and positive-sum.")
    weights = weights / weights.sum(axis=1, keepdims=True)
    scenario_values = weights @ scores.T
    baseline_scores = scenario_values.mean(axis=0)
    baseline_index = int(np.argmax(baseline_scores))
    flexible = float(np.mean(np.max(scenario_values, axis=1)))
    baseline = float(baseline_scores[baseline_index])
    return MCDAValueOfInformationResult(max(0.0, flexible - baseline), baseline, flexible, baseline_index, np.argmax(scenario_values, axis=1))
