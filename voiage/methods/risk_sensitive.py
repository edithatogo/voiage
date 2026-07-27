"""Risk-sensitive and constrained value of information calculations."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from voiage.config import DEFAULT_DTYPE
from voiage.exceptions import raise_input_error


@dataclass(frozen=True)
class RiskSensitiveVOIResult:
    """Risk-adjusted fixed-policy and flexible-policy values."""

    value: float
    constrained_value: float
    flexible_value: float
    constrained_strategy_index: int
    scenario_optimal_strategy_indices: np.ndarray


def value_of_risk_sensitive_information(
    scenario_net_benefits: np.ndarray,
    risk_aversion: float = 0.0,
    lower_tail_fraction: float = 0.2,
) -> RiskSensitiveVOIResult:
    """Value information under mean-minus-tail-risk utility."""
    values = np.asarray(scenario_net_benefits, dtype=DEFAULT_DTYPE)
    if values.ndim != 2 or min(values.shape) < 1 or not np.all(np.isfinite(values)):
        raise_input_error("scenario_net_benefits must be a finite 2D array.")
    if risk_aversion < 0 or not 0 < lower_tail_fraction <= 1:
        raise_input_error("risk_aversion must be non-negative and tail fraction in (0, 1].")
    tail_count = max(1, int(np.ceil(values.shape[0] * lower_tail_fraction)))
    tail_mean = np.sort(values, axis=0)[:tail_count].mean(axis=0)
    utility = values.mean(axis=0) - risk_aversion * (values.mean(axis=0) - tail_mean)
    constrained_index = int(np.argmax(utility))
    constrained = float(utility[constrained_index])
    scenario_utility = values - risk_aversion * (values - tail_mean[None, :])
    flexible = float(np.mean(np.max(scenario_utility, axis=1)))
    return RiskSensitiveVOIResult(max(0.0, flexible - constrained), constrained, flexible, constrained_index, np.argmax(scenario_utility, axis=1))
