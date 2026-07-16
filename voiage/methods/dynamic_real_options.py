"""Dynamic real-options value of information analysis."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from voiage.config import DEFAULT_DTYPE
from voiage.exceptions import raise_input_error


@dataclass(frozen=True)
class DynamicRealOptionsResult:
    """Result envelope for staged, irreversible decision analysis."""

    expected_net_benefits: np.ndarray
    decision_stage_names: list[str]
    strategy_names: list[str]
    optimal_strategy_names: list[str]
    waiting_value: float
    option_value: float
    policy_path_regret: np.ndarray
    timing_sensitivity: np.ndarray
    robust_strategy_name: str
    pareto_strategy_names: list[str]
    method_maturity: str
    diagnostics: dict[str, object] = field(default_factory=dict)
    reporting: dict[str, object] = field(default_factory=dict)


def _pareto_strategies(values: np.ndarray, names: Sequence[str]) -> list[str]:
    """Return strategies not weakly dominated across decision stages."""
    keep: list[str] = []
    for index, name in enumerate(names):
        dominated = False
        for other in range(values.shape[0]):
            if other == index:
                continue
            if np.all(values[other] >= values[index]) and np.any(
                values[other] > values[index]
            ):
                dominated = True
                break
        if not dominated:
            keep.append(str(name))
    return keep


def value_of_dynamic_real_options(
    net_benefits: np.ndarray,
    decision_stage_names: Sequence[str],
    strategy_names: Sequence[str],
    stage_weights: Mapping[str, float] | None = None,
    discount_rate: float = 0.0,
    irreversibility_penalty: float = 0.0,
    lock_in_penalty: float = 0.0,
    evidence_arrival_times: Mapping[str, float] | None = None,
    exercise_rules: Mapping[str, str] | None = None,
) -> DynamicRealOptionsResult:
    """Value staged evidence when decisions are delayed or irreversible.

    ``net_benefits`` has shape ``(samples, strategies, decision_stages)``.
    Stage weights represent uncertainty over when evidence becomes available;
    discounting and the two penalties are applied to later-stage values.
    """
    values = np.asarray(net_benefits, dtype=DEFAULT_DTYPE)
    stages = [str(item) for item in decision_stage_names]
    strategies = [str(item) for item in strategy_names]
    if values.ndim != 3 or min(values.shape) < 1:
        raise_input_error("net_benefits must be a non-empty 3D array.")
    if values.shape[1] != len(strategies) or values.shape[2] != len(stages):
        raise_input_error("Names must match the strategy and stage dimensions.")
    if len(set(stages)) != len(stages) or len(set(strategies)) != len(strategies):
        raise_input_error("Decision stage and strategy names must be unique.")
    if not np.all(np.isfinite(values)):
        raise_input_error("net_benefits must contain only finite values.")
    if discount_rate < 0 or irreversibility_penalty < 0 or lock_in_penalty < 0:
        raise_input_error("Rates and option penalties must be non-negative.")

    weights = np.asarray(
        [
            1.0 if stage_weights is None else float(stage_weights.get(stage, 0.0))
            for stage in stages
        ],
        dtype=DEFAULT_DTYPE,
    )
    if np.any(weights < 0) or not float(weights.sum()) > 0:
        raise_input_error(
            "stage_weights must be non-negative and sum to a positive value."
        )
    weights /= weights.sum()
    times = np.asarray(
        [
            0.0
            if evidence_arrival_times is None
            else float(evidence_arrival_times.get(stage, 0.0))
            for stage in stages
        ],
        dtype=DEFAULT_DTYPE,
    )
    if np.any(times < 0) or not np.all(np.isfinite(times)):
        raise_input_error("evidence_arrival_times must be finite and non-negative.")

    expected = np.mean(values, axis=0)
    discount = np.power(1.0 + float(discount_rate), -times)
    stage_adjustment = discount * np.maximum(0.0, 1.0 - irreversibility_penalty * times)
    adjusted = expected * stage_adjustment[None, :]
    adjusted[:, 1:] -= float(lock_in_penalty) * times[1:][None, :]
    optimal_indices = np.argmax(adjusted, axis=0)
    optimal_names = [strategies[int(index)] for index in optimal_indices]
    immediate_best = float(np.max(adjusted[:, 0]))
    flexible_value = float(np.dot(weights, np.max(adjusted, axis=0)))
    waiting_value = max(0.0, flexible_value - immediate_best)
    baseline = adjusted[:, 0]
    option_value = max(0.0, flexible_value - float(np.dot(weights, baseline)))
    regret = np.maximum(0.0, np.max(adjusted, axis=0)[None, :] - adjusted)
    robust_index = int(np.argmax(np.min(adjusted, axis=1)))
    return DynamicRealOptionsResult(
        expected_net_benefits=adjusted,
        decision_stage_names=stages,
        strategy_names=strategies,
        optimal_strategy_names=optimal_names,
        waiting_value=waiting_value,
        option_value=option_value,
        policy_path_regret=regret,
        timing_sensitivity=weights * discount,
        robust_strategy_name=strategies[robust_index],
        pareto_strategy_names=_pareto_strategies(adjusted, strategies),
        method_maturity="fixture-backed",
        diagnostics={
            "discount_rate": float(discount_rate),
            "irreversibility_penalty": float(irreversibility_penalty),
            "lock_in_penalty": float(lock_in_penalty),
            "evidence_arrival_times": dict(zip(stages, times.tolist(), strict=True)),
            "exercise_rules": dict(exercise_rules or {}),
        },
        reporting={
            "reporting_standard": "CHEERS-VOI",
            "analysis_type": "value_of_dynamic_real_options",
            "method_maturity": "fixture-backed",
            "decision_stage_names": stages,
        },
    )
