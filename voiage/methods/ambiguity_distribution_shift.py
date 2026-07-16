"""VOI under ambiguity sets and distribution shift."""

from dataclasses import dataclass

import numpy as np

from voiage.config import DEFAULT_DTYPE
from voiage.exceptions import raise_input_error
from voiage.reporting import build_cheers_reporting
from voiage.schema import ValueArray


@dataclass(frozen=True)
class AmbiguityDistributionShiftResult:
    """Structured robust and distribution-shift VOI result."""

    value: float
    strategy_names: list[str]
    scenario_names: list[str]
    scenario_probabilities: np.ndarray
    scenario_expected_net_benefits: np.ndarray
    robust_net_benefits: np.ndarray
    robust_strategy_index: int
    robust_strategy_name: str
    robust_value: float
    informed_optimal_strategy_indices: np.ndarray
    informed_optimal_strategy_names: list[str]
    informed_expected_value: float
    scenario_regret: np.ndarray
    shift_sensitivity: np.ndarray
    ambiguity_radius: float
    information_cost: float
    method_maturity: str
    diagnostics: dict[str, object]
    reporting: dict[str, object]


def _validate_inputs(
    value_array: ValueArray,
    shift_weights: np.ndarray,
    strategy_names: list[str] | None,
    scenario_names: list[str] | None,
    scenario_probabilities: np.ndarray | list[float] | None,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str], np.ndarray]:
    """Validate and normalize source values and target-shift weights."""
    if not isinstance(value_array, ValueArray):
        raise_input_error("`value_array` must be a ValueArray object.")
    values = np.asarray(value_array.numpy_values, dtype=DEFAULT_DTYPE)
    # ValueArray enforces this shape, but keep the guard for defensive callers.
    if values.ndim != 2 or min(values.shape) < 1:  # pragma: no cover - schema invariant
        raise_input_error("Distribution-shift VOI requires 2D net benefits.")
    if not np.all(np.isfinite(values)):
        raise_input_error("Net-benefit values must be finite.")

    weights = np.asarray(shift_weights, dtype=DEFAULT_DTYPE)
    if weights.ndim != 2 or weights.shape[1] != values.shape[0] or weights.shape[0] < 1:
        raise_input_error("shift weights must be scenario x sample.")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise_input_error("Shift weights must be finite and non-negative.")
    totals = np.sum(weights, axis=1)
    if np.any(totals <= 0):
        raise_input_error("Each shift-weight scenario must be positive.")
    weights = weights / totals[:, None]

    strategies = strategy_names or value_array.strategy_names
    if len(strategies) != values.shape[1]:
        raise_input_error("`strategy_names` length must match strategy count.")
    scenarios = scenario_names or [
        f"scenario_{idx + 1}" for idx in range(weights.shape[0])
    ]
    if len(scenarios) != weights.shape[0]:
        raise_input_error("`scenario_names` length must match scenario count.")

    probabilities = np.ones(weights.shape[0], dtype=DEFAULT_DTYPE)
    if scenario_probabilities is not None:
        probabilities = np.asarray(scenario_probabilities, dtype=DEFAULT_DTYPE)
        if probabilities.ndim != 1 or len(probabilities) != weights.shape[0]:
            raise_input_error("Scenario probabilities must match scenario count.")
        if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0):
            raise_input_error("Scenario probabilities must be finite and non-negative.")
        if np.sum(probabilities) <= 0:
            raise_input_error("Scenario probabilities must sum to a positive value.")
        probabilities = probabilities / np.sum(probabilities)
    return values, weights, strategies, scenarios, probabilities


def value_of_ambiguity_distribution_shift(
    value_array: ValueArray,
    shift_weights: np.ndarray | list[list[float]],
    strategy_names: list[str] | None = None,
    scenario_names: list[str] | None = None,
    scenario_probabilities: np.ndarray | list[float] | None = None,
    ambiguity_radius: float = 0.0,
    information_cost: float = 0.0,
) -> AmbiguityDistributionShiftResult:
    r"""Calculate robust VOI under source-target distribution shift.

    Each row of ``shift_weights`` reweights source samples into a plausible
    target or drift scenario. The baseline uses a radius-penalized maximin
    score; the information value is the expected value of choosing after the
    shift scenario is resolved, less acquisition cost. The method remains
    fixture-backed pending stable promotion evidence.
    """
    if not np.isfinite(ambiguity_radius) or ambiguity_radius < 0:
        raise_input_error("`ambiguity_radius` must be finite and non-negative.")
    if not np.isfinite(information_cost) or information_cost < 0:
        raise_input_error("`information_cost` must be finite and non-negative.")
    values, weights, strategies, scenarios, probabilities = _validate_inputs(
        value_array,
        np.asarray(shift_weights, dtype=DEFAULT_DTYPE),
        strategy_names,
        scenario_names,
        scenario_probabilities,
    )
    expected = weights @ values
    shift_sensitivity = np.ptp(expected, axis=0)
    robust = np.min(expected, axis=0) - ambiguity_radius * shift_sensitivity
    robust_idx = int(np.argmax(robust))
    informed_indices = np.argmax(expected, axis=1).astype(int)
    informed_values = expected[np.arange(expected.shape[0]), informed_indices]
    informed_expected = float(probabilities @ informed_values)
    robust_value = float(robust[robust_idx])
    value = max(0.0, informed_expected - robust_value - information_cost)
    scenario_best = np.max(expected, axis=1)
    regret = scenario_best[:, None] - expected
    source_mean = np.mean(values, axis=0)
    drift = expected - source_mean[None, :]

    diagnostics: dict[str, object] = {
        "n_samples": int(values.shape[0]),
        "n_strategies": int(values.shape[1]),
        "n_shift_scenarios": int(weights.shape[0]),
        "ambiguity_metric": "radius-penalized worst-case expected net benefit",
        "drift_monitoring_status": "fixture-backed",
        "source_target_shift": drift.tolist(),
        "worst_case_regret": float(np.max(regret)),
        "robustness_envelope": robust.tolist(),
        "parity_status": "deferred",
        "open_data_status": "blocked: no licensed source-target drift snapshot committed",
    }
    return AmbiguityDistributionShiftResult(
        value=value,
        strategy_names=strategies,
        scenario_names=scenarios,
        scenario_probabilities=probabilities,
        scenario_expected_net_benefits=expected,
        robust_net_benefits=robust,
        robust_strategy_index=robust_idx,
        robust_strategy_name=strategies[robust_idx],
        robust_value=robust_value,
        informed_optimal_strategy_indices=informed_indices,
        informed_optimal_strategy_names=[
            strategies[int(idx)] for idx in informed_indices
        ],
        informed_expected_value=informed_expected,
        scenario_regret=regret,
        shift_sensitivity=shift_sensitivity,
        ambiguity_radius=float(ambiguity_radius),
        information_cost=float(information_cost),
        method_maturity="fixture-backed",
        diagnostics=diagnostics,
        reporting=build_cheers_reporting(
            analysis_type="value_of_ambiguity_distribution_shift",
            method_family="value_of_ambiguity_distribution_shift",
            method_maturity="fixture-backed",
            diagnostics={
                "n_samples": int(values.shape[0]),
                "n_strategies": int(values.shape[1]),
                "n_shift_scenarios": int(weights.shape[0]),
            },
        ),
    )
