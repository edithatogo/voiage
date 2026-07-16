"""Monitoring and surveillance value of information."""

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from voiage.config import DEFAULT_DTYPE
from voiage.exceptions import raise_input_error


@dataclass(frozen=True)
class MonitoringSurveillanceResult:
    """Fixture-backed result envelope for ongoing monitoring VOI."""

    value: float
    monitoring_value: float
    signal_detection_value: float
    decision_revision_value: float
    stopping_value: float
    strategy_names: list[str]
    expected_net_benefits: np.ndarray
    optimal_strategy_by_period: dict[str, str]
    monitoring_cost_matrix: np.ndarray
    detection_delay_matrix: np.ndarray
    false_signal_rate_matrix: np.ndarray
    decision_revision_matrix: np.ndarray
    surveillance_frequency: float
    stopping_period: int | None
    method_maturity: str = "fixture-backed"
    diagnostics: dict[str, object] = field(default_factory=dict)
    reporting: dict[str, object] = field(default_factory=dict)


def value_of_monitoring_surveillance(
    net_benefits: np.ndarray,
    strategy_names: Sequence[str],
    monitoring_costs: np.ndarray,
    detection_delays: np.ndarray,
    false_signal_rates: np.ndarray,
    decision_revision_probabilities: np.ndarray,
    *,
    surveillance_frequency: float = 1.0,
    stopping_threshold: float = 0.5,
    analysis_id: str = "monitoring-surveillance-analysis",
    decision_problem_id: str = "unspecified",
) -> MonitoringSurveillanceResult:
    """Value periodic monitoring, signal detection, and decision revision.

    ``net_benefits`` has shape ``(samples, strategies, periods)``.  All
    monitoring matrices have shape ``(periods, strategies)``.
    """
    values = np.asarray(net_benefits, dtype=DEFAULT_DTYPE)
    strategies = [str(item) for item in strategy_names]
    costs = np.asarray(monitoring_costs, dtype=DEFAULT_DTYPE)
    delays = np.asarray(detection_delays, dtype=DEFAULT_DTYPE)
    false_signals = np.asarray(false_signal_rates, dtype=DEFAULT_DTYPE)
    revisions = np.asarray(decision_revision_probabilities, dtype=DEFAULT_DTYPE)
    if values.ndim != 3 or min(values.shape) < 1:
        raise_input_error("net_benefits must be a non-empty 3D array.")
    periods = values.shape[2]
    expected_shape = (periods, len(strategies))
    if any(
        matrix.shape != expected_shape
        for matrix in (costs, delays, false_signals, revisions)
    ):
        raise_input_error("Monitoring matrices must have period x strategy shape.")
    if len(set(strategies)) != len(strategies) or not strategies:
        raise_input_error("Strategy names must be non-empty and unique.")
    if not all(
        np.all(np.isfinite(matrix))
        for matrix in (values, costs, delays, false_signals, revisions)
    ):
        raise_input_error("Inputs must contain only finite values.")
    if surveillance_frequency <= 0 or not np.isfinite(surveillance_frequency):
        raise_input_error("surveillance_frequency must be positive and finite.")
    if not 0 <= stopping_threshold <= 1:
        raise_input_error("stopping_threshold must be in [0, 1].")
    if (
        np.any(costs < 0)
        or np.any(delays < 0)
        or np.any(false_signals < 0)
        or np.any(revisions < 0)
        or np.any(revisions > 1)
    ):
        raise_input_error(
            "Costs, delays, and false-signal rates must be non-negative; revision probabilities must be in [0, 1]."
        )

    raw = np.mean(values, axis=0).T
    adjusted = raw - costs - delays - false_signals + revisions * surveillance_frequency
    optimal = np.argmax(adjusted, axis=1)
    baseline = float(np.max(np.mean(raw, axis=0)))
    monitored = float(np.mean(np.max(adjusted, axis=1)))
    value = max(0.0, monitored - baseline)
    monitoring_value = max(0.0, float(np.mean(raw)) - float(np.mean(raw - costs)))
    signal_detection_value = max(
        0.0, float(np.mean(revisions * surveillance_frequency))
    )
    decision_revision_value = max(0.0, float(np.mean(revisions * raw)))
    stopping_period = next(
        (
            period
            for period, probability in enumerate(np.mean(revisions, axis=1))
            if probability >= stopping_threshold
        ),
        None,
    )
    stopping_value = max(0.0, value * (1.0 if stopping_period is not None else 0.0))
    return MonitoringSurveillanceResult(
        value=value,
        monitoring_value=monitoring_value,
        signal_detection_value=signal_detection_value,
        decision_revision_value=decision_revision_value,
        stopping_value=stopping_value,
        strategy_names=strategies,
        expected_net_benefits=adjusted,
        optimal_strategy_by_period={
            str(period): strategies[int(index)] for period, index in enumerate(optimal)
        },
        monitoring_cost_matrix=costs,
        detection_delay_matrix=delays,
        false_signal_rate_matrix=false_signals,
        decision_revision_matrix=revisions,
        surveillance_frequency=float(surveillance_frequency),
        stopping_period=stopping_period,
        diagnostics={
            "analysis_id": analysis_id,
            "decision_problem_id": decision_problem_id,
            "n_periods": periods,
            "n_strategies": len(strategies),
            "expected_detection_delay": float(np.mean(delays)),
            "false_signal_rate": float(np.mean(false_signals)),
        },
        reporting={
            "reporting_standard": "CHEERS-VOI",
            "analysis_type": "value_of_monitoring_surveillance",
            "method_maturity": "fixture-backed",
        },
    )
