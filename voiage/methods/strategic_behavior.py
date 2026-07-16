"""Fixture-backed VOI for strategic behavior and game-theoretic responses."""

from dataclasses import dataclass

import numpy as np

from voiage.config import DEFAULT_DTYPE
from voiage.exceptions import raise_input_error
from voiage.reporting import build_cheers_reporting


@dataclass(frozen=True)
class StrategicBehaviorResult:
    """Result envelope for strategic information decisions."""

    value: float
    selected_scenario_indices: np.ndarray
    equilibrium_probability: float
    response_sensitivity: float
    equilibrium_fragility: float
    strategic_regret: float
    disclosure_value: float
    incentive_value: float
    bargaining_value: float
    adversarial_response: float
    expected_value_strategic_information: float
    baseline_value: float
    equilibrium_threshold: float
    method_maturity: str
    diagnostics: dict[str, object]
    reporting: dict[str, object]


def value_of_strategic_behavior(
    scenario_values: np.ndarray | list[float],
    equilibrium_probabilities: np.ndarray | list[float],
    incentive_response_values: np.ndarray | list[float],
    disclosure_values: np.ndarray | list[float],
    bargaining_values: np.ndarray | list[float],
    adversarial_risks: np.ndarray | list[float],
    response_sensitivities: np.ndarray | list[float],
    strategic_regrets: np.ndarray | list[float],
    *,
    equilibrium_threshold: float = 0.5,
) -> StrategicBehaviorResult:
    """Estimate VOI when other actors respond strategically to information."""
    arrays = [
        np.asarray(values, dtype=DEFAULT_DTYPE)
        for values in (
            scenario_values,
            equilibrium_probabilities,
            incentive_response_values,
            disclosure_values,
            bargaining_values,
            adversarial_risks,
            response_sensitivities,
            strategic_regrets,
        )
    ]
    (
        values,
        equilibrium,
        incentives,
        disclosure,
        bargaining,
        adversarial,
        sensitivity,
        regrets,
    ) = arrays
    if values.ndim != 1 or len(values) < 2:
        raise_input_error("scenario_values must be a vector with at least two rows.")
    if any(array.ndim != 1 or len(array) != len(values) for array in arrays[1:]):
        raise_input_error("strategic inputs must have the same one-dimensional length.")
    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise_input_error("strategic inputs must be finite.")
    if (
        np.any(values < 0)
        or np.any(incentives < 0)
        or np.any(disclosure < 0)
        or np.any(bargaining < 0)
        or np.any(regrets < 0)
    ):
        raise_input_error(
            "scenario values and strategic value components must be non-negative."
        )
    if any(
        np.any((array < 0) | (array > 1))
        for array in (equilibrium, adversarial, sensitivity)
    ):
        raise_input_error(
            "equilibrium, adversarial, and sensitivity rates must be between zero and one."
        )
    if not np.isfinite(equilibrium_threshold) or not 0 <= equilibrium_threshold <= 1:
        raise_input_error(
            "equilibrium_threshold must be finite and between zero and one."
        )

    selected = equilibrium >= equilibrium_threshold
    equilibrium_probability = (
        float(np.mean(equilibrium[selected])) if np.any(selected) else 0.0
    )
    response_sensitivity = (
        float(np.mean(sensitivity[selected])) if np.any(selected) else 0.0
    )
    equilibrium_fragility = (
        float(np.mean(1.0 - equilibrium[selected])) if np.any(selected) else 1.0
    )
    strategic_regret = float(np.sum(regrets[selected]))
    disclosure_value = float(np.sum(disclosure[selected]))
    incentive_value = float(np.sum(incentives[selected]))
    bargaining_value = float(np.sum(bargaining[selected]))
    adversarial_response = float(np.sum(values[selected] * adversarial[selected]))
    expected = float(
        np.sum(values[selected] * equilibrium[selected])
        + incentive_value
        + disclosure_value
        + bargaining_value
        - adversarial_response
        - strategic_regret
    )
    baseline = float(np.max(values))
    value = max(0.0, expected - baseline)
    diagnostics: dict[str, object] = {
        "scenario_count": len(values),
        "selected_scenario_count": int(np.sum(selected)),
        "equilibrium_definition": "scenario equilibrium probability compared with equilibrium threshold",
        "response_sensitivity_definition": "mean actor-response sensitivity for selected scenarios",
        "adversarial_response_definition": "value-weighted adversarial risk",
        "parity_status": "deferred",
        "open_data_status": "blocked: no calibrated strategic-interaction data with provenance committed",
        "stable_promotion": "blocked pending calibrated game data, cross-language parity, and governance review",
    }
    return StrategicBehaviorResult(
        value=value,
        selected_scenario_indices=np.flatnonzero(selected),
        equilibrium_probability=equilibrium_probability,
        response_sensitivity=response_sensitivity,
        equilibrium_fragility=equilibrium_fragility,
        strategic_regret=strategic_regret,
        disclosure_value=disclosure_value,
        incentive_value=incentive_value,
        bargaining_value=bargaining_value,
        adversarial_response=adversarial_response,
        expected_value_strategic_information=expected,
        baseline_value=baseline,
        equilibrium_threshold=equilibrium_threshold,
        method_maturity="fixture-backed",
        diagnostics=diagnostics,
        reporting=build_cheers_reporting(
            analysis_type="value_of_strategic_behavior",
            method_family="strategic_behavior",
            method_maturity="fixture-backed",
            diagnostics=diagnostics,
        ),
    )
