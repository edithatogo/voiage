"""Causal-identification and transportability value of information."""

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from voiage.config import DEFAULT_DTYPE
from voiage.exceptions import raise_input_error


@dataclass(frozen=True)
class CausalTransportabilityResult:
    """Fixture-backed result for source-to-target causal evidence analysis."""

    value: float
    causal_identification_value: float
    transportability_value: float
    external_validity_value: float
    source_population_ids: list[str]
    target_population_ids: list[str]
    strategy_names: list[str]
    expected_net_benefits: np.ndarray
    optimal_strategy_by_target_population: dict[str, str]
    transport_weight_matrix: np.ndarray
    validity_penalty_matrix: np.ndarray
    robust_strategy: str
    pareto_strategies: list[str]
    reference_target_population: str
    method_maturity: str = "fixture-backed"
    diagnostics: dict[str, object] = field(default_factory=dict)
    reporting: dict[str, object] = field(default_factory=dict)


def value_of_causal_transportability(
    net_benefits: np.ndarray,
    source_population_ids: Sequence[str],
    target_population_ids: Sequence[str],
    strategy_names: Sequence[str],
    transport_weights: np.ndarray,
    validity_penalties: np.ndarray,
    *,
    analysis_id: str = "causal-transportability-analysis",
    decision_problem_id: str = "unspecified",
    reference_target_population: str | None = None,
) -> CausalTransportabilityResult:
    """Value causal evidence transported from source to target populations.

    ``net_benefits`` has shape ``(samples, sources, strategies)``.  Each
    target receives a weighted source estimate, reduced by its validity
    penalty; the returned value is the gain from target-specific decisions
    over the reference target's decision rule.
    """
    values = np.asarray(net_benefits, dtype=DEFAULT_DTYPE)
    sources = [str(item) for item in source_population_ids]
    targets = [str(item) for item in target_population_ids]
    strategies = [str(item) for item in strategy_names]
    weights = np.asarray(transport_weights, dtype=DEFAULT_DTYPE)
    penalties = np.asarray(validity_penalties, dtype=DEFAULT_DTYPE)
    if values.ndim != 3 or min(values.shape) < 1:
        raise_input_error(
            "net_benefits must be a non-empty 3D array (samples, sources, strategies)."
        )
    if values.shape[1:] != (len(sources), len(strategies)):
        raise_input_error(
            "net_benefits dimensions must match source and strategy names."
        )
    if len(set(sources)) != len(sources) or len(set(targets)) != len(targets):
        raise_input_error("Population identifiers must be unique.")
    if len(set(strategies)) != len(strategies):
        raise_input_error("Strategy names must be unique.")
    if (
        weights.shape != (len(sources), len(targets))
        or penalties.shape != weights.shape
    ):
        raise_input_error(
            "Transport weights and validity penalties must be source x target matrices."
        )
    if (
        not np.all(np.isfinite(values))
        or not np.all(np.isfinite(weights))
        or not np.all(np.isfinite(penalties))
    ):
        raise_input_error("Inputs must contain only finite values.")
    if np.any(weights < 0) or np.any(penalties < 0) or np.any(weights > 1):
        raise_input_error(
            "Transport weights must be in [0, 1] and penalties non-negative."
        )
    if reference_target_population is None:
        reference_target_population = targets[0]
    if reference_target_population not in targets:
        raise_input_error(
            "reference_target_population must identify a target population."
        )

    source_means = np.mean(values, axis=0)
    denominator = np.sum(weights, axis=0)
    if np.any(denominator <= 0):
        raise_input_error(
            "Every target population must have positive transport weight."
        )
    expected = (weights.T @ source_means) / denominator[:, None] - penalties.T
    optimum = np.argmax(expected, axis=1)
    reference_index = targets.index(reference_target_population)
    reference_strategy = int(optimum[reference_index])
    reference_value = float(expected[reference_index, reference_strategy])
    target_specific_value = float(np.mean(np.max(expected, axis=1)))
    value = max(0.0, target_specific_value - reference_value)
    causal_value = max(
        0.0,
        float(
            np.mean(np.max(source_means, axis=1))
            - np.max(source_means[:, reference_strategy])
        ),
    )
    transport_value = max(
        0.0, target_specific_value - float(np.mean(np.max(source_means, axis=1)))
    )
    external_value = max(
        0.0,
        float(
            np.mean(
                np.max(expected, axis=1)
                - np.max(expected[:, reference_strategy][:, None], axis=1)
            )
        ),
    )
    pareto = [
        strategies[i]
        for i in range(len(strategies))
        if not any(
            i != j
            and np.all(expected[:, j] >= expected[:, i])
            and np.any(expected[:, j] > expected[:, i])
            for j in range(len(strategies))
        )
    ]
    robust = strategies[int(np.argmax(np.min(expected, axis=0)))]
    return CausalTransportabilityResult(
        value=value,
        causal_identification_value=causal_value,
        transportability_value=transport_value,
        external_validity_value=external_value,
        source_population_ids=sources,
        target_population_ids=targets,
        strategy_names=strategies,
        expected_net_benefits=expected,
        optimal_strategy_by_target_population={
            target: strategies[int(index)]
            for target, index in zip(targets, optimum, strict=True)
        },
        transport_weight_matrix=weights,
        validity_penalty_matrix=penalties,
        robust_strategy=robust,
        pareto_strategies=pareto,
        reference_target_population=reference_target_population,
        diagnostics={
            "analysis_id": analysis_id,
            "decision_problem_id": decision_problem_id,
            "n_sources": len(sources),
            "n_targets": len(targets),
            "n_strategies": len(strategies),
        },
        reporting={
            "reporting_standard": "CHEERS-VOI",
            "analysis_type": "value_of_causal_transportability",
            "method_maturity": "fixture-backed",
        },
    )
