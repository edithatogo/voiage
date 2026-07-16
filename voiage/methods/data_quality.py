"""Data-quality, privacy, measurement-error, and linkage VOI."""

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from voiage.config import DEFAULT_DTYPE
from voiage.exceptions import raise_input_error


@dataclass(frozen=True)
class DataQualityResult:
    """Fixture-backed result envelope for data-quality decision analysis."""

    value: float
    acquisition_cost_value: float
    privacy_value: float
    measurement_error_value: float
    linkage_value: float
    data_quality_profile_ids: list[str]
    strategy_names: list[str]
    expected_net_benefits: np.ndarray
    optimal_strategy_by_data_quality_profile: dict[str, str]
    acquisition_cost_matrix: np.ndarray
    privacy_constraint_matrix: np.ndarray
    measurement_error_matrix: np.ndarray
    linkage_weight_matrix: np.ndarray
    consensus_strategy: str
    robust_strategy: str
    pareto_strategies: list[str]
    reference_data_quality_profile: str
    method_maturity: str = "fixture-backed"
    diagnostics: dict[str, object] = field(default_factory=dict)
    reporting: dict[str, object] = field(default_factory=dict)


def value_of_data_quality(
    net_benefits: np.ndarray,
    data_quality_profile_ids: Sequence[str],
    strategy_names: Sequence[str],
    acquisition_costs: np.ndarray,
    privacy_constraints: np.ndarray,
    measurement_error_rates: np.ndarray,
    linkage_weights: np.ndarray,
    *,
    analysis_id: str = "data-quality-analysis",
    decision_problem_id: str = "unspecified",
    reference_data_quality_profile: str | None = None,
) -> DataQualityResult:
    """Value improved data quality under cost, privacy, and linkage tradeoffs.

    ``net_benefits`` has shape ``(samples, strategies, profiles)``. Quality
    penalties are supplied as profile-by-strategy matrices.
    """
    values = np.asarray(net_benefits, dtype=DEFAULT_DTYPE)
    profiles = [str(item) for item in data_quality_profile_ids]
    strategies = [str(item) for item in strategy_names]
    costs = np.asarray(acquisition_costs, dtype=DEFAULT_DTYPE)
    privacy = np.asarray(privacy_constraints, dtype=DEFAULT_DTYPE)
    error = np.asarray(measurement_error_rates, dtype=DEFAULT_DTYPE)
    linkage = np.asarray(linkage_weights, dtype=DEFAULT_DTYPE)
    if values.ndim != 3 or min(values.shape) < 1:
        raise_input_error(
            "net_benefits must be a non-empty 3D array (samples, strategies, profiles)."
        )
    if values.shape[1:] != (len(strategies), len(profiles)):
        raise_input_error(
            "net_benefits dimensions must match strategy and profile names."
        )
    expected_shape = (len(profiles), len(strategies))
    if any(
        matrix.shape != expected_shape for matrix in (costs, privacy, error, linkage)
    ):
        raise_input_error("Quality matrices must have profile x strategy shape.")
    if len(set(profiles)) != len(profiles) or len(set(strategies)) != len(strategies):
        raise_input_error("Profile and strategy names must be unique.")
    if not all(
        np.all(np.isfinite(matrix))
        for matrix in (values, costs, privacy, error, linkage)
    ):
        raise_input_error("Inputs must contain only finite values.")
    if (
        np.any(costs < 0)
        or np.any(privacy < 0)
        or np.any(error < 0)
        or np.any(error > 1)
    ):
        raise_input_error(
            "Costs, privacy constraints, and measurement error must be non-negative; error must be <= 1."
        )
    if np.any(linkage < 0) or np.any(linkage > 1):
        raise_input_error("Linkage weights must be in [0, 1].")
    reference_data_quality_profile = reference_data_quality_profile or profiles[0]
    if reference_data_quality_profile not in profiles:
        raise_input_error("reference_data_quality_profile must identify a profile.")

    raw = np.mean(values, axis=0).T
    adjusted = raw * linkage - costs - privacy - error
    optimal = np.argmax(adjusted, axis=1)
    reference_index = profiles.index(reference_data_quality_profile)
    reference_strategy = int(optimal[reference_index])
    reference_value = float(adjusted[reference_index, reference_strategy])
    flexible_value = float(np.mean(np.max(adjusted, axis=1)))
    value = max(0.0, flexible_value - reference_value)
    acquisition_value = max(0.0, float(np.mean(raw)) - float(np.mean(raw - costs)))
    privacy_value = max(
        0.0, float(np.mean(raw - costs)) - float(np.mean(raw - costs - privacy))
    )
    error_value = max(
        0.0, float(np.mean(raw - costs - privacy)) - float(np.mean(adjusted))
    )
    linkage_value = max(0.0, float(np.mean(raw)) - float(np.mean(raw * linkage)))
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
    return DataQualityResult(
        value=value,
        acquisition_cost_value=acquisition_value,
        privacy_value=privacy_value,
        measurement_error_value=error_value,
        linkage_value=linkage_value,
        data_quality_profile_ids=profiles,
        strategy_names=strategies,
        expected_net_benefits=adjusted,
        optimal_strategy_by_data_quality_profile={
            profile: strategies[int(index)]
            for profile, index in zip(profiles, optimal, strict=True)
        },
        acquisition_cost_matrix=costs,
        privacy_constraint_matrix=privacy,
        measurement_error_matrix=error,
        linkage_weight_matrix=linkage,
        consensus_strategy=consensus,
        robust_strategy=robust,
        pareto_strategies=pareto,
        reference_data_quality_profile=reference_data_quality_profile,
        diagnostics={
            "analysis_id": analysis_id,
            "decision_problem_id": decision_problem_id,
            "n_profiles": len(profiles),
            "n_strategies": len(strategies),
        },
        reporting={
            "reporting_standard": "CHEERS-VOI",
            "analysis_type": "value_of_data_quality_privacy_linkage",
            "method_maturity": "fixture-backed",
        },
    )
