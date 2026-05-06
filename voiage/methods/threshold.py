"""Threshold, tipping-point, and robust VOI calculations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence  # noqa: TC003
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from voiage.exceptions import raise_input_error
from voiage.methods._frontier_profiles import (
    coerce_profile_surface,
    expected_net_benefits,
    normalize_profile_weights,
    optimal_strategy_summary,
    pareto_strategy_indices,
    resolve_reference_profile,
    samplewise_profile_change_probability,
    samplewise_profile_regret,
    switching_values,
)
from voiage.reporting import build_cheers_reporting

if TYPE_CHECKING:
    from voiage.schema import ValueArray


@dataclass(frozen=True)
class ThresholdProfile:
    """Metadata for one threshold profile."""

    id: str
    label: str | None = None
    weight: float = 1.0
    profile_metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate threshold-profile metadata."""
        if not self.id or not str(self.id).strip():
            raise_input_error("Threshold profile id must be a non-empty string.")
        weight = float(self.weight)
        if not np.isfinite(weight) or weight < 0.0:
            raise_input_error(
                "Threshold profile weight must be finite and non-negative."
            )
        object.__setattr__(self, "id", str(self.id))
        object.__setattr__(self, "label", self.label or self.id)
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "profile_metadata", dict(self.profile_metadata))


@dataclass(frozen=True)
class ThresholdProfileSet:
    """Ordered collection of threshold profiles."""

    profiles: tuple[ThresholdProfile, ...]

    def __init__(self, profiles: Sequence[ThresholdProfile | str]):
        if not profiles:
            raise_input_error("At least one threshold profile is required.")
        normalized = tuple(
            item
            if isinstance(item, ThresholdProfile)
            else ThresholdProfile(id=str(item))
            for item in profiles
        )
        ids = [item.id for item in normalized]
        if len(set(ids)) != len(ids):
            raise_input_error("Threshold profile ids must be unique.")
        object.__setattr__(self, "profiles", normalized)

    @property
    def ids(self) -> list[str]:
        """Return profile identifiers in analysis order."""
        return [item.id for item in self.profiles]

    @property
    def labels(self) -> list[str]:
        """Return profile labels in analysis order."""
        return [str(item.label) for item in self.profiles]

    @property
    def weights(self) -> list[float]:
        """Return the unnormalized profile weights."""
        return [float(item.weight) for item in self.profiles]


@dataclass(frozen=True)
class ThresholdResult:
    """Structured threshold / robust VOI result."""

    analysis_id: str
    decision_problem_id: str
    value: float
    threshold_profile_ids: list[str]
    threshold_profile_labels: list[str]
    strategy_names: list[str]
    expected_net_benefits: np.ndarray
    optimal_strategy_by_threshold_profile: dict[str, str]
    threshold_crossing_probability_matrix: np.ndarray
    decision_reversal_matrix: np.ndarray
    robust_strategy: str
    tipping_point_strategy: str
    pareto_strategies: list[str]
    reference_threshold_profile: str
    method_maturity: str
    diagnostics: dict[str, object]
    reporting: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-compatible payload."""
        return {
            "analysis_id": self.analysis_id,
            "decision_problem_id": self.decision_problem_id,
            "analysis_type": "value_of_threshold_information",
            "method_maturity": self.method_maturity,
            "value": float(self.value),
            "threshold_profile_ids": list(self.threshold_profile_ids),
            "strategy_names": list(self.strategy_names),
            "expected_net_benefits": np.asarray(self.expected_net_benefits).tolist(),
            "optimal_strategy_by_threshold_profile": dict(
                self.optimal_strategy_by_threshold_profile
            ),
            "threshold_crossing_probability_matrix": np.asarray(
                self.threshold_crossing_probability_matrix
            ).tolist(),
            "decision_reversal_matrix": np.asarray(
                self.decision_reversal_matrix
            ).tolist(),
            "robust_strategy": self.robust_strategy,
            "tipping_point_strategy": self.tipping_point_strategy,
            "pareto_strategies": list(self.pareto_strategies),
            "reference_threshold_profile": self.reference_threshold_profile,
            "reporting": dict(self.reporting),
            "diagnostics": dict(self.diagnostics),
        }


def _coerce_threshold_profiles(
    n_profiles: int,
    threshold_profiles: ThresholdProfileSet | Sequence[ThresholdProfile | str] | None,
    threshold_profile_names: Sequence[str] | None,
    surface_profile_names: Sequence[str] | None,
) -> ThresholdProfileSet:
    if threshold_profiles is not None:
        profile_set = (
            threshold_profiles
            if isinstance(threshold_profiles, ThresholdProfileSet)
            else ThresholdProfileSet(threshold_profiles)
        )
    elif threshold_profile_names is not None:
        profile_set = ThresholdProfileSet(
            [str(name) for name in threshold_profile_names]
        )
    elif surface_profile_names is not None:
        profile_set = ThresholdProfileSet([str(name) for name in surface_profile_names])
    else:
        profile_set = ThresholdProfileSet(
            [f"threshold_{idx + 1}" for idx in range(n_profiles)]
        )

    if len(profile_set.profiles) != n_profiles:
        raise_input_error(
            "`threshold_profiles` length must match the number of profiles."
        )
    return profile_set


def value_of_threshold_information(
    net_benefits: ValueArray | np.ndarray,
    threshold_profiles: ThresholdProfileSet
    | Sequence[ThresholdProfile | str]
    | None = None,
    strategy_names: Sequence[str] | None = None,
    threshold_profile_names: Sequence[str] | None = None,
    threshold_profile_weights: Sequence[float] | Mapping[str, float] | None = None,
    reference_threshold_profile: str | int | None = None,
    analysis_id: str | None = None,
    decision_problem_id: str | None = None,
    decision_context: str | None = None,
) -> ThresholdResult:
    """Calculate threshold-crossing, tipping-point, and robust VOI."""
    values, final_strategy_names, final_profile_names = coerce_profile_surface(
        net_benefits,
        strategy_names,
        threshold_profile_names,
        kind="Threshold VOI",
        axis_label="samples",
        profile_names_label="threshold_profile_names",
    )
    profile_set = _coerce_threshold_profiles(
        values.shape[2],
        threshold_profiles,
        threshold_profile_names,
        final_profile_names,
    )
    profile_ids = profile_set.ids
    profile_labels = profile_set.labels
    profile_weights = normalize_profile_weights(
        profile_ids,
        threshold_profile_weights
        if threshold_profile_weights is not None
        else profile_set.weights,
        kind="threshold",
    )
    reference_index = resolve_reference_profile(
        profile_ids,
        reference_threshold_profile,
        kind="threshold",
    )

    expected = expected_net_benefits(values)
    (
        optimal_indices,
        optimal_names,
        _optimal_expected,
        _weighted_expected,
        _consensus_idx,
        _consensus_name,
        _consensus_weighted_expected,
        _robust_idx,
        robust_name,
        _robust_expected,
    ) = optimal_strategy_summary(expected, final_strategy_names, profile_weights)

    cross_probability = samplewise_profile_change_probability(values)
    reversal_matrix = samplewise_profile_regret(values, optimal_indices)
    switching = switching_values(expected, optimal_indices, reference_index)
    value = float(np.sum(profile_weights * switching))
    pareto_indices = pareto_strategy_indices(expected)
    pareto_names = [final_strategy_names[idx] for idx in pareto_indices]

    tipping_counts = np.bincount(optimal_indices, minlength=len(final_strategy_names))
    tipping_point_candidates = np.flatnonzero(tipping_counts == np.max(tipping_counts))
    tipping_point_index = int(tipping_point_candidates[0])
    if len(tipping_point_candidates) > 1:
        tied_weighted_expected = profile_weights @ expected[:, tipping_point_candidates]
        tipping_point_index = int(
            tipping_point_candidates[int(np.argmax(tied_weighted_expected))]
        )
    tipping_point_strategy = final_strategy_names[tipping_point_index]

    diagnostics = {
        "n_samples": int(values.shape[0]),
        "n_strategies": int(values.shape[1]),
        "n_threshold_profiles": int(values.shape[2]),
    }
    reporting = build_cheers_reporting(
        analysis_type="value_of_threshold_information",
        method_family="value_of_threshold_information",
        method_maturity="planned",
        analysis_id=analysis_id,
        decision_problem_id=decision_problem_id,
        decision_context=decision_context,
        diagnostics=diagnostics,
    )
    reporting["threshold_profile_ids"] = profile_ids
    reporting["threshold_profile_labels"] = profile_labels

    return ThresholdResult(
        analysis_id=analysis_id or "threshold-screening-001",
        decision_problem_id=decision_problem_id or "screening-program-001",
        value=value,
        threshold_profile_ids=profile_ids,
        threshold_profile_labels=profile_labels,
        strategy_names=final_strategy_names,
        expected_net_benefits=expected,
        optimal_strategy_by_threshold_profile=dict(
            zip(profile_ids, optimal_names, strict=True)
        ),
        threshold_crossing_probability_matrix=cross_probability,
        decision_reversal_matrix=reversal_matrix,
        robust_strategy=robust_name,
        tipping_point_strategy=tipping_point_strategy,
        pareto_strategies=pareto_names,
        reference_threshold_profile=profile_ids[reference_index],
        method_maturity="planned",
        diagnostics=diagnostics,
        reporting=reporting,
    )


def value_of_threshold(
    net_benefits: ValueArray | np.ndarray,
    threshold_profiles: ThresholdProfileSet
    | Sequence[ThresholdProfile | str]
    | None = None,
    strategy_names: Sequence[str] | None = None,
    threshold_profile_names: Sequence[str] | None = None,
    threshold_profile_weights: Sequence[float] | Mapping[str, float] | None = None,
    reference_threshold_profile: str | int | None = None,
    analysis_id: str | None = None,
    decision_problem_id: str | None = None,
    decision_context: str | None = None,
) -> ThresholdResult:
    """Alias for :func:`value_of_threshold_information`."""
    return value_of_threshold_information(
        net_benefits,
        threshold_profiles=threshold_profiles,
        strategy_names=strategy_names,
        threshold_profile_names=threshold_profile_names,
        threshold_profile_weights=threshold_profile_weights,
        reference_threshold_profile=reference_threshold_profile,
        analysis_id=analysis_id,
        decision_problem_id=decision_problem_id,
        decision_context=decision_context,
    )
