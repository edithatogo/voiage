"""Model-validation and discrepancy-reduction VOI calculations."""

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
    regret_matrix,
    resolve_reference_profile,
    switching_values,
)
from voiage.reporting import build_cheers_reporting

if TYPE_CHECKING:
    from voiage.schema import ValueArray


@dataclass(frozen=True)
class ValidationProfile:
    """Metadata for one validation profile."""

    id: str
    label: str | None = None
    weight: float = 1.0
    profile_metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate validation-profile metadata."""
        if not self.id or not str(self.id).strip():
            raise_input_error("Validation profile id must be a non-empty string.")
        weight = float(self.weight)
        if not np.isfinite(weight) or weight < 0.0:
            raise_input_error(
                "Validation profile weight must be finite and non-negative."
            )
        object.__setattr__(self, "id", str(self.id))
        object.__setattr__(self, "label", self.label or self.id)
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "profile_metadata", dict(self.profile_metadata))


@dataclass(frozen=True)
class ValidationProfileSet:
    """Ordered collection of validation profiles."""

    profiles: tuple[ValidationProfile, ...]

    def __init__(self, profiles: Sequence[ValidationProfile | str]):
        if not profiles:
            raise_input_error("At least one validation profile is required.")
        normalized = tuple(
            item
            if isinstance(item, ValidationProfile)
            else ValidationProfile(id=str(item))
            for item in profiles
        )
        ids = [item.id for item in normalized]
        if len(set(ids)) != len(ids):
            raise_input_error("Validation profile ids must be unique.")
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
class ModelValidationResult:
    """Structured model-validation VOI result."""

    analysis_id: str
    decision_problem_id: str
    value: float
    discrepancy_reduction_value: float
    validation_profile_ids: list[str]
    validation_profile_labels: list[str]
    strategy_names: list[str]
    expected_net_benefits: np.ndarray
    optimal_strategy_by_validation_profile: dict[str, str]
    discrepancy_matrix: np.ndarray
    consensus_strategy: str
    robust_strategy: str
    pareto_strategies: list[str]
    reference_validation_profile: str
    method_maturity: str
    diagnostics: dict[str, object]
    reporting: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-compatible payload."""
        return {
            "analysis_id": self.analysis_id,
            "decision_problem_id": self.decision_problem_id,
            "analysis_type": "value_of_model_validation",
            "method_maturity": self.method_maturity,
            "value": float(self.value),
            "discrepancy_reduction_value": float(self.discrepancy_reduction_value),
            "validation_profile_ids": list(self.validation_profile_ids),
            "strategy_names": list(self.strategy_names),
            "expected_net_benefits": np.asarray(self.expected_net_benefits).tolist(),
            "optimal_strategy_by_validation_profile": dict(
                self.optimal_strategy_by_validation_profile
            ),
            "discrepancy_matrix": np.asarray(self.discrepancy_matrix).tolist(),
            "consensus_strategy": self.consensus_strategy,
            "robust_strategy": self.robust_strategy,
            "pareto_strategies": list(self.pareto_strategies),
            "reference_validation_profile": self.reference_validation_profile,
            "reporting": dict(self.reporting),
            "diagnostics": dict(self.diagnostics),
        }


def _coerce_validation_profiles(
    n_profiles: int,
    validation_profiles: ValidationProfileSet
    | Sequence[ValidationProfile | str]
    | None,
    validation_profile_names: Sequence[str] | None,
    surface_profile_names: Sequence[str] | None,
) -> ValidationProfileSet:
    if validation_profiles is not None:
        profile_set = (
            validation_profiles
            if isinstance(validation_profiles, ValidationProfileSet)
            else ValidationProfileSet(validation_profiles)
        )
    elif validation_profile_names is not None:
        profile_set = ValidationProfileSet(
            [str(name) for name in validation_profile_names]
        )
    elif surface_profile_names is not None:
        profile_set = ValidationProfileSet(
            [str(name) for name in surface_profile_names]
        )
    else:
        profile_set = ValidationProfileSet(
            [f"validation_{idx + 1}" for idx in range(n_profiles)]
        )

    if len(profile_set.profiles) != n_profiles:
        raise_input_error(
            "`validation_profiles` length must match the number of profiles."
        )
    return profile_set


def value_of_model_validation(
    net_benefits: ValueArray | np.ndarray,
    validation_profiles: ValidationProfileSet
    | Sequence[ValidationProfile | str]
    | None = None,
    strategy_names: Sequence[str] | None = None,
    validation_profile_names: Sequence[str] | None = None,
    validation_profile_weights: Sequence[float] | Mapping[str, float] | None = None,
    reference_validation_profile: str | int | None = None,
    analysis_id: str | None = None,
    decision_problem_id: str | None = None,
    decision_context: str | None = None,
) -> ModelValidationResult:
    """Calculate value of external validation and discrepancy reduction."""
    values, final_strategy_names, final_profile_names = coerce_profile_surface(
        net_benefits,
        strategy_names,
        validation_profile_names,
        kind="Model validation VOI",
        axis_label="samples",
        profile_names_label="validation_profile_names",
    )
    profile_set = _coerce_validation_profiles(
        values.shape[2],
        validation_profiles,
        validation_profile_names,
        final_profile_names,
    )
    profile_ids = profile_set.ids
    profile_labels = profile_set.labels
    profile_weights = normalize_profile_weights(
        profile_ids,
        validation_profile_weights
        if validation_profile_weights is not None
        else profile_set.weights,
        kind="validation",
    )
    reference_index = resolve_reference_profile(
        profile_ids,
        reference_validation_profile,
        kind="validation",
    )

    expected = expected_net_benefits(values)
    (
        optimal_indices,
        optimal_names,
        optimal_expected,
        _weighted_expected,
        consensus_idx,
        consensus_name,
        _consensus_weighted_expected,
        _robust_idx,
        robust_name,
        _robust_expected,
    ) = optimal_strategy_summary(expected, final_strategy_names, profile_weights)

    discrepancy = regret_matrix(expected, optimal_indices)
    switching = switching_values(expected, optimal_indices, reference_index)
    value = float(np.sum(profile_weights * switching))
    consensus_profile_expected = expected[:, consensus_idx]
    discrepancy_reduction_value = float(
        max(
            0.0,
            np.sum(profile_weights * (optimal_expected - consensus_profile_expected)),
        )
    )
    pareto_indices = pareto_strategy_indices(expected)
    pareto_names = [final_strategy_names[idx] for idx in pareto_indices]

    diagnostics = {
        "n_samples": int(values.shape[0]),
        "n_strategies": int(values.shape[1]),
        "n_validation_profiles": int(values.shape[2]),
    }
    reporting = build_cheers_reporting(
        analysis_type="value_of_model_validation",
        method_family="value_of_model_validation",
        method_maturity="fixture-backed",
        analysis_id=analysis_id,
        decision_problem_id=decision_problem_id,
        decision_context=decision_context,
        diagnostics=diagnostics,
    )
    reporting["validation_profile_ids"] = profile_ids
    reporting["validation_profile_labels"] = profile_labels

    return ModelValidationResult(
        analysis_id=analysis_id or "validation-screening-001",
        decision_problem_id=decision_problem_id or "screening-program-001",
        value=value,
        discrepancy_reduction_value=discrepancy_reduction_value,
        validation_profile_ids=profile_ids,
        validation_profile_labels=profile_labels,
        strategy_names=final_strategy_names,
        expected_net_benefits=expected,
        optimal_strategy_by_validation_profile=dict(
            zip(profile_ids, optimal_names, strict=True)
        ),
        discrepancy_matrix=discrepancy,
        consensus_strategy=consensus_name,
        robust_strategy=robust_name,
        pareto_strategies=pareto_names,
        reference_validation_profile=profile_ids[reference_index],
        method_maturity="fixture-backed",
        diagnostics=diagnostics,
        reporting=reporting,
    )


def value_of_validation(
    net_benefits: ValueArray | np.ndarray,
    validation_profiles: ValidationProfileSet
    | Sequence[ValidationProfile | str]
    | None = None,
    strategy_names: Sequence[str] | None = None,
    validation_profile_names: Sequence[str] | None = None,
    validation_profile_weights: Sequence[float] | Mapping[str, float] | None = None,
    reference_validation_profile: str | int | None = None,
    analysis_id: str | None = None,
    decision_problem_id: str | None = None,
    decision_context: str | None = None,
) -> ModelValidationResult:
    """Alias for :func:`value_of_model_validation`."""
    return value_of_model_validation(
        net_benefits,
        validation_profiles=validation_profiles,
        strategy_names=strategy_names,
        validation_profile_names=validation_profile_names,
        validation_profile_weights=validation_profile_weights,
        reference_validation_profile=reference_validation_profile,
        analysis_id=analysis_id,
        decision_problem_id=decision_problem_id,
        decision_context=decision_context,
    )
