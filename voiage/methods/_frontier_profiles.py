"""Shared helpers for multi-profile frontier analysis surfaces."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np

from voiage.config import DEFAULT_DTYPE
from voiage.exceptions import raise_input_error

if TYPE_CHECKING:
    from voiage.schema import ValueArray


def coerce_profile_surface(
    net_benefits: ValueArray | np.ndarray,
    strategy_names: Sequence[str] | None,
    profile_names: Sequence[str] | None,
    *,
    kind: str,
    axis_label: str,
    profile_names_label: str = "profile_names",
) -> tuple[np.ndarray, list[str], list[str] | None]:
    """Normalize a 3D net-benefit surface for a frontier analysis."""
    if hasattr(net_benefits, "numpy_values") and hasattr(
        net_benefits, "strategy_names"
    ):
        values = np.asarray(net_benefits.numpy_values, dtype=DEFAULT_DTYPE)
        final_strategy_names = (
            list(strategy_names)
            if strategy_names is not None
            else net_benefits.strategy_names
        )
        final_profile_names = (
            list(profile_names)
            if profile_names is not None
            else net_benefits.perspective_names
        )
    elif isinstance(net_benefits, np.ndarray):
        values = np.asarray(net_benefits, dtype=DEFAULT_DTYPE)
        if values.ndim != 3:
            raise_input_error(
                f"{kind} requires 3D net benefits "
                f"({axis_label} x strategies x profiles)."
            )
        n_strategies = values.shape[1]
        final_strategy_names = (
            list(strategy_names)
            if strategy_names is not None
            else [f"Strategy {idx}" for idx in range(n_strategies)]
        )
        final_profile_names = list(profile_names) if profile_names is not None else None
    else:
        raise_input_error("`net_benefits` must be a ValueArray or NumPy array.")

    if values.ndim != 3:
        raise_input_error(
            f"{kind} requires 3D net benefits ({axis_label} x strategies x profiles)."
        )
    if values.shape[0] < 1 or values.shape[1] < 1 or values.shape[2] < 1:
        raise_input_error(
            "At least one sample, one strategy, and one profile are required."
        )
    if not np.all(np.isfinite(values)):
        raise_input_error("Net-benefit values must be finite.")
    if len(final_strategy_names) != values.shape[1]:
        raise_input_error(
            "`strategy_names` length must match the number of strategies."
        )
    if final_profile_names is not None and len(final_profile_names) != values.shape[2]:
        raise_input_error(
            f"`{profile_names_label}` length must match the number of profiles."
        )

    return values, [str(item) for item in final_strategy_names], final_profile_names


def normalize_profile_weights(
    profile_ids: Sequence[str],
    weights: Sequence[float] | Mapping[str, float] | None,
    *,
    kind: str,
) -> np.ndarray:
    """Normalize profile weights to a probability simplex."""
    if weights is None:
        return np.full(len(profile_ids), 1.0 / len(profile_ids), dtype=DEFAULT_DTYPE)

    if isinstance(weights, Mapping):
        missing = [
            profile_id for profile_id in profile_ids if profile_id not in weights
        ]
        if missing:
            raise_input_error(f"`{kind}_weights` must include every profile id.")
        values = np.asarray(
            [float(weights[profile_id]) for profile_id in profile_ids],
            dtype=DEFAULT_DTYPE,
        )
    else:
        values = np.asarray(weights, dtype=DEFAULT_DTYPE)

    if values.ndim != 1 or len(values) != len(profile_ids):
        raise_input_error(f"`{kind}_weights` must contain one value per profile.")
    if not np.all(np.isfinite(values)):
        raise_input_error(f"`{kind}_weights` must be finite.")
    if np.any(values < 0):
        raise_input_error(f"`{kind}_weights` must be non-negative.")

    total = float(np.sum(values))
    if total <= 0:
        raise_input_error(f"`{kind}_weights` must sum to a positive value.")
    return values / total


def resolve_reference_profile(
    profile_ids: Sequence[str],
    reference_profile: str | int | None,
    *,
    kind: str,
) -> int:
    """Resolve a reference profile identifier to an index."""
    if reference_profile is None:
        return 0
    if isinstance(reference_profile, int):
        if reference_profile < 0 or reference_profile >= len(profile_ids):
            raise_input_error(f"`reference_{kind}_profile` index is out of range.")
        return reference_profile
    try:
        return list(profile_ids).index(reference_profile)
    except ValueError:
        raise_input_error(f"`reference_{kind}_profile` must identify a profile.")


def expected_net_benefits(values: np.ndarray) -> np.ndarray:
    """Collapse the sample axis into profile-specific expected net benefits."""
    return np.mean(values, axis=0, dtype=DEFAULT_DTYPE).T


def optimal_strategy_summary(
    expected: np.ndarray,
    strategy_names: Sequence[str],
    profile_weights: np.ndarray,
) -> tuple[
    np.ndarray,
    list[str],
    np.ndarray,
    np.ndarray,
    int,
    str,
    float,
    int,
    str,
    float,
]:
    """Compute profile-specific and weighted strategy summaries."""
    optimal_strategy_indices = np.argmax(expected, axis=1)
    optimal_strategy_names = [
        str(strategy_names[idx]) for idx in optimal_strategy_indices
    ]
    optimal_expected_net_benefits = expected[
        np.arange(expected.shape[0]), optimal_strategy_indices
    ]

    weighted_expected = profile_weights @ expected
    consensus_strategy_index = int(np.argmax(weighted_expected))
    consensus_strategy_name = str(strategy_names[consensus_strategy_index])
    consensus_weighted_expected_net_benefit = float(
        weighted_expected[consensus_strategy_index]
    )

    robust_expected = np.min(expected, axis=0)
    robust_strategy_index = int(np.argmax(robust_expected))
    robust_strategy_name = str(strategy_names[robust_strategy_index])
    robust_weighted_expected_net_benefit = float(robust_expected[robust_strategy_index])

    return (
        optimal_strategy_indices,
        optimal_strategy_names,
        optimal_expected_net_benefits,
        weighted_expected,
        consensus_strategy_index,
        consensus_strategy_name,
        consensus_weighted_expected_net_benefit,
        robust_strategy_index,
        robust_strategy_name,
        robust_weighted_expected_net_benefit,
    )


def pareto_strategy_indices(expected: np.ndarray) -> list[int]:
    """Return the indices of non-dominated strategies."""
    values = expected.T
    ge = np.all(values[:, None, :] >= values[None, :, :], axis=2)
    gt = np.any(values[:, None, :] > values[None, :, :], axis=2)
    return np.where(~np.any(ge & gt, axis=0))[0].tolist()


def regret_matrix(
    expected: np.ndarray, optimal_strategy_indices: np.ndarray
) -> np.ndarray:
    """Compute the profile-by-profile regret matrix."""
    n_profiles = expected.shape[0]
    regret = np.empty((n_profiles, n_profiles), dtype=DEFAULT_DTYPE)
    for i in range(n_profiles):
        best_value = expected[i, optimal_strategy_indices[i]]
        for j in range(n_profiles):
            regret[i, j] = best_value - expected[i, optimal_strategy_indices[j]]
    return regret


def switching_values(
    expected: np.ndarray,
    optimal_strategy_indices: np.ndarray,
    reference_index: int,
) -> np.ndarray:
    """Compute switching value relative to the reference profile."""
    reference_strategy_index = optimal_strategy_indices[reference_index]
    return (
        expected[np.arange(expected.shape[0]), optimal_strategy_indices]
        - expected[
            np.arange(expected.shape[0]),
            reference_strategy_index,
        ]
    )


def samplewise_profile_change_probability(values: np.ndarray) -> np.ndarray:
    """Compute profile-by-profile strategy-change probabilities."""
    samplewise_optima = np.argmax(values, axis=1)
    n_profiles = values.shape[2]
    matrix = np.empty((n_profiles, n_profiles), dtype=DEFAULT_DTYPE)
    for i in range(n_profiles):
        for j in range(n_profiles):
            matrix[i, j] = float(
                np.mean(samplewise_optima[:, i] != samplewise_optima[:, j])
            )
    return matrix


def samplewise_profile_regret(
    values: np.ndarray, optimal_strategy_indices: np.ndarray
) -> np.ndarray:
    """Compute profile-by-profile samplewise regret."""
    n_profiles = values.shape[2]
    matrix = np.empty((n_profiles, n_profiles), dtype=DEFAULT_DTYPE)
    samplewise_max = np.max(values, axis=1)
    for i in range(n_profiles):
        for j in range(n_profiles):
            chosen_values = values[:, optimal_strategy_indices[j], i]
            matrix[i, j] = float(np.mean(samplewise_max[:, i] - chosen_values))
    return matrix
