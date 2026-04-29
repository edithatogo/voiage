"""Distributional and equity-weighted VOI calculations."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from voiage.config import DEFAULT_DTYPE
from voiage.exceptions import raise_input_error
from voiage.reporting import build_cheers_reporting
from voiage.schema import ValueArray


@dataclass(frozen=True)
class DistributionalEquityResult:
    """Structured distributional and equity-weighted VOI result."""

    value: float
    subgroup_labels: list[str]
    subgroup_weights: np.ndarray
    equity_weights: np.ndarray
    subgroup_optimal_strategy_indices: np.ndarray
    subgroup_optimal_strategy_names: list[str]
    subgroup_expected_net_benefits: np.ndarray
    equity_weighted_expected_net_benefits: np.ndarray
    overall_optimal_strategy_index: int
    overall_optimal_strategy_name: str
    social_welfare_optimal_strategy_index: int
    social_welfare_optimal_strategy_name: str
    social_welfare_value: float
    method_maturity: str
    diagnostics: dict[str, object]
    reporting: dict[str, object]


def _validate_distributional_inputs(
    value_array: ValueArray,
    subgroups: np.ndarray | list[Any],
    strategy_names: list[str] | None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Validate the distributional contract inputs."""
    if not isinstance(value_array, ValueArray):
        raise_input_error("`value_array` must be a ValueArray object.")

    nb_values = np.asarray(value_array.numpy_values, dtype=DEFAULT_DTYPE)
    if nb_values.ndim != 2:
        raise_input_error(
            "Distributional VOI requires 2D net benefits (samples x strategies)."
        )
    if nb_values.shape[0] < 1 or nb_values.shape[1] < 1:
        raise_input_error("At least one sample and one strategy are required.")
    if not np.all(np.isfinite(nb_values)):
        raise_input_error("Net-benefit values must be finite.")

    subgroup_arr = np.asarray(subgroups)
    if subgroup_arr.ndim != 1:
        raise_input_error("`subgroups` must be a 1D array.")
    if len(subgroup_arr) != nb_values.shape[0]:
        raise_input_error("`subgroups` length must match the number of samples.")

    final_strategy_names = strategy_names or value_array.strategy_names
    if len(final_strategy_names) != nb_values.shape[1]:
        raise_input_error(
            "`strategy_names` length must match the number of strategies."
        )

    return nb_values, subgroup_arr, final_strategy_names


def _bin_numeric_subgroups(
    subgroups: np.ndarray,
    n_bins: int | None,
) -> np.ndarray:
    """Convert numeric subgroup values to quantile bins when requested."""
    if n_bins is None:
        return subgroups
    if n_bins < 2:
        raise_input_error("`n_bins` must be at least 2.")
    if not np.issubdtype(subgroups.dtype, np.number):
        raise_input_error("`n_bins` can only be used with numeric subgroups.")

    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(subgroups.astype(float), quantiles))
    if len(edges) <= 2:
        return np.repeat("all", len(subgroups))

    bin_indices = np.digitize(subgroups, edges[1:-1], right=True)
    return np.asarray([f"bin_{idx + 1}" for idx in bin_indices])


def _coerce_group_weights(
    labels: list[str],
    weights: np.ndarray | list[float] | dict[str, float] | None,
) -> np.ndarray:
    """Normalize subgroup weights."""
    if weights is None:
        return np.full(len(labels), 1.0 / len(labels), dtype=DEFAULT_DTYPE)

    if isinstance(weights, dict):
        missing = [label for label in labels if label not in weights]
        if missing:
            raise_input_error("`equity_weights` must include every subgroup label.")
        values = np.asarray(
            [float(weights[label]) for label in labels], dtype=DEFAULT_DTYPE
        )
    else:
        values = np.asarray(weights, dtype=DEFAULT_DTYPE)

    if values.ndim != 1 or len(values) != len(labels):
        raise_input_error("`equity_weights` must contain one value per subgroup.")
    if not np.all(np.isfinite(values)):
        raise_input_error("`equity_weights` must be finite.")
    if np.any(values < 0):
        raise_input_error("`equity_weights` must be non-negative.")

    total = float(np.sum(values))
    if total <= 0:
        raise_input_error("`equity_weights` must sum to a positive value.")
    return values / total


def value_of_distributional_equity(
    value_array: ValueArray,
    subgroups: np.ndarray | list[Any],
    strategy_names: list[str] | None = None,
    equity_weights: np.ndarray | list[float] | dict[str, float] | None = None,
    n_bins: int | None = None,
) -> DistributionalEquityResult:
    """Calculate distributional and equity-weighted value of heterogeneity.

    Parameters
    ----------
    value_array : ValueArray
        2D net-benefit samples with shape ``(n_samples, n_strategies)``.
    subgroups : numpy.ndarray or list[Any]
        Subgroup label for each sample.
    strategy_names : list[str], optional
        Optional strategy labels.
    equity_weights : sequence or mapping, optional
        Non-negative subgroup weights used for social-welfare summaries.
    n_bins : int, optional
        Number of quantile bins to use when ``subgroups`` is numeric.

    Returns
    -------
    DistributionalEquityResult
        Result containing subgroup-specific and equity-weighted summaries.
    """
    nb_values, subgroup_arr, final_strategy_names = _validate_distributional_inputs(
        value_array,
        subgroups,
        strategy_names,
    )
    subgroup_arr = _bin_numeric_subgroups(subgroup_arr, n_bins)

    labels = [str(label) for label in np.unique(subgroup_arr)]
    subgroup_weights = np.empty(len(labels), dtype=DEFAULT_DTYPE)
    subgroup_optimal_indices = np.empty(len(labels), dtype=int)
    subgroup_expected = np.empty(len(labels), dtype=DEFAULT_DTYPE)
    subgroup_population_means = np.empty(
        (len(labels), nb_values.shape[1]), dtype=DEFAULT_DTYPE
    )

    for idx, label in enumerate(labels):
        mask = subgroup_arr == label
        subgroup_means = np.mean(nb_values[mask], axis=0)
        optimal_idx = int(np.argmax(subgroup_means))
        subgroup_weights[idx] = float(np.mean(mask))
        subgroup_optimal_indices[idx] = optimal_idx
        subgroup_expected[idx] = subgroup_means[optimal_idx]
        subgroup_population_means[idx] = subgroup_means

    equity_arr = _coerce_group_weights(labels, equity_weights)
    equity_weighted_expected_net_benefits = equity_arr @ subgroup_population_means

    overall_mean_nb = np.mean(nb_values, axis=0)
    overall_optimal_idx = int(np.argmax(overall_mean_nb))
    overall_enb = float(overall_mean_nb[overall_optimal_idx])

    social_welfare_idx = int(np.argmax(equity_weighted_expected_net_benefits))
    social_welfare_value = float(
        equity_weighted_expected_net_benefits[social_welfare_idx]
    )

    return DistributionalEquityResult(
        value=max(
            0.0, float(np.sum(subgroup_weights * subgroup_expected) - overall_enb)
        ),
        subgroup_labels=labels,
        subgroup_weights=subgroup_weights,
        equity_weights=equity_arr,
        subgroup_optimal_strategy_indices=subgroup_optimal_indices,
        subgroup_optimal_strategy_names=[
            final_strategy_names[int(idx)] for idx in subgroup_optimal_indices
        ],
        subgroup_expected_net_benefits=subgroup_expected,
        equity_weighted_expected_net_benefits=equity_weighted_expected_net_benefits,
        overall_optimal_strategy_index=overall_optimal_idx,
        overall_optimal_strategy_name=final_strategy_names[overall_optimal_idx],
        social_welfare_optimal_strategy_index=social_welfare_idx,
        social_welfare_optimal_strategy_name=final_strategy_names[social_welfare_idx],
        social_welfare_value=social_welfare_value,
        method_maturity="experimental",
        diagnostics={
            "n_samples": int(nb_values.shape[0]),
            "n_strategies": int(nb_values.shape[1]),
            "n_subgroups": len(labels),
            "equity_weight_definition": (
                "Normalized subgroup weights used to summarize subgroup means"
            ),
        },
        reporting=build_cheers_reporting(
            analysis_type="value_of_distributional_equity",
            method_family="value_of_distributional_equity",
            method_maturity="experimental",
            diagnostics={
                "n_samples": int(nb_values.shape[0]),
                "n_strategies": int(nb_values.shape[1]),
                "n_subgroups": len(labels),
            },
        ),
    )
