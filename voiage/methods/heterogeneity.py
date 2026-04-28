"""Value of Heterogeneity calculations."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from voiage.config import DEFAULT_DTYPE
from voiage.exceptions import raise_input_error
from voiage.schema import ValueArray


@dataclass(frozen=True)
class HeterogeneityResult:
    """Structured Value of Heterogeneity result.

    Attributes
    ----------
    value : float
        Value of tailoring decisions to subgroups.
    subgroup_labels : list[str]
        Unique subgroup labels in analysis order.
    subgroup_weights : numpy.ndarray
        Population weight for each subgroup.
    subgroup_optimal_strategy_indices : numpy.ndarray
        Optimal strategy index per subgroup.
    subgroup_optimal_strategy_names : list[str]
        Optimal strategy name per subgroup.
    subgroup_expected_net_benefits : numpy.ndarray
        Expected net benefit of the subgroup-optimal strategy.
    overall_optimal_strategy_index : int
        Optimal strategy index if a single decision is used for everyone.
    overall_optimal_strategy_name : str
        Name of the overall-optimal strategy.
    overall_expected_net_benefit : float
        Expected net benefit of the overall-optimal strategy.
    """

    value: float
    subgroup_labels: list[str]
    subgroup_weights: np.ndarray
    subgroup_optimal_strategy_indices: np.ndarray
    subgroup_optimal_strategy_names: list[str]
    subgroup_expected_net_benefits: np.ndarray
    overall_optimal_strategy_index: int
    overall_optimal_strategy_name: str
    overall_expected_net_benefit: float


def _validate_heterogeneity_inputs(
    value_array: ValueArray,
    subgroups: np.ndarray | list[Any],
    strategy_names: list[str] | None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Validate and normalize VOH inputs."""
    if not isinstance(value_array, ValueArray):
        raise_input_error("`value_array` must be a ValueArray object.")

    nb_values = np.asarray(value_array.numpy_values, dtype=DEFAULT_DTYPE)
    if nb_values.ndim != 2:
        raise_input_error(
            "Value of heterogeneity requires 2D net benefits (samples x strategies)."
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


def value_of_heterogeneity(
    value_array: ValueArray,
    subgroups: np.ndarray | list[Any],
    strategy_names: list[str] | None = None,
    n_bins: int | None = None,
) -> HeterogeneityResult:
    """Calculate the value of tailoring decisions to subgroups.

    Parameters
    ----------
    value_array : ValueArray
        2D net-benefit samples with shape ``(n_samples, n_strategies)``.
    subgroups : numpy.ndarray or list[Any]
        Subgroup label for each sample.
    strategy_names : list[str], optional
        Optional strategy labels.
    n_bins : int, optional
        Number of quantile bins to use when ``subgroups`` is numeric.

    Returns
    -------
    HeterogeneityResult
        Result containing subgroup-specific and overall expected net benefits.

    Notes
    -----
    The reported value is the gain from allowing different optimal strategies
    in different subgroups rather than applying one strategy to everyone.
    """
    nb_values, subgroup_arr, final_strategy_names = _validate_heterogeneity_inputs(
        value_array,
        subgroups,
        strategy_names,
    )
    subgroup_arr = _bin_numeric_subgroups(subgroup_arr, n_bins)

    labels = [str(label) for label in np.unique(subgroup_arr)]
    weights = np.empty(len(labels), dtype=DEFAULT_DTYPE)
    optimal_indices = np.empty(len(labels), dtype=int)
    subgroup_enb = np.empty(len(labels), dtype=DEFAULT_DTYPE)

    for idx, label in enumerate(labels):
        mask = subgroup_arr == label
        subgroup_mean_nb = np.mean(nb_values[mask], axis=0)
        optimal_idx = int(np.argmax(subgroup_mean_nb))
        weights[idx] = float(np.mean(mask))
        optimal_indices[idx] = optimal_idx
        subgroup_enb[idx] = subgroup_mean_nb[optimal_idx]

    subgroup_specific_enb = float(np.sum(weights * subgroup_enb))
    overall_mean_nb = np.mean(nb_values, axis=0)
    overall_optimal_idx = int(np.argmax(overall_mean_nb))
    overall_enb = float(overall_mean_nb[overall_optimal_idx])

    return HeterogeneityResult(
        value=max(0.0, subgroup_specific_enb - overall_enb),
        subgroup_labels=labels,
        subgroup_weights=weights,
        subgroup_optimal_strategy_indices=optimal_indices,
        subgroup_optimal_strategy_names=[
            final_strategy_names[int(idx)] for idx in optimal_indices
        ],
        subgroup_expected_net_benefits=subgroup_enb,
        overall_optimal_strategy_index=overall_optimal_idx,
        overall_optimal_strategy_name=final_strategy_names[overall_optimal_idx],
        overall_expected_net_benefit=overall_enb,
    )


def identify_optimal_subgroups(result: HeterogeneityResult) -> dict[str, str]:
    """Return the optimal strategy name for each subgroup.

    Parameters
    ----------
    result : HeterogeneityResult
        Result produced by :func:`value_of_heterogeneity`.

    Returns
    -------
    dict[str, str]
        Mapping from subgroup label to subgroup-optimal strategy name.
    """
    return dict(
        zip(
            result.subgroup_labels,
            result.subgroup_optimal_strategy_names,
            strict=True,
        )
    )
