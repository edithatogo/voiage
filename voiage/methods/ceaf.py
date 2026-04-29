"""Cost-effectiveness acceptability frontier calculations."""

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from voiage.config import DEFAULT_DTYPE
from voiage.exceptions import raise_input_error
from voiage.reporting import build_cheers_reporting
from voiage.schema import ValueArray


@dataclass(frozen=True)
class CEAFResult:
    """Container for a cost-effectiveness acceptability frontier result.

    Attributes
    ----------
    wtp_thresholds : numpy.ndarray
        Willingness-to-pay thresholds used for the frontier.
    optimal_strategy_indices : numpy.ndarray
        Index of the expected-optimal strategy at each threshold.
    optimal_strategy_names : list[str]
        Names of the expected-optimal strategies.
    acceptability_probabilities : numpy.ndarray
        Probability that the expected-optimal strategy is cost-effective.
    probability_lower : numpy.ndarray
        Lower uncertainty band for the acceptability probability.
    probability_upper : numpy.ndarray
        Upper uncertainty band for the acceptability probability.
    expected_net_benefit : numpy.ndarray
        Expected net benefit of the selected strategy at each threshold.
    """

    wtp_thresholds: np.ndarray
    optimal_strategy_indices: np.ndarray
    optimal_strategy_names: list[str]
    acceptability_probabilities: np.ndarray
    probability_lower: np.ndarray
    probability_upper: np.ndarray
    expected_net_benefit: np.ndarray
    reporting: dict[str, object]


def _validate_ceaf_inputs(
    value_array: ValueArray,
    wtp_thresholds: np.ndarray | list[float],
    strategy_names: list[str] | None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Validate and normalize CEAF inputs.

    Parameters
    ----------
    value_array : ValueArray
        3D value array with shape ``(n_samples, n_strategies, n_thresholds)``.
    wtp_thresholds : numpy.ndarray or list[float]
        Willingness-to-pay thresholds to evaluate.
    strategy_names : list[str], optional
        Override names for the strategies.

    Returns
    -------
    tuple of numpy.ndarray, numpy.ndarray, list[str]
        Normalized net-benefit values, WTP thresholds, and strategy names.
    """
    if not isinstance(value_array, ValueArray):
        raise_input_error("`value_array` must be a ValueArray object.")

    nb_values = np.asarray(value_array.numpy_values, dtype=DEFAULT_DTYPE)
    expected_ndim = 3
    if nb_values.ndim != expected_ndim:
        raise_input_error(
            "For CEAF, net-benefit values must be 3D "
            "(samples x strategies x WTP thresholds)."
        )
    if nb_values.shape[0] < 1 or nb_values.shape[1] < 1 or nb_values.shape[2] < 1:
        raise_input_error("CEAF requires at least one sample, strategy, and threshold.")
    if not np.all(np.isfinite(nb_values)):
        raise_input_error("CEAF net-benefit values must be finite.")

    wtp_arr = np.asarray(wtp_thresholds, dtype=DEFAULT_DTYPE)
    if wtp_arr.ndim != 1:
        raise_input_error("`wtp_thresholds` must be a 1D array.")
    if len(wtp_arr) != nb_values.shape[2]:
        raise_input_error(
            f"Length of wtp_thresholds ({len(wtp_arr)}) must match the third "
            f"dimension of net-benefit values ({nb_values.shape[2]})."
        )

    final_strategy_names = strategy_names or value_array.strategy_names
    if len(final_strategy_names) != nb_values.shape[1]:
        raise_input_error(
            f"Length of strategy_names ({len(final_strategy_names)}) must match "
            f"the second dimension of net-benefit values ({nb_values.shape[1]})."
        )

    return nb_values, wtp_arr, final_strategy_names


def calculate_ceaf(
    value_array: ValueArray,
    wtp_thresholds: np.ndarray | list[float],
    strategy_names: list[str] | None = None,
    confidence_level: float = 0.95,
) -> CEAFResult:
    """Calculate a cost-effectiveness acceptability frontier.

    Parameters
    ----------
    value_array : ValueArray
        3D net-benefit surface with samples, strategies, and WTP thresholds.
    wtp_thresholds : numpy.ndarray or list[float]
        Willingness-to-pay thresholds used for the frontier.
    strategy_names : list[str], optional
        Override strategy names.
    confidence_level : float, default=0.95
        Confidence level used to build the probability band.

    Returns
    -------
    CEAFResult
        Frontier result with optimal strategies, probabilities, and uncertainty
        bounds.

    Notes
    -----
    The frontier is built by selecting the strategy with the highest expected
    net benefit at each threshold, then estimating how often that strategy is
    optimal across PSA samples.
    """
    if not 0 < confidence_level < 1:
        raise_input_error("`confidence_level` must be between 0 and 1.")

    nb_values, wtp_arr, final_strategy_names = _validate_ceaf_inputs(
        value_array,
        wtp_thresholds,
        strategy_names,
    )
    n_samples = nb_values.shape[0]

    mean_nb = np.mean(nb_values, axis=0)
    optimal_strategy_indices = np.argmax(mean_nb, axis=0)
    sample_optimal_indices = np.argmax(nb_values, axis=1)
    wtp_indices = np.arange(nb_values.shape[2])
    acceptability_probabilities = np.mean(
        sample_optimal_indices == optimal_strategy_indices[np.newaxis, :],
        axis=0,
    )
    expected_net_benefit = mean_nb[optimal_strategy_indices, wtp_indices]

    z_value = float(norm.ppf(0.5 + confidence_level / 2.0))
    standard_error = np.sqrt(
        acceptability_probabilities * (1.0 - acceptability_probabilities) / n_samples
    )
    probability_lower = np.clip(
        acceptability_probabilities - z_value * standard_error,
        0.0,
        1.0,
    )
    probability_upper = np.clip(
        acceptability_probabilities + z_value * standard_error,
        0.0,
        1.0,
    )

    return CEAFResult(
        wtp_thresholds=wtp_arr,
        optimal_strategy_indices=optimal_strategy_indices,
        optimal_strategy_names=[
            final_strategy_names[int(idx)] for idx in optimal_strategy_indices
        ],
        acceptability_probabilities=acceptability_probabilities,
        probability_lower=probability_lower,
        probability_upper=probability_upper,
        expected_net_benefit=expected_net_benefit,
        reporting=build_cheers_reporting(
            analysis_type="calculate_ceaf",
            method_family="cost_effectiveness_acceptability_frontier",
            method_maturity="stable",
            estimator="frontier_probability",
            diagnostics={
                "n_samples": int(n_samples),
                "n_thresholds": len(wtp_arr),
            },
        ),
    )
