"""Cost-effectiveness acceptability frontier calculations."""

from dataclasses import dataclass

import numpy as np

from voiage import _runtime
from voiage.config import DEFAULT_DTYPE
from voiage.exceptions import raise_dimension_mismatch_error, raise_input_error
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

    def to_dict(
        self,
        *,
        analysis_id: str,
        decision_problem_id: str,
    ) -> dict[str, object]:
        """Serialize this result to the stable v1 CEAF payload."""
        return _runtime.serialize_ceaf_result(
            analysis_id=analysis_id,
            decision_problem_id=decision_problem_id,
            wtp_thresholds=self.wtp_thresholds.tolist(),
            optimal_strategy_indices=self.optimal_strategy_indices.tolist(),
            optimal_strategy_names=list(self.optimal_strategy_names),
            acceptability_probabilities=self.acceptability_probabilities.tolist(),
            probability_lower=self.probability_lower.tolist(),
            probability_upper=self.probability_upper.tolist(),
            expected_net_benefit=self.expected_net_benefit.tolist(),
            reporting=dict(self.reporting),
        )


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
        raise_dimension_mismatch_error("`wtp_thresholds` must be a 1D array.")
    if len(wtp_arr) != nb_values.shape[2]:
        raise_dimension_mismatch_error(
            f"Length of wtp_thresholds ({len(wtp_arr)}) must match the third "
            f"dimension of net-benefit values ({nb_values.shape[2]})."
        )
    if not np.all(np.isfinite(wtp_arr)):
        raise_input_error(
            "`wtp_thresholds` must contain only finite values.",
            diagnostic_code="non_finite_value",
        )

    final_strategy_names = strategy_names or value_array.strategy_names
    if len(final_strategy_names) != nb_values.shape[1]:
        raise_dimension_mismatch_error(
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
    r"""Calculate a cost-effectiveness acceptability frontier.

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
    At each willingness-to-pay threshold :math:`\lambda`, CEAF selects the
    strategy with the largest expected net benefit:

    .. math::

       d^*(\lambda) = \arg\max_d E[NB_d(\lambda)].

    The acceptability probability is the fraction of PSA samples for which the
    selected strategy is also optimal under the sample-level net benefits.

    References
    ----------
    Fenwick, E., Claxton, K., & Sculpher, M. (2001). Representing uncertainty:
    the cost-effectiveness acceptability curve, the cost-effectiveness
    acceptability frontier, and the expected value of sample information.
    Briggs, A. H., O'Brien, B. J., & Blackhouse, G. (2002). Thinking
    outside the box: CEAF as a decision summary.

    Examples
    --------
    >>> import numpy as np
    >>> from voiage.methods.ceaf import calculate_ceaf
    >>> from voiage.schema import ValueArray
    >>> values = np.array(
    ...     [
    ...         [[10.0, 11.0], [12.0, 9.0]],
    ...         [[9.0, 10.5], [11.0, 10.0]],
    ...     ]
    ... )
    >>> va = ValueArray.from_numpy_perspectives(
    ...     values,
    ...     strategy_names=["A", "B"],
    ...     perspective_names=["10000", "20000"],
    ... )
    >>> result = calculate_ceaf(va, [10000.0, 20000.0])
    >>> result.wtp_thresholds.tolist()
    [10000.0, 20000.0]
    """
    if not np.isfinite(confidence_level):
        raise_input_error(
            "`confidence_level` must be finite.",
            diagnostic_code="non_finite_value",
        )
    if not 0 < confidence_level < 1:
        raise_input_error("`confidence_level` must be between 0 and 1.")
    nb_values, wtp_arr, final_strategy_names = _validate_ceaf_inputs(
        value_array,
        wtp_thresholds,
        strategy_names,
    )
    native = _runtime.compute_ceaf(
        nb_values.tolist(),
        wtp_arr.tolist(),
        confidence_level,
    )
    optimal_strategy_indices = np.asarray(native["optimal_strategy_indices"], dtype=int)
    acceptability_probabilities = np.asarray(
        native["acceptability_probabilities"], dtype=DEFAULT_DTYPE
    )
    probability_lower = np.asarray(native["probability_lower"], dtype=DEFAULT_DTYPE)
    probability_upper = np.asarray(native["probability_upper"], dtype=DEFAULT_DTYPE)
    expected_net_benefit = np.asarray(
        native["expected_net_benefit"], dtype=DEFAULT_DTYPE
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
                "n_samples": int(nb_values.shape[0]),
                "n_thresholds": len(wtp_arr),
            },
        ),
    )
