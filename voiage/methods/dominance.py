"""Dominance and ICER analysis for cost-effectiveness results."""

from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from voiage.config import DEFAULT_DTYPE
from voiage.exceptions import raise_input_error
from voiage.reporting import build_cheers_reporting


@dataclass(frozen=True)
class DominanceResult:
    """Structured dominance-analysis result.

    Attributes
    ----------
    strategy_names : list[str]
        Strategy labels in the original input order.
    costs : numpy.ndarray
        Strategy costs.
    effects : numpy.ndarray
        Strategy effects.
    frontier_indices : list[int]
        Indices on the cost-effectiveness frontier.
    strongly_dominated_indices : list[int]
        Indices removed by strong dominance.
    extended_dominated_indices : list[int]
        Indices removed by extended dominance.
    status : list[str]
        Per-strategy status labels.
    incremental_costs : numpy.ndarray
        Incremental costs between frontier strategies.
    incremental_effects : numpy.ndarray
        Incremental effects between frontier strategies.
    icers : numpy.ndarray
        Incremental cost-effectiveness ratios along the frontier.

    Examples
    --------
    >>> import numpy as np
    >>> from voiage.methods.dominance import calculate_dominance
    >>> result = calculate_dominance(np.array([10, 12]), np.array([1.0, 1.2]))
    >>> result.frontier_indices
    [0, 1]
    """

    strategy_names: list[str]
    costs: np.ndarray
    effects: np.ndarray
    frontier_indices: list[int]
    strongly_dominated_indices: list[int]
    extended_dominated_indices: list[int]
    status: list[str]
    incremental_costs: np.ndarray
    incremental_effects: np.ndarray
    icers: np.ndarray
    reporting: dict[str, object]


def _validate_cost_effect_inputs(
    costs: np.ndarray | list[float],
    effects: np.ndarray | list[float],
    strategy_names: list[str] | None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Validate dominance inputs."""
    cost_arr = np.asarray(costs, dtype=DEFAULT_DTYPE)
    effect_arr = np.asarray(effects, dtype=DEFAULT_DTYPE)
    if cost_arr.ndim != 1 or effect_arr.ndim != 1:
        raise_input_error("`costs` and `effects` must be 1D arrays.")
    if len(cost_arr) != len(effect_arr):
        raise_input_error("`costs` and `effects` must have the same length.")
    if len(cost_arr) < 2:
        raise_input_error("At least two strategies are required.")
    if not np.all(np.isfinite(cost_arr)) or not np.all(np.isfinite(effect_arr)):
        raise_input_error("`costs` and `effects` must contain only finite values.")

    final_names = strategy_names or [f"Strategy {idx}" for idx in range(len(cost_arr))]
    if len(final_names) != len(cost_arr):
        raise_input_error("`strategy_names` length must match `costs` and `effects`.")
    return cost_arr, effect_arr, final_names


def calculate_dominance(
    costs: np.ndarray | list[float],
    effects: np.ndarray | list[float],
    strategy_names: list[str] | None = None,
) -> DominanceResult:
    """Identify dominance classes and frontier ICERs.

    Parameters
    ----------
    costs : numpy.ndarray or list[float]
        Strategy costs.
    effects : numpy.ndarray or list[float]
        Strategy effects.
    strategy_names : list[str], optional
        Optional strategy labels.

    Returns
    -------
    DominanceResult
        Dominance classification with frontier indices and ICERs.

    Notes
    -----
    Strong dominance removes any strategy that is at least as costly and no
    more effective than another strategy, with one dimension strictly worse.
    Extended dominance removes frontier strategies whose incremental
    cost-effectiveness ratios are not strictly increasing along the frontier.

    References
    ----------
    Drummond, M. F., Sculpher, M. J., Claxton, K., Stoddart, G. L., & Torrance,
    G. W. (2015). *Methods for the Economic Evaluation of Health Care
    Programmes*.
    Briggs, A. H., Claxton, K., & Sculpher, M. J. (2006). *Decision Modelling
    for Health Economic Evaluation*.

    Examples
    --------
    >>> import numpy as np
    >>> from voiage.methods.dominance import calculate_dominance
    >>> result = calculate_dominance(
    ...     costs=np.array([10.0, 12.0, 13.0]),
    ...     effects=np.array([1.0, 1.1, 1.3]),
    ...     strategy_names=["A", "B", "C"],
    ... )
    >>> result.frontier_indices
    [0, 1, 2]
    """
    cost_arr, effect_arr, final_names = _validate_cost_effect_inputs(
        costs,
        effects,
        strategy_names,
    )
    strong = calculate_strong_dominance(cost_arr, effect_arr)
    frontier = cost_effectiveness_frontier(cost_arr, effect_arr)
    extended = [
        idx for idx in range(len(cost_arr)) if idx not in strong and idx not in frontier
    ]
    incremental_costs, incremental_effects, icers = calculate_icers(
        cost_arr,
        effect_arr,
        frontier,
    )

    status = ["frontier"] * len(cost_arr)
    for idx in strong:
        status[idx] = "strongly_dominated"
    for idx in extended:
        status[idx] = "extended_dominated"

    return DominanceResult(
        strategy_names=final_names,
        costs=cost_arr,
        effects=effect_arr,
        frontier_indices=frontier,
        strongly_dominated_indices=strong,
        extended_dominated_indices=extended,
        status=status,
        incremental_costs=incremental_costs,
        incremental_effects=incremental_effects,
        icers=icers,
        reporting=build_cheers_reporting(
            analysis_type="calculate_dominance",
            method_family="dominance_analysis",
            method_maturity="stable",
            estimator="deterministic_frontier",
            diagnostics={
                "n_strategies": len(cost_arr),
                "frontier_size": len(frontier),
            },
        ),
    )


def calculate_strong_dominance(
    costs: np.ndarray | list[float],
    effects: np.ndarray | list[float],
) -> list[int]:
    """Return indices of strongly dominated strategies.

    Parameters
    ----------
    costs : numpy.ndarray or list[float]
        Strategy costs.
    effects : numpy.ndarray or list[float]
        Strategy effects.

    Returns
    -------
    list[int]
        Indices of strategies that are more costly and no more effective than
        at least one alternative.

    Notes
    -----
    A strategy is strongly dominated if another strategy has cost less than or
    equal to it and effect greater than or equal to it, with at least one
    strict inequality.

    Examples
    --------
    >>> import numpy as np
    >>> from voiage.methods.dominance import calculate_strong_dominance
    >>> calculate_strong_dominance(np.array([10.0, 12.0]), np.array([1.0, 0.9]))
    [1]
    """
    cost_arr, effect_arr, _ = _validate_cost_effect_inputs(costs, effects, None)
    dominated: list[int] = []
    for idx in range(len(cost_arr)):
        other_indices = [other for other in range(len(cost_arr)) if other != idx]
        for other in other_indices:
            no_more_costly = cost_arr[other] <= cost_arr[idx]
            no_less_effective = effect_arr[other] >= effect_arr[idx]
            strict_improvement = (
                cost_arr[other] < cost_arr[idx] or effect_arr[other] > effect_arr[idx]
            )
            if no_more_costly and no_less_effective and strict_improvement:
                dominated.append(idx)
                break
    return sorted(dominated)


def cost_effectiveness_frontier(
    costs: np.ndarray | list[float],
    effects: np.ndarray | list[float],
) -> list[int]:
    """Return strategy indices on the cost-effectiveness frontier.

    Parameters
    ----------
    costs : numpy.ndarray or list[float]
        Strategy costs.
    effects : numpy.ndarray or list[float]
        Strategy effects.

    Returns
    -------
    list[int]
        Ordered frontier indices after removing strongly dominated strategies.

    Notes
    -----
    The frontier is the ordered subset of non-dominated strategies with
    strictly increasing effect and strictly improving ICERs after the
    extended-dominance pruning pass.

    Examples
    --------
    >>> import numpy as np
    >>> from voiage.methods.dominance import cost_effectiveness_frontier
    >>> cost_effectiveness_frontier(np.array([10.0, 12.0, 13.0]), np.array([1.0, 1.1, 1.3]))
    [0, 1, 2]
    """
    cost_arr, effect_arr, _ = _validate_cost_effect_inputs(costs, effects, None)
    strong = set(calculate_strong_dominance(cost_arr, effect_arr))
    candidates = [idx for idx in range(len(cost_arr)) if idx not in strong]
    candidates.sort(key=lambda idx: (effect_arr[idx], cost_arr[idx]))

    frontier: list[int] = []
    for idx in candidates:
        if frontier and np.isclose(effect_arr[idx], effect_arr[frontier[-1]]):
            if cost_arr[idx] < cost_arr[frontier[-1]]:
                frontier[-1] = idx
            continue
        frontier.append(idx)

    changed = True
    while changed and len(frontier) >= 3:
        changed = False
        for pos in range(1, len(frontier) - 1):
            prev_idx = frontier[pos - 1]
            curr_idx = frontier[pos]
            next_idx = frontier[pos + 1]
            prev_icer = _incremental_icer(
                cost_arr[prev_idx],
                effect_arr[prev_idx],
                cost_arr[curr_idx],
                effect_arr[curr_idx],
            )
            next_icer = _incremental_icer(
                cost_arr[curr_idx],
                effect_arr[curr_idx],
                cost_arr[next_idx],
                effect_arr[next_idx],
            )
            if next_icer <= prev_icer:
                del frontier[pos]
                changed = True
                break

    return frontier


def calculate_extended_dominance(
    costs: np.ndarray | list[float],
    effects: np.ndarray | list[float],
) -> list[int]:
    """Return indices removed from the frontier by extended dominance.

    Parameters
    ----------
    costs : numpy.ndarray or list[float]
        Strategy costs.
    effects : numpy.ndarray or list[float]
        Strategy effects.

    Returns
    -------
    list[int]
        Indices that are neither strongly dominated nor on the frontier.

    Notes
    -----
    Extended dominance removes strategies that fall below the linear segment
    between two frontier strategies and therefore cannot be optimal at any
    willingness-to-pay threshold.

    Examples
    --------
    >>> import numpy as np
    >>> from voiage.methods.dominance import calculate_extended_dominance
    >>> calculate_extended_dominance(np.array([10.0, 11.0, 13.0]), np.array([1.0, 1.2, 1.3]))
    []
    """
    cost_arr, effect_arr, _ = _validate_cost_effect_inputs(costs, effects, None)
    strong = set(calculate_strong_dominance(cost_arr, effect_arr))
    frontier = set(cost_effectiveness_frontier(cost_arr, effect_arr))
    return sorted(
        idx for idx in range(len(cost_arr)) if idx not in strong and idx not in frontier
    )


def calculate_icers(
    costs: np.ndarray | list[float],
    effects: np.ndarray | list[float],
    frontier_indices: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Calculate incremental costs, effects, and ICERs along the frontier.

    Parameters
    ----------
    costs : numpy.ndarray or list[float]
        Strategy costs.
    effects : numpy.ndarray or list[float]
        Strategy effects.
    frontier_indices : list[int], optional
        Precomputed frontier ordering.

    Returns
    -------
    tuple of numpy.ndarray
        Incremental costs, incremental effects, and ICERs.

    Notes
    -----
    For adjacent frontier strategies :math:`i-1` and :math:`i`, the ICER is

    .. math::

       \mathrm{ICER}_i = \frac{C_i - C_{i-1}}{E_i - E_{i-1}}.

    Examples
    --------
    >>> import numpy as np
    >>> from voiage.methods.dominance import calculate_icers
    >>> inc_costs, inc_effects, icers = calculate_icers(
    ...     np.array([10.0, 12.0, 13.0]),
    ...     np.array([1.0, 1.1, 1.3]),
    ... )
    >>> inc_costs.tolist()
    [2.0, 1.0]
    """
    cost_arr, effect_arr, _ = _validate_cost_effect_inputs(costs, effects, None)
    frontier = frontier_indices or cost_effectiveness_frontier(cost_arr, effect_arr)
    if len(frontier) < 2:
        return np.array([]), np.array([]), np.array([])

    incremental_costs = []
    incremental_effects = []
    icers = []
    for prev_idx, next_idx in pairwise(frontier):
        inc_cost = cost_arr[next_idx] - cost_arr[prev_idx]
        inc_effect = effect_arr[next_idx] - effect_arr[prev_idx]
        incremental_costs.append(inc_cost)
        incremental_effects.append(inc_effect)
        icers.append(inc_cost / inc_effect if inc_effect > 0 else np.inf)

    return (
        np.asarray(incremental_costs, dtype=DEFAULT_DTYPE),
        np.asarray(incremental_effects, dtype=DEFAULT_DTYPE),
        np.asarray(icers, dtype=DEFAULT_DTYPE),
    )


def _incremental_icer(
    prev_cost: float,
    prev_effect: float,
    next_cost: float,
    next_effect: float,
) -> float:
    """Calculate an incremental ICER for two adjacent strategies."""
    incremental_effect = next_effect - prev_effect
    if incremental_effect <= 0:
        return float("inf")
    return float((next_cost - prev_cost) / incremental_effect)
