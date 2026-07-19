"""Value of Perspective analysis.

This module treats decision perspective as an explicit analysis dimension. It
compares the same strategies under multiple stakeholder or policy objective
functions and reports the regret from applying one perspective's decision rule
under another perspective.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from voiage.config import DEFAULT_DTYPE
from voiage.exceptions import raise_input_error
from voiage.reporting import build_cheers_reporting
from voiage.schema import ValueArray

METHOD_CONTRACT_VERSION = "1.1.0"
_TIE_POLICIES = frozenset({"first", "split", "error"})


@dataclass(frozen=True)
class Perspective:
    """Metadata for one decision-analysis perspective.

    Parameters
    ----------
    id : str
        Stable machine-readable perspective identifier.
    label : str, optional
        Human-readable label. Defaults to ``id`` when omitted.
    cost_categories : tuple[str, ...], default=()
        Cost categories included in the perspective.
    effect_measures : tuple[str, ...], default=()
        Effect measures included in the perspective.
    utility_weights : mapping[str, float], optional
        Utility weights used to construct net benefit.
    equity_weights : mapping[str, float], optional
        Equity weights used to construct net benefit.
    willingness_to_pay : float, optional
        Willingness-to-pay threshold for this perspective.
    discount_rate : float, optional
        Discount rate for this perspective.
    population : float, optional
        Population scaling factor for this perspective.
    stakeholder_metadata : mapping[str, object], optional
        Additional stakeholder metadata.
    """

    id: str
    label: str | None = None
    cost_categories: tuple[str, ...] = ()
    effect_measures: tuple[str, ...] = ()
    utility_weights: Mapping[str, float] = field(default_factory=dict)
    equity_weights: Mapping[str, float] = field(default_factory=dict)
    willingness_to_pay: float | None = None
    discount_rate: float | None = None
    population: float | None = None
    stakeholder_metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate perspective metadata."""
        if not self.id or not str(self.id).strip():
            raise_input_error("Perspective id must be a non-empty string.")
        object.__setattr__(self, "id", str(self.id))
        object.__setattr__(self, "label", self.label or self.id)
        object.__setattr__(
            self, "cost_categories", tuple(str(item) for item in self.cost_categories)
        )
        object.__setattr__(
            self, "effect_measures", tuple(str(item) for item in self.effect_measures)
        )

        for name, value in (
            ("willingness_to_pay", self.willingness_to_pay),
            ("discount_rate", self.discount_rate),
            ("population", self.population),
        ):
            if value is not None and not np.isfinite(float(value)):
                raise_input_error(f"Perspective {name} must be finite.")


@dataclass(frozen=True)
class PerspectiveSet:
    """Ordered collection of unique decision perspectives."""

    perspectives: tuple[Perspective, ...]

    def __init__(self, perspectives: Sequence[Perspective | str]):
        """Create a PerspectiveSet from Perspective objects or ids."""
        if not perspectives:
            raise_input_error("At least one perspective is required.")

        normalized = tuple(
            item if isinstance(item, Perspective) else Perspective(id=str(item))
            for item in perspectives
        )
        ids = [item.id for item in normalized]
        if len(set(ids)) != len(ids):
            raise_input_error("Perspective ids must be unique.")

        object.__setattr__(self, "perspectives", normalized)

    @property
    def ids(self) -> list[str]:
        """Return perspective identifiers in analysis order."""
        return [item.id for item in self.perspectives]

    @property
    def labels(self) -> list[str]:
        """Return perspective labels in analysis order."""
        return [str(item.label) for item in self.perspectives]


@dataclass(frozen=True)
class ValueOfPerspectiveResult:
    """Structured Value of Perspective result.

    Attributes
    ----------
    value : float
        Weighted switching value relative to the reference perspective.
    perspective_ids : list[str]
        Perspective identifiers in analysis order.
    perspective_labels : list[str]
        Human-readable perspective labels.
    strategy_names : list[str]
        Strategy labels in analysis order.
    expected_net_benefits : numpy.ndarray
        Expected net benefit with shape ``(n_perspectives, n_strategies)``.
    optimal_strategy_indices : numpy.ndarray
        Perspective-specific optimal strategy index.
    optimal_strategy_names : list[str]
        Perspective-specific optimal strategy name.
    optimal_expected_net_benefits : numpy.ndarray
        Expected net benefit of each perspective-specific optimum.
    regret_matrix : numpy.ndarray
        Matrix where row ``i`` and column ``j`` is regret in perspective ``i``
        when using the strategy optimal under perspective ``j``.
    switching_values : numpy.ndarray
        Regret avoided by switching away from the reference perspective's
        strategy toward each perspective's own optimal strategy.
    consensus_strategy_index : int
        Strategy maximizing weighted expected net benefit across perspectives.
    consensus_strategy_name : str
        Name of the consensus strategy.
    consensus_weighted_expected_net_benefit : float
        Weighted expected net benefit of the consensus strategy.
    robust_strategy_index : int
        Strategy maximizing the minimum expected net benefit across
        perspectives.
    robust_strategy_name : str
        Name of the robust maximin strategy.
    pareto_strategy_indices : list[int]
        Strategy indices not dominated across perspectives.
    pareto_strategy_names : list[str]
        Strategy names not dominated across perspectives.
    perspective_weights : numpy.ndarray
        Normalized perspective weights used for consensus and value summaries.
    reference_perspective_id : str
        Perspective used as the reference decision rule.
    method_maturity : str
        Maturity label. Value of Perspective is fixture-backed.
    diagnostics : dict[str, object]
        Deterministic diagnostics for downstream reporting.
    """

    value: float
    perspective_ids: list[str]
    perspective_labels: list[str]
    strategy_names: list[str]
    expected_net_benefits: np.ndarray
    optimal_strategy_indices: np.ndarray
    optimal_strategy_names: list[str]
    optimal_expected_net_benefits: np.ndarray
    regret_matrix: np.ndarray
    switching_values: np.ndarray
    consensus_strategy_index: int
    consensus_strategy_name: str
    consensus_weighted_expected_net_benefit: float
    robust_strategy_index: int
    robust_strategy_name: str
    pareto_strategy_indices: list[int]
    pareto_strategy_names: list[str]
    perspective_weights: np.ndarray
    reference_perspective_id: str
    method_maturity: str
    diagnostics: dict[str, object]
    reporting: dict[str, object] = field(default_factory=dict)


def _coerce_net_benefits(
    net_benefits: ValueArray | np.ndarray,
    strategy_names: Sequence[str] | None,
    perspective_names: Sequence[str] | None,
) -> tuple[np.ndarray, list[str], list[str] | None]:
    """Normalize net-benefit input to a finite 3D NumPy array."""
    if isinstance(net_benefits, ValueArray):
        values = np.asarray(net_benefits.numpy_values, dtype=DEFAULT_DTYPE)
        final_strategy_names = (
            list(strategy_names)
            if strategy_names is not None
            else net_benefits.strategy_names
        )
        final_perspective_names = (
            list(perspective_names)
            if perspective_names is not None
            else net_benefits.perspective_names
        )
    elif isinstance(net_benefits, np.ndarray):
        values = np.asarray(net_benefits, dtype=DEFAULT_DTYPE)
        if values.ndim != 3:
            raise_input_error(
                "Value of Perspective requires 3D net benefits "
                "(samples x strategies x perspectives)."
            )
        n_strategies = values.shape[1]
        final_strategy_names = (
            list(strategy_names)
            if strategy_names is not None
            else [f"Strategy {idx}" for idx in range(n_strategies)]
        )
        final_perspective_names = (
            list(perspective_names) if perspective_names is not None else None
        )
    else:
        raise_input_error("`net_benefits` must be a ValueArray or NumPy array.")

    if values.ndim != 3:
        raise_input_error(
            "Value of Perspective requires 3D net benefits "
            "(samples x strategies x perspectives)."
        )
    if values.shape[0] < 1 or values.shape[1] < 1 or values.shape[2] < 1:
        raise_input_error(
            "At least one sample, one strategy, and one perspective are required."
        )
    if not np.all(np.isfinite(values)):
        raise_input_error("Net-benefit values must be finite.")
    if len(final_strategy_names) != values.shape[1]:
        raise_input_error(
            "`strategy_names` length must match the number of strategies."
        )
    if (
        final_perspective_names is not None
        and len(final_perspective_names) != values.shape[2]
    ):
        raise_input_error(
            "`perspective_names` length must match the number of perspectives."
        )

    return values, [str(item) for item in final_strategy_names], final_perspective_names


def _coerce_perspectives(
    n_perspectives: int,
    perspectives: PerspectiveSet | Sequence[Perspective | str] | None,
    perspective_names: Sequence[str] | None,
) -> PerspectiveSet:
    """Normalize perspective metadata."""
    if perspectives is not None:
        perspective_set = (
            perspectives
            if isinstance(perspectives, PerspectiveSet)
            else PerspectiveSet(perspectives)
        )
    elif perspective_names is not None:
        perspective_set = PerspectiveSet([str(name) for name in perspective_names])
    else:
        perspective_set = PerspectiveSet(
            [f"Perspective {idx}" for idx in range(n_perspectives)]
        )

    if len(perspective_set.perspectives) != n_perspectives:
        raise_input_error(
            "`perspectives` length must match the number of perspectives."
        )
    return perspective_set


def _coerce_weights(
    weights: Sequence[float] | Mapping[str, float] | None,
    perspective_ids: Sequence[str],
) -> np.ndarray:
    """Normalize perspective weights to a finite probability vector."""
    n_perspectives = len(perspective_ids)
    if weights is None:
        weight_arr = np.full(n_perspectives, 1.0 / n_perspectives, dtype=DEFAULT_DTYPE)
    elif isinstance(weights, Mapping):
        missing = [item for item in perspective_ids if item not in weights]
        if missing:
            raise_input_error(
                "`perspective_weights` must include every perspective id."
            )
        weight_arr = np.asarray(
            [float(weights[item]) for item in perspective_ids], dtype=DEFAULT_DTYPE
        )
    else:
        weight_arr = np.asarray(weights, dtype=DEFAULT_DTYPE)

    if weight_arr.ndim != 1 or len(weight_arr) != n_perspectives:
        raise_input_error(
            "`perspective_weights` must contain one weight per perspective."
        )
    if not np.all(np.isfinite(weight_arr)):
        raise_input_error("`perspective_weights` must be finite.")
    if np.any(weight_arr < 0):
        raise_input_error("`perspective_weights` must be non-negative.")

    total = float(np.sum(weight_arr))
    if total <= 0:
        raise_input_error("`perspective_weights` must sum to a positive value.")
    return weight_arr / total


def _resolve_reference_index(
    reference_perspective: str | int | None,
    perspective_ids: Sequence[str],
) -> int:
    """Return the reference perspective index."""
    if reference_perspective is None:
        return 0
    if isinstance(reference_perspective, int):
        if reference_perspective < 0 or reference_perspective >= len(perspective_ids):
            raise_input_error("`reference_perspective` index is out of range.")
        return reference_perspective
    try:
        return list(perspective_ids).index(reference_perspective)
    except ValueError:
        raise_input_error("`reference_perspective` must identify a perspective.")


def _pareto_strategy_indices(expected_net_benefits: np.ndarray) -> list[int]:
    """Return strategy indices not dominated across perspectives."""
    n_strategies = expected_net_benefits.shape[1]
    pareto: list[int] = []
    for candidate in range(n_strategies):
        candidate_values = expected_net_benefits[:, candidate]
        dominated = False
        for challenger in range(n_strategies):
            if challenger == candidate:
                continue
            challenger_values = expected_net_benefits[:, challenger]
            if np.all(challenger_values >= candidate_values) and np.any(
                challenger_values > candidate_values
            ):
                dominated = True
                break
        if not dominated:
            pareto.append(candidate)
    return pareto


def value_of_perspective(
    net_benefits: ValueArray | np.ndarray,
    perspectives: PerspectiveSet | Sequence[Perspective | str] | None = None,
    strategy_names: Sequence[str] | None = None,
    perspective_names: Sequence[str] | None = None,
    perspective_weights: Sequence[float] | Mapping[str, float] | None = None,
    reference_perspective: str | int | None = None,
    tie_policy: str = "first",
    tie_tolerance: float = 1e-12,
) -> ValueOfPerspectiveResult:
    r"""Compare decision value across multiple perspectives.

    Parameters
    ----------
    net_benefits : ValueArray or numpy.ndarray
        Net-benefit samples with shape
        ``(n_samples, n_strategies, n_perspectives)``.
    perspectives : PerspectiveSet or sequence, optional
        Ordered perspective metadata. Plain strings are interpreted as
        perspective ids.
    strategy_names : sequence of str, optional
        Strategy labels.
    perspective_names : sequence of str, optional
        Perspective labels used when ``perspectives`` is omitted.
    perspective_weights : sequence or mapping, optional
        Non-negative weights used for consensus and weighted switching value.
        Mappings must be keyed by perspective id.
    reference_perspective : str or int, optional
        Perspective whose optimal strategy is used as the reference decision
        rule. Defaults to the first perspective.

    Returns
    -------
    ValueOfPerspectiveResult
        Perspective-specific optima, regret matrix, switching values,
        consensus and robust strategies, and Pareto strategy set.

    Notes
    -----
    The perspective-specific optimum is

    .. math::

       d^*_p = \arg\max_d E[NB_{d,p}].

    The reported switching value compares the reference perspective's optimal
    strategy with each perspective's own optimum, and the returned value is
    the weighted average of those switching values.

    References
    ----------
    Voiage contributors. Value of Perspective frontier contract and analysis
    notes, including the proof-of-concept preprint linked from the project
    documentation.
    Keeney, R. L., & Raiffa, H. (1993). *Decisions with Multiple Objectives*.

    Examples
    --------
    >>> import numpy as np
    >>> from voiage.analysis import DecisionAnalysis
    >>> from voiage.schema import ValueArray
    >>> values = np.array(
    ...     [
    ...         [[10.0, 7.0], [8.0, 11.0]],
    ...         [[11.0, 8.0], [9.0, 10.0]],
    ...     ]
    ... )
    >>> result = DecisionAnalysis(
    ...     ValueArray.from_numpy_perspectives(
    ...         values,
    ...         strategy_names=["A", "B"],
    ...         perspective_names=["payer", "societal"],
    ...     )
    ... ).value_of_perspective()
    >>> result.reference_perspective_id
    'payer'
    """
    values, final_strategy_names, source_perspective_names = _coerce_net_benefits(
        net_benefits,
        strategy_names,
        perspective_names,
    )
    perspective_set = _coerce_perspectives(
        values.shape[2],
        perspectives,
        source_perspective_names,
    )
    perspective_ids = perspective_set.ids
    perspective_labels = perspective_set.labels
    weights = _coerce_weights(perspective_weights, perspective_ids)
    reference_index = _resolve_reference_index(reference_perspective, perspective_ids)
    if tie_policy not in _TIE_POLICIES:
        raise_input_error("`tie_policy` must be 'first', 'split', or 'error'.")
    if not np.isfinite(tie_tolerance) or tie_tolerance < 0:
        raise_input_error("`tie_tolerance` must be finite and non-negative.")

    expected_net_benefits = np.mean(values, axis=0).T
    maxima = np.max(expected_net_benefits, axis=1, keepdims=True)
    optimal_masks = np.isclose(
        expected_net_benefits, maxima, rtol=0.0, atol=tie_tolerance
    )
    ties_detected = np.sum(optimal_masks, axis=1) > 1
    if tie_policy == "error" and np.any(ties_detected):
        raise_input_error("Tied expected-value optima detected.")
    optimal_indices = np.argmax(optimal_masks, axis=1).astype(int)
    optimal_enb = expected_net_benefits[
        np.arange(expected_net_benefits.shape[0]), optimal_indices
    ]

    regret = np.empty(
        (expected_net_benefits.shape[0], expected_net_benefits.shape[0]),
        dtype=DEFAULT_DTYPE,
    )
    for row_idx in range(expected_net_benefits.shape[0]):
        for column_idx in range(expected_net_benefits.shape[0]):
            strategy_idx = int(optimal_indices[column_idx])
            selected = (
                optimal_masks[column_idx]
                if tie_policy == "split"
                else np.arange(expected_net_benefits.shape[1]) == strategy_idx
            )
            regret[row_idx, column_idx] = max(
                0.0,
                float(
                    optimal_enb[row_idx]
                    - np.mean(expected_net_benefits[row_idx, selected])
                ),
            )

    switching_values = regret[:, reference_index].copy()
    consensus_enb_by_strategy = weights @ expected_net_benefits
    consensus_idx = int(np.argmax(consensus_enb_by_strategy))
    robust_enb_by_strategy = np.min(expected_net_benefits, axis=0)
    robust_idx = int(np.argmax(robust_enb_by_strategy))
    pareto_indices = _pareto_strategy_indices(expected_net_benefits)
    reference_mask = (
        optimal_masks[reference_index]
        if tie_policy == "split"
        else np.arange(expected_net_benefits.shape[1])
        == int(optimal_indices[reference_index])
    )
    switching_se: list[float] = []
    switching_ci95: list[list[float]] = []
    for perspective_index, own_mask in enumerate(optimal_masks):
        target_values = np.mean(values[:, own_mask, perspective_index], axis=1)
        reference_values = np.mean(values[:, reference_mask, perspective_index], axis=1)
        losses = target_values - reference_values
        standard_error = (
            float(np.std(losses, ddof=1) / np.sqrt(values.shape[0]))
            if values.shape[0] > 1
            else 0.0
        )
        switching_se.append(standard_error)
        estimate = float(switching_values[perspective_index])
        switching_ci95.append(
            [max(0.0, estimate - 1.96 * standard_error), estimate + 1.96 * standard_error]
        )

    return ValueOfPerspectiveResult(
        value=float(weights @ switching_values),
        perspective_ids=perspective_ids,
        perspective_labels=perspective_labels,
        strategy_names=final_strategy_names,
        expected_net_benefits=expected_net_benefits,
        optimal_strategy_indices=optimal_indices,
        optimal_strategy_names=[
            final_strategy_names[int(idx)] for idx in optimal_indices
        ],
        optimal_expected_net_benefits=optimal_enb,
        regret_matrix=regret,
        switching_values=switching_values,
        consensus_strategy_index=consensus_idx,
        consensus_strategy_name=final_strategy_names[consensus_idx],
        consensus_weighted_expected_net_benefit=float(
            consensus_enb_by_strategy[consensus_idx]
        ),
        robust_strategy_index=robust_idx,
        robust_strategy_name=final_strategy_names[robust_idx],
        pareto_strategy_indices=pareto_indices,
        pareto_strategy_names=[final_strategy_names[idx] for idx in pareto_indices],
        perspective_weights=weights,
        reference_perspective_id=perspective_ids[reference_index],
        method_maturity="fixture-backed",
        diagnostics={
            "n_samples": int(values.shape[0]),
            "n_strategies": int(values.shape[1]),
            "n_perspectives": int(values.shape[2]),
            "regret_definition": (
                "row i, column j is regret in perspective i when using the "
                "strategy optimal under perspective j"
            ),
            "estimand": "directional_current_information_evop",
            "decision_rule": "expected_value",
            "method_contract_version": METHOD_CONTRACT_VERSION,
            "tie_policy": tie_policy,
            "ties_detected": ties_detected.tolist(),
            "switching_standard_errors": switching_se,
            "switching_ci95": switching_ci95,
        },
        reporting=build_cheers_reporting(
            analysis_type="value_of_perspective",
            method_family="value_of_perspective",
            method_maturity="fixture-backed",
            perspective_ids=perspective_ids,
            perspective_labels=perspective_labels,
            diagnostics={
                "n_samples": int(values.shape[0]),
                "n_strategies": int(values.shape[1]),
                "n_perspectives": int(values.shape[2]),
            },
        ),
    )


def perspective_optimal_strategies(
    result: ValueOfPerspectiveResult,
) -> dict[str, str]:
    """Return the optimal strategy name for each perspective."""
    return dict(zip(result.perspective_ids, result.optimal_strategy_names, strict=True))
