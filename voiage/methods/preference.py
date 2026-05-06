"""Preference heterogeneity and individualized care analysis.

This module treats decision preference as an explicit analysis dimension. It
compares the same strategies under multiple preference profiles and reports the
regret from applying one profile's decision rule under another profile.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from voiage.config import DEFAULT_DTYPE
from voiage.exceptions import raise_input_error
from voiage.reporting import build_cheers_reporting
from voiage.schema import ValueArray


@dataclass(frozen=True)
class PreferenceProfile:
    """Metadata for one preference profile.

    Parameters
    ----------
    id : str
        Stable machine-readable preference-profile identifier.
    label : str, optional
        Human-readable label. Defaults to ``id`` when omitted.
    weight : float, default=1.0
        Non-negative preference-profile weight used for weighted summaries.
    utility_weights : mapping[str, float], optional
        Utility weights used to construct profile-specific net benefit.
    equity_weights : mapping[str, float], optional
        Equity weights used to construct profile-specific net benefit.
    willingness_to_pay : float, optional
        Willingness-to-pay threshold for this preference profile.
    discount_rate : float, optional
        Discount rate for this preference profile.
    population : float, optional
        Population scaling factor for this preference profile.
    stakeholder_metadata : mapping[str, object], optional
        Additional stakeholder or profile metadata.
    """

    id: str
    label: str | None = None
    weight: float = 1.0
    utility_weights: Mapping[str, float] = field(default_factory=dict)
    equity_weights: Mapping[str, float] = field(default_factory=dict)
    willingness_to_pay: float | None = None
    discount_rate: float | None = None
    population: float | None = None
    stakeholder_metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate preference-profile metadata."""
        if not self.id or not str(self.id).strip():
            raise_input_error("Preference profile id must be a non-empty string.")
        object.__setattr__(self, "id", str(self.id))
        object.__setattr__(self, "label", self.label or self.id)
        weight = float(self.weight)
        if not np.isfinite(weight) or weight < 0.0:
            raise_input_error(
                "Preference profile weight must be finite and non-negative."
            )
        object.__setattr__(self, "weight", weight)
        object.__setattr__(
            self,
            "utility_weights",
            {str(key): float(value) for key, value in self.utility_weights.items()},
        )
        object.__setattr__(
            self,
            "equity_weights",
            {str(key): float(value) for key, value in self.equity_weights.items()},
        )
        object.__setattr__(
            self,
            "stakeholder_metadata",
            dict(self.stakeholder_metadata),
        )

        for name, value in (
            ("willingness_to_pay", self.willingness_to_pay),
            ("discount_rate", self.discount_rate),
            ("population", self.population),
        ):
            if value is not None and not np.isfinite(float(value)):
                raise_input_error(f"Preference profile {name} must be finite.")
        if self.population is not None and float(self.population) <= 0.0:
            raise_input_error("Preference profile population must be positive.")


@dataclass(frozen=True)
class PreferenceProfileSet:
    """Ordered collection of unique preference profiles."""

    profiles: tuple[PreferenceProfile, ...]

    def __init__(self, profiles: Sequence[PreferenceProfile | str]):
        """Create a PreferenceProfileSet from profile objects or ids."""
        if not profiles:
            raise_input_error("At least one preference profile is required.")

        normalized = tuple(
            item
            if isinstance(item, PreferenceProfile)
            else PreferenceProfile(id=str(item))
            for item in profiles
        )
        ids = [item.id for item in normalized]
        if len(set(ids)) != len(ids):
            raise_input_error("Preference profile ids must be unique.")

        object.__setattr__(self, "profiles", normalized)

    @property
    def ids(self) -> list[str]:
        """Return preference-profile identifiers in analysis order."""
        return [item.id for item in self.profiles]

    @property
    def labels(self) -> list[str]:
        """Return preference-profile labels in analysis order."""
        return [str(item.label) for item in self.profiles]

    @property
    def weights(self) -> list[float]:
        """Return the unnormalized preference-profile weights."""
        return [float(item.weight) for item in self.profiles]


@dataclass(frozen=True)
class PreferenceHeterogeneityResult:
    """Structured preference heterogeneity result.

    Attributes
    ----------
    analysis_id : str
        Stable analysis identifier for the preference analysis.
    decision_problem_id : str
        Stable decision-problem identifier for the preference analysis.
    value : float
        Weighted switching value relative to the reference preference profile.
    individualized_care_value : float
        Weighted value of tailoring strategy choice to each preference profile.
    preference_profile_ids : list[str]
        Preference-profile identifiers in analysis order.
    preference_profile_labels : list[str]
        Human-readable preference-profile labels.
    strategy_names : list[str]
        Strategy labels in analysis order.
    expected_net_benefits : numpy.ndarray
        Expected net benefit with shape ``(n_profiles, n_strategies)``.
    optimal_strategy_indices : numpy.ndarray
        Preference-profile-specific optimal strategy index.
    optimal_strategy_names : list[str]
        Preference-profile-specific optimal strategy name.
    optimal_expected_net_benefits : numpy.ndarray
        Expected net benefit of each preference-profile-specific optimum.
    regret_matrix : numpy.ndarray
        Matrix where row ``i`` and column ``j`` is regret in profile ``i``
        when using the strategy optimal under profile ``j``.
    switching_values : numpy.ndarray
        Regret avoided by switching away from the reference profile's strategy
        toward each profile's own optimal strategy.
    consensus_strategy_index : int
        Strategy maximizing weighted expected net benefit across profiles.
    consensus_strategy_name : str
        Name of the consensus strategy.
    consensus_weighted_expected_net_benefit : float
        Weighted expected net benefit of the consensus strategy.
    individualized_care_strategy_index : int
        Strategy maximizing weighted expected net benefit when tailoring by
        preference profile.
    individualized_care_strategy_name : str
        Name of the individualized-care strategy summary.
    individualized_care_weighted_expected_net_benefit : float
        Weighted expected net benefit of the individualized-care strategy.
    robust_strategy_index : int
        Strategy maximizing the minimum expected net benefit across profiles.
    robust_strategy_name : str
        Name of the robust maximin strategy.
    pareto_strategy_indices : list[int]
        Strategy indices not dominated across profiles.
    pareto_strategy_names : list[str]
        Strategy names not dominated across profiles.
    preference_profile_weights : numpy.ndarray
        Normalized preference-profile weights used for consensus and value
        summaries.
    reference_preference_profile : str
        Preference profile used as the reference decision rule.
    method_maturity : str
        Maturity label for the fixture-backed preference surface.
    diagnostics : dict[str, object]
        Deterministic diagnostics for downstream reporting.
    """

    analysis_id: str
    decision_problem_id: str
    value: float
    individualized_care_value: float
    preference_profile_ids: list[str]
    preference_profile_labels: list[str]
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
    individualized_care_strategy_index: int
    individualized_care_strategy_name: str
    individualized_care_weighted_expected_net_benefit: float
    robust_strategy_index: int
    robust_strategy_name: str
    pareto_strategy_indices: list[int]
    pareto_strategy_names: list[str]
    preference_profile_weights: np.ndarray
    reference_preference_profile: str
    method_maturity: str
    diagnostics: dict[str, object]
    reporting: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-compatible payload."""
        return {
            "analysis_id": self.analysis_id,
            "decision_problem_id": self.decision_problem_id,
            "analysis_type": "value_of_preference_information",
            "method_maturity": self.method_maturity,
            "value": float(self.value),
            "individualized_care_value": float(self.individualized_care_value),
            "preference_profile_ids": list(self.preference_profile_ids),
            "strategy_names": list(self.strategy_names),
            "expected_net_benefits": np.asarray(self.expected_net_benefits).tolist(),
            "optimal_strategy_by_preference_profile": dict(
                self.optimal_strategy_by_preference_profile
            ),
            "regret_matrix": np.asarray(self.regret_matrix).tolist(),
            "switching_values": dict(
                zip(
                    self.preference_profile_ids,
                    np.asarray(self.switching_values).tolist(),
                    strict=True,
                )
            ),
            "consensus_strategy": self.consensus_strategy,
            "robust_strategy": self.robust_strategy,
            "pareto_strategies": list(self.pareto_strategy_names),
            "reference_preference_profile": self.reference_preference_profile,
            "reporting": dict(self.reporting),
            "diagnostics": dict(self.diagnostics),
        }

    @property
    def optimal_strategy_by_preference_profile(self) -> dict[str, str]:
        """Map each preference profile to its optimal strategy name."""
        return dict(
            zip(
                self.preference_profile_ids,
                self.optimal_strategy_names,
                strict=True,
            )
        )

    @property
    def consensus_strategy(self) -> str:
        """Return the consensus strategy name."""
        return self.consensus_strategy_name

    @property
    def robust_strategy(self) -> str:
        """Return the robust strategy name."""
        return self.robust_strategy_name

    @property
    def pareto_strategies(self) -> list[str]:
        """Return the non-dominated strategy names."""
        return list(self.pareto_strategy_names)


def _coerce_net_benefits(
    net_benefits: ValueArray | np.ndarray,
    strategy_names: Sequence[str] | None,
    preference_profile_names: Sequence[str] | None,
) -> tuple[np.ndarray, list[str], list[str] | None]:
    """Normalize net-benefit input to a finite 3D NumPy array."""
    if isinstance(net_benefits, ValueArray):
        values = np.asarray(net_benefits.numpy_values, dtype=DEFAULT_DTYPE)
        final_strategy_names = (
            list(strategy_names)
            if strategy_names is not None
            else net_benefits.strategy_names
        )
        final_profile_names = (
            list(preference_profile_names)
            if preference_profile_names is not None
            else net_benefits.perspective_names
        )
    elif isinstance(net_benefits, np.ndarray):
        values = np.asarray(net_benefits, dtype=DEFAULT_DTYPE)
        if values.ndim != 3:
            raise_input_error(
                "Preference heterogeneity requires 3D net benefits "
                "(samples x strategies x preference_profiles)."
            )
        n_strategies = values.shape[1]
        final_strategy_names = (
            list(strategy_names)
            if strategy_names is not None
            else [f"Strategy {idx}" for idx in range(n_strategies)]
        )
        final_profile_names = (
            list(preference_profile_names)
            if preference_profile_names is not None
            else None
        )
    else:
        raise_input_error("`net_benefits` must be a ValueArray or NumPy array.")

    if values.ndim != 3:
        raise_input_error(
            "Preference heterogeneity requires 3D net benefits "
            "(samples x strategies x preference_profiles)."
        )
    if values.shape[0] < 1 or values.shape[1] < 1 or values.shape[2] < 1:
        raise_input_error(
            "At least one sample, one strategy, and one preference profile are required."
        )
    if not np.all(np.isfinite(values)):
        raise_input_error("Net-benefit values must be finite.")
    if len(final_strategy_names) != values.shape[1]:
        raise_input_error(
            "`strategy_names` length must match the number of strategies."
        )
    if final_profile_names is not None and len(final_profile_names) != values.shape[2]:
        raise_input_error(
            "`preference_profile_names` length must match the number of preference profiles."
        )

    return values, [str(item) for item in final_strategy_names], final_profile_names


def _coerce_profiles(
    n_profiles: int,
    preference_profiles: PreferenceProfileSet
    | Sequence[PreferenceProfile | str]
    | None,
    preference_profile_names: Sequence[str] | None,
) -> PreferenceProfileSet:
    """Normalize preference-profile metadata."""
    if preference_profiles is not None:
        profile_set = (
            preference_profiles
            if isinstance(preference_profiles, PreferenceProfileSet)
            else PreferenceProfileSet(preference_profiles)
        )
    elif preference_profile_names is not None:
        profile_set = PreferenceProfileSet(
            [str(name) for name in preference_profile_names]
        )
    else:
        profile_set = PreferenceProfileSet(
            [f"Preference profile {idx}" for idx in range(n_profiles)]
        )

    if len(profile_set.profiles) != n_profiles:
        raise_input_error(
            "`preference_profiles` length must match the number of profiles."
        )
    return profile_set


def _coerce_weights(
    weights: Sequence[float] | Mapping[str, float] | None,
    profile_ids: Sequence[str],
    default_weights: Sequence[float],
) -> np.ndarray:
    """Normalize profile weights to a finite probability vector."""
    n_profiles = len(profile_ids)
    if weights is None:
        weight_arr = np.asarray(default_weights, dtype=DEFAULT_DTYPE)
    elif isinstance(weights, Mapping):
        missing = [item for item in profile_ids if item not in weights]
        if missing:
            raise_input_error(
                "`preference_profile_weights` must include every profile id."
            )
        weight_arr = np.asarray(
            [float(weights[item]) for item in profile_ids], dtype=DEFAULT_DTYPE
        )
    else:
        weight_arr = np.asarray(weights, dtype=DEFAULT_DTYPE)

    if weight_arr.ndim != 1 or len(weight_arr) != n_profiles:
        raise_input_error(
            "`preference_profile_weights` must contain one weight per profile."
        )
    if not np.all(np.isfinite(weight_arr)):
        raise_input_error("`preference_profile_weights` must be finite.")
    if np.any(weight_arr < 0):
        raise_input_error("`preference_profile_weights` must be non-negative.")

    total = float(np.sum(weight_arr))
    if total <= 0:
        raise_input_error("`preference_profile_weights` must sum to a positive value.")
    return weight_arr / total


def _resolve_reference_index(
    reference_preference_profile: str | int | None,
    profile_ids: Sequence[str],
) -> int:
    """Return the reference preference-profile index."""
    if reference_preference_profile is None:
        return 0
    if isinstance(reference_preference_profile, int):
        if reference_preference_profile < 0 or reference_preference_profile >= len(
            profile_ids
        ):
            raise_input_error("`reference_preference_profile` index is out of range.")
        return reference_preference_profile
    try:
        return list(profile_ids).index(reference_preference_profile)
    except ValueError:
        raise_input_error(
            "`reference_preference_profile` must identify a preference profile."
        )


def _pareto_strategy_indices(expected_net_benefits: np.ndarray) -> list[int]:
    """Return strategy indices not dominated across preference profiles."""
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


def value_of_preference(
    net_benefits: ValueArray | np.ndarray,
    preference_profiles: PreferenceProfileSet
    | Sequence[PreferenceProfile | str]
    | None = None,
    strategy_names: Sequence[str] | None = None,
    preference_profile_names: Sequence[str] | None = None,
    preference_profile_weights: Sequence[float] | Mapping[str, float] | None = None,
    reference_preference_profile: str | int | None = None,
    analysis_id: str | None = None,
    decision_problem_id: str | None = None,
    decision_context: str | None = None,
) -> PreferenceHeterogeneityResult:
    r"""Compare decision value across multiple preference profiles.

    Parameters
    ----------
    net_benefits : ValueArray or numpy.ndarray
        Net-benefit samples with shape
        ``(n_samples, n_strategies, n_preference_profiles)``.
    preference_profiles : PreferenceProfileSet or sequence, optional
        Ordered preference-profile metadata. Plain strings are interpreted as
        preference-profile ids.
    strategy_names : sequence of str, optional
        Strategy labels.
    preference_profile_names : sequence of str, optional
        Preference-profile labels used when ``preference_profiles`` is omitted.
    preference_profile_weights : sequence or mapping, optional
        Non-negative weights used for consensus and weighted switching value.
        Mappings must be keyed by profile id.
    reference_preference_profile : str or int, optional
        Preference profile whose optimal strategy is used as the reference
        decision rule. Defaults to the first profile.

    Returns
    -------
    PreferenceHeterogeneityResult
        Profile-specific optima, regret matrix, switching values, consensus
        and robust strategies, Pareto strategy set, and individualized-care
        summary.

    Notes
    -----
    The profile-specific optimum is

    .. math::

       d^*_p = \arg\max_d E[NB_{d,p}].

    The reported switching value compares the reference profile's optimal
    strategy with each profile's own optimum, and the returned value is the
    weighted average of those switching values. The individualized-care value
    compares the same profile-specific optima with a single consensus strategy.

    Examples
    --------
    >>> import numpy as np
    >>> from voiage.analysis import DecisionAnalysis
    >>> values = np.array(
    ...     [
    ...         [[10.0, 7.0], [8.0, 11.0], [5.0, 9.0]],
    ...         [[10.0, 7.0], [8.0, 11.0], [5.0, 9.0]],
    ...     ]
    ... )
    >>> result = DecisionAnalysis(values).value_of_preference(
    ...     preference_profile_names=["access_first", "outcomes_first"],
    ...     strategy_names=["A", "B", "C"],
    ... )
    >>> result.reference_preference_profile
    'access_first'
    """
    values, final_strategy_names, source_profile_names = _coerce_net_benefits(
        net_benefits,
        strategy_names,
        preference_profile_names,
    )
    profile_set = _coerce_profiles(
        values.shape[2],
        preference_profiles,
        source_profile_names,
    )
    profile_ids = profile_set.ids
    profile_labels = profile_set.labels
    weights = _coerce_weights(
        preference_profile_weights,
        profile_ids,
        profile_set.weights,
    )
    reference_index = _resolve_reference_index(
        reference_preference_profile,
        profile_ids,
    )

    expected_net_benefits = np.mean(values, axis=0).T
    optimal_indices = np.argmax(expected_net_benefits, axis=1).astype(int)
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
            regret[row_idx, column_idx] = max(
                0.0,
                float(
                    optimal_enb[row_idx] - expected_net_benefits[row_idx, strategy_idx]
                ),
            )

    switching_values = regret[:, reference_index].copy()
    consensus_enb_by_strategy = weights @ expected_net_benefits
    consensus_idx = int(np.argmax(consensus_enb_by_strategy))
    robust_enb_by_strategy = np.min(expected_net_benefits, axis=0)
    robust_idx = int(np.argmax(robust_enb_by_strategy))
    pareto_indices = _pareto_strategy_indices(expected_net_benefits)

    individualized_care_enb = float(weights @ optimal_enb)
    individualized_care_profile_gain = np.maximum(
        0.0, optimal_enb - expected_net_benefits[:, consensus_idx]
    )
    reporting_payload = build_cheers_reporting(
        analysis_type="value_of_preference_information",
        method_family="value_of_preference_information",
        method_maturity="fixture-backed",
        analysis_id=analysis_id,
        decision_problem_id=decision_problem_id,
        decision_context=decision_context,
        diagnostics={
            "n_samples": int(values.shape[0]),
            "n_strategies": int(values.shape[1]),
            "n_preference_profiles": int(values.shape[2]),
        },
        estimator="profile_mean",
    )
    reporting_payload["preference_profile_ids"] = profile_ids
    reporting_payload["preference_profile_labels"] = profile_labels

    return PreferenceHeterogeneityResult(
        analysis_id=analysis_id or "preference-screening-001",
        decision_problem_id=decision_problem_id or "screening-program-001",
        value=float(weights @ switching_values),
        individualized_care_value=float(weights @ individualized_care_profile_gain),
        preference_profile_ids=profile_ids,
        preference_profile_labels=profile_labels,
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
        individualized_care_strategy_index=-1,
        individualized_care_strategy_name="Profile-specific optima",
        individualized_care_weighted_expected_net_benefit=individualized_care_enb,
        robust_strategy_index=robust_idx,
        robust_strategy_name=final_strategy_names[robust_idx],
        pareto_strategy_indices=pareto_indices,
        pareto_strategy_names=[final_strategy_names[idx] for idx in pareto_indices],
        preference_profile_weights=weights,
        reference_preference_profile=profile_ids[reference_index],
        method_maturity="fixture-backed",
        diagnostics={
            "n_samples": int(values.shape[0]),
            "n_strategies": int(values.shape[1]),
            "n_preference_profiles": int(values.shape[2]),
            "regret_definition": (
                "row i, column j is regret in profile i when using the strategy "
                "optimal under profile j"
            ),
            "reference_preference_profile": profile_ids[reference_index],
            "individualized_care_weighted_expected_net_benefit": individualized_care_enb,
        },
        reporting=reporting_payload,
    )


def preference_optimal_strategies(
    result: PreferenceHeterogeneityResult,
) -> dict[str, str]:
    """Map each preference profile to its preferred strategy name."""
    if not isinstance(result, PreferenceHeterogeneityResult):
        raise_input_error("`result` must be a PreferenceHeterogeneityResult.")
    return dict(
        zip(result.preference_profile_ids, result.optimal_strategy_names, strict=True)
    )


def value_of_preference_heterogeneity(
    net_benefits: ValueArray | np.ndarray,
    preference_profiles: PreferenceProfileSet
    | Sequence[PreferenceProfile | str]
    | None = None,
    strategy_names: Sequence[str] | None = None,
    preference_profile_names: Sequence[str] | None = None,
    preference_profile_weights: Sequence[float] | Mapping[str, float] | None = None,
    reference_preference_profile: str | int | None = None,
    analysis_id: str | None = None,
    decision_problem_id: str | None = None,
    decision_context: str | None = None,
) -> PreferenceHeterogeneityResult:
    """Alias for :func:`value_of_preference`."""
    return value_of_preference(
        net_benefits,
        preference_profiles=preference_profiles,
        strategy_names=strategy_names,
        preference_profile_names=preference_profile_names,
        preference_profile_weights=preference_profile_weights,
        reference_preference_profile=reference_preference_profile,
        analysis_id=analysis_id,
        decision_problem_id=decision_problem_id,
        decision_context=decision_context,
    )


def value_of_preference_information(
    net_benefits: ValueArray | np.ndarray,
    preference_profiles: PreferenceProfileSet
    | Sequence[PreferenceProfile | str]
    | None = None,
    strategy_names: Sequence[str] | None = None,
    preference_profile_names: Sequence[str] | None = None,
    preference_profile_weights: Sequence[float] | Mapping[str, float] | None = None,
    reference_preference_profile: str | int | None = None,
    analysis_id: str | None = None,
    decision_problem_id: str | None = None,
    decision_context: str | None = None,
) -> PreferenceHeterogeneityResult:
    """Alias for :func:`value_of_preference` using the contract wording."""
    return value_of_preference(
        net_benefits,
        preference_profiles=preference_profiles,
        strategy_names=strategy_names,
        preference_profile_names=preference_profile_names,
        preference_profile_weights=preference_profile_weights,
        reference_preference_profile=reference_preference_profile,
        analysis_id=analysis_id,
        decision_problem_id=decision_problem_id,
        decision_context=decision_context,
    )
