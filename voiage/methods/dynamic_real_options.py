"""Dynamic real-options value of information analysis."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from voiage.config import DEFAULT_DTYPE
from voiage.exceptions import raise_input_error


@dataclass(frozen=True)
class DynamicRealOptionsResult:
    """Result envelope for staged, irreversible decision analysis."""

    expected_net_benefits: np.ndarray
    decision_stage_names: list[str]
    strategy_names: list[str]
    optimal_strategy_names: list[str]
    waiting_value: float
    option_value: float
    policy_path_regret: np.ndarray
    timing_sensitivity: np.ndarray
    robust_strategy_name: str
    pareto_strategy_names: list[str]
    method_maturity: str
    diagnostics: dict[str, object] = field(default_factory=dict)
    reporting: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ValueOfFlexibilityResult:
    """Experimental flexible-versus-commitment option-value result."""

    analysis_type: str
    method_maturity: str
    value_unit: str
    stage_semantics: str
    decision_stage_names: list[str]
    strategy_names: list[str]
    provenance: dict[str, str]
    flexible_value: float
    constrained_value: float
    value_of_flexibility: float
    flexible_policy_path: list[str]
    constrained_policy_path: list[str]
    commitment_baseline: str
    waiting_value: float | None
    option_value: float
    information_value_component: float
    decomposition_status: str
    exercise_decisions: None
    ordered_scenario_policy_changes: list[bool]
    policy_path_regret: np.ndarray
    diagnostics: dict[str, object] = field(default_factory=dict)
    reporting: dict[str, object] = field(default_factory=dict)


def _pareto_strategies(values: np.ndarray, names: Sequence[str]) -> list[str]:
    """Return strategies not weakly dominated across decision stages."""
    keep: list[str] = []
    for index, name in enumerate(names):
        dominated = False
        for other in range(values.shape[0]):
            if other == index:
                continue
            if np.all(values[other] >= values[index]) and np.any(
                values[other] > values[index]
            ):
                dominated = True
                break
        if not dominated:
            keep.append(str(name))
    return keep


def _named_values(
    net_benefits: np.ndarray,
    decision_stage_names: Sequence[str],
    strategy_names: Sequence[str],
) -> tuple[np.ndarray, list[str], list[str]]:
    values = np.asarray(net_benefits, dtype=DEFAULT_DTYPE)
    stages = [str(item) for item in decision_stage_names]
    strategies = [str(item) for item in strategy_names]
    if values.ndim != 3 or min(values.shape) < 1:
        raise_input_error("net_benefits must be a non-empty 3D array.")
    if values.shape[1] != len(strategies) or values.shape[2] != len(stages):
        raise_input_error("Names must match the strategy and stage dimensions.")
    if len(set(stages)) != len(stages) or len(set(strategies)) != len(strategies):
        raise_input_error("Decision stage and strategy names must be unique.")
    if not np.all(np.isfinite(values)):
        raise_input_error("net_benefits must contain only finite values.")
    return values, stages, strategies


def _exact_named_mapping(
    mapping: Mapping[str, float] | None,
    names: Sequence[str],
    *,
    label: str,
    default: np.ndarray,
) -> np.ndarray:
    if mapping is None:
        return default.astype(DEFAULT_DTYPE, copy=True)
    if set(mapping) != set(names):
        raise_input_error(f"{label} keys must exactly match decision_stage_names.")
    result = np.asarray([float(mapping[name]) for name in names], dtype=DEFAULT_DTYPE)
    if not np.all(np.isfinite(result)):
        raise_input_error(f"{label} must contain only finite values.")
    return result


def _adjusted_scenario_values(
    values: np.ndarray,
    stages: Sequence[str],
    stage_weights: Mapping[str, float] | None,
    discount_rate: float,
    irreversibility_penalty: float,
    lock_in_penalty: float,
    evidence_arrival_times: Mapping[str, float] | None,
    *,
    require_strict_arrival_times: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    scalar_controls = np.asarray(
        [discount_rate, irreversibility_penalty, lock_in_penalty],
        dtype=DEFAULT_DTYPE,
    )
    if not np.all(np.isfinite(scalar_controls)):
        raise_input_error("Rates and option penalties must be finite.")
    if discount_rate < 0 or irreversibility_penalty < 0 or lock_in_penalty < 0:
        raise_input_error("Rates and option penalties must be non-negative.")
    weights = _exact_named_mapping(
        stage_weights,
        stages,
        label="stage_weights",
        default=np.ones(len(stages), dtype=DEFAULT_DTYPE),
    )
    weight_sum = float(weights.sum())
    if np.any(weights < 0) or not np.isfinite(weight_sum) or weight_sum <= 0:
        raise_input_error(
            "stage_weights must be non-negative and have a finite positive sum."
        )
    weights /= weight_sum
    times = _exact_named_mapping(
        evidence_arrival_times,
        stages,
        label="evidence_arrival_times",
        default=np.arange(len(stages), dtype=DEFAULT_DTYPE),
    )
    if np.any(times < 0):
        raise_input_error("evidence_arrival_times must be finite and non-negative.")
    if (
        require_strict_arrival_times
        and evidence_arrival_times is not None
        and len(times) > 1
        and np.any(np.diff(times) <= 0)
    ):
        raise_input_error("evidence_arrival_times must be strictly increasing.")
    remaining = 1.0 - float(irreversibility_penalty) * times
    if np.any(remaining < -1e-12):
        raise_input_error(
            "irreversibility_penalty and evidence_arrival_times imply negative value."
        )
    discount = np.power(1.0 + float(discount_rate), -times)
    adjustment = discount * np.maximum(0.0, remaining)
    adjusted = np.mean(values, axis=0) * adjustment[None, :]
    adjusted[:, 1:] -= float(lock_in_penalty) * times[1:][None, :]
    if not np.all(np.isfinite(adjusted)):
        raise_input_error("Adjusted scenario values must be finite.")
    return adjusted, weights, times, discount


def _validated_provenance(provenance: Mapping[str, str]) -> dict[str, str]:
    """Validate the minimal deterministic provenance carried into results."""
    if set(provenance) != {"fixture_id", "execution_mode"}:
        raise_input_error(
            "provenance must contain exactly fixture_id and execution_mode."
        )
    fixture_id = provenance["fixture_id"]
    execution_mode = provenance["execution_mode"]
    if not isinstance(fixture_id, str) or not fixture_id.strip():
        raise_input_error("provenance.fixture_id must be a non-empty string.")
    if execution_mode != "deterministic":
        raise_input_error("provenance.execution_mode must be 'deterministic'.")
    return {
        "fixture_id": fixture_id.strip(),
        "execution_mode": execution_mode,
    }


def _policy_sets(
    stages: Sequence[str],
    strategies: Sequence[str],
    flexible_policy_sets: Mapping[str, Sequence[str]] | None,
    constrained_strategy_names: Sequence[str] | None,
) -> tuple[list[list[int]], list[int]]:
    strategy_index = {name: index for index, name in enumerate(strategies)}
    if flexible_policy_sets is None:
        flexible_names = {stage: list(strategies) for stage in stages}
    else:
        if set(flexible_policy_sets) != set(stages):
            raise_input_error(
                "flexible_policy_sets keys must exactly match decision_stage_names."
            )
        flexible_names = {
            stage: [str(name) for name in flexible_policy_sets[stage]]
            for stage in stages
        }
    flexible_indices: list[list[int]] = []
    for stage in stages:
        names = flexible_names[stage]
        if not names or len(set(names)) != len(names):
            raise_input_error("Each flexible policy set must be non-empty and unique.")
        unknown = [name for name in names if name not in strategy_index]
        if unknown:
            raise_input_error(f"Unknown flexible strategies for {stage}: {unknown}.")
        flexible_indices.append([strategy_index[name] for name in names])

    constrained = (
        list(strategies)
        if constrained_strategy_names is None
        else [str(name) for name in constrained_strategy_names]
    )
    if not constrained or len(set(constrained)) != len(constrained):
        raise_input_error("constrained_strategy_names must be non-empty and unique.")
    unknown = [name for name in constrained if name not in strategy_index]
    if unknown:
        raise_input_error(f"Unknown constrained strategies: {unknown}.")
    for name in constrained:
        if any(name not in flexible_names[stage] for stage in stages):
            raise_input_error(
                "Every constrained strategy must be feasible in every timing scenario."
            )
    return flexible_indices, [strategy_index[name] for name in constrained]


def _canonical_argmax(
    candidates: Sequence[int], values: np.ndarray, strategy_names: Sequence[str]
) -> tuple[int, list[str]]:
    """Select the lexicographically first strategy among numerical ties."""
    maximum = float(np.max(values))
    tolerance = 1e-12 * max(1.0, abs(maximum))
    tied = [
        candidate
        for candidate, value in zip(candidates, values, strict=True)
        if abs(float(value) - maximum) <= tolerance
    ]
    ordered = sorted(tied, key=lambda candidate: strategy_names[candidate])
    return ordered[0], [str(strategy_names[candidate]) for candidate in ordered]


def value_of_flexibility(
    net_benefits: np.ndarray,
    decision_stage_names: Sequence[str],
    strategy_names: Sequence[str],
    stage_weights: Mapping[str, float],
    provenance: Mapping[str, str],
    discount_rate: float = 0.0,
    irreversibility_penalty: float = 0.0,
    lock_in_penalty: float = 0.0,
    evidence_arrival_times: Mapping[str, float] | None = None,
    *,
    flexible_policy_sets: Mapping[str, Sequence[str]] | None = None,
    constrained_strategy_names: Sequence[str] | None = None,
    value_unit: str = "value-unit",
    stage_semantics: str = "timing_scenarios",
    information_value_included: bool = False,
) -> ValueOfFlexibilityResult:
    """Value scenario-contingent flexibility against ex-ante commitment.

    This experimental v1 estimator treats stages as mutually exclusive timing
    scenarios. It does not accept lifecycle-period aggregation or an embedded
    information-value component.
    """
    values, stages, strategies = _named_values(
        net_benefits, decision_stage_names, strategy_names
    )
    if stage_weights is None:
        raise_input_error("stage_weights must be declared for every timing scenario.")
    if not isinstance(value_unit, str) or not value_unit.strip():
        raise_input_error("value_unit must be a non-empty comparable unit.")
    if stage_semantics != "timing_scenarios":
        raise_input_error("stage_semantics must be 'timing_scenarios' for v1.")
    if information_value_included:
        raise_input_error(
            "information_value_included must be false to prevent double counting."
        )
    controls = np.asarray(
        [discount_rate, irreversibility_penalty, lock_in_penalty], dtype=DEFAULT_DTYPE
    )
    if not np.all(np.isfinite(controls)):
        raise_input_error("Rates and option penalties must be finite.")
    if np.any(controls != 0):
        raise_input_error(
            "Non-zero discount, irreversibility and lock-in controls are unsupported "
            "in the timing-scenario v1 contract until policy-dependent semantics "
            "and units are governed."
        )
    result_provenance = _validated_provenance(provenance)
    adjusted, weights, times, _discount = _adjusted_scenario_values(
        values,
        stages,
        stage_weights,
        discount_rate,
        irreversibility_penalty,
        lock_in_penalty,
        evidence_arrival_times,
    )
    flexible_indices, constrained_indices = _policy_sets(
        stages, strategies, flexible_policy_sets, constrained_strategy_names
    )

    flexible_path_indices: list[int] = []
    flexible_stage_values: list[float] = []
    flexible_stage_ties: dict[str, list[str]] = {}
    for stage_index, feasible in enumerate(flexible_indices):
        local = adjusted[feasible, stage_index]
        selected, tied_names = _canonical_argmax(feasible, local, strategies)
        flexible_path_indices.append(selected)
        flexible_stage_values.append(float(adjusted[selected, stage_index]))
        flexible_stage_ties[stages[stage_index]] = tied_names
    flexible_value = float(np.dot(weights, flexible_stage_values))

    commitment_values = adjusted[constrained_indices, :] @ weights
    commitment_index, commitment_ties = _canonical_argmax(
        constrained_indices, commitment_values, strategies
    )
    constrained_value = float(adjusted[commitment_index, :] @ weights)
    difference = flexible_value - constrained_value
    tolerance = 1e-12 * max(1.0, abs(flexible_value), abs(constrained_value))
    if difference < -tolerance:
        raise_input_error(
            "Flexible policy value is below its feasible commitment subset."
        )
    value = 0.0 if abs(difference) <= tolerance else float(difference)
    flexible_path = [strategies[index] for index in flexible_path_indices]
    commitment_name = strategies[commitment_index]
    constrained_path = [commitment_name] * len(stages)
    ordered_scenario_policy_changes = [False] + [
        flexible_path[index] != flexible_path[index - 1]
        for index in range(1, len(flexible_path))
    ]
    flexible_maxima = np.asarray(flexible_stage_values, dtype=DEFAULT_DTYPE)
    regret = np.maximum(0.0, flexible_maxima[None, :] - adjusted)
    return ValueOfFlexibilityResult(
        analysis_type="value_of_flexibility",
        method_maturity="experimental",
        value_unit=value_unit.strip(),
        stage_semantics=stage_semantics,
        decision_stage_names=stages,
        strategy_names=strategies,
        provenance=result_provenance,
        flexible_value=flexible_value,
        constrained_value=constrained_value,
        value_of_flexibility=value,
        flexible_policy_path=flexible_path,
        constrained_policy_path=constrained_path,
        commitment_baseline=commitment_name,
        waiting_value=None,
        option_value=value,
        information_value_component=0.0,
        decomposition_status="information-value-excluded",
        exercise_decisions=None,
        ordered_scenario_policy_changes=ordered_scenario_policy_changes,
        policy_path_regret=regret,
        diagnostics={
            "stage_weights": dict(zip(stages, weights.tolist(), strict=True)),
            "evidence_arrival_times": dict(zip(stages, times.tolist(), strict=True)),
            "discount_rate": float(discount_rate),
            "irreversibility_penalty": float(irreversibility_penalty),
            "lock_in_penalty": float(lock_in_penalty),
            "assurance": "exact-enumeration",
            "tie_policy": "canonical-lexicographic",
            "flexible_stage_ties": flexible_stage_ties,
            "commitment_ties": commitment_ties,
        },
        reporting={
            "analysis_type": "value_of_flexibility",
            "adjacent_to_information_value": True,
            "information_value_included": False,
            "method_maturity": "experimental",
            "tie_policy": "canonical-lexicographic",
        },
    )


def value_of_dynamic_real_options(
    net_benefits: np.ndarray,
    decision_stage_names: Sequence[str],
    strategy_names: Sequence[str],
    stage_weights: Mapping[str, float] | None = None,
    discount_rate: float = 0.0,
    irreversibility_penalty: float = 0.0,
    lock_in_penalty: float = 0.0,
    evidence_arrival_times: Mapping[str, float] | None = None,
    exercise_rules: Mapping[str, str] | None = None,
) -> DynamicRealOptionsResult:
    """Value staged evidence when decisions are delayed or irreversible.

    ``net_benefits`` has shape ``(samples, strategies, decision_stages)``.
    Stage weights represent uncertainty over when evidence becomes available;
    discounting and the two penalties are applied to later-stage values.
    """
    if exercise_rules:
        raise_input_error(
            "exercise_rules are not executable in v1; use explicit policy sets."
        )
    values, stages, strategies = _named_values(
        net_benefits, decision_stage_names, strategy_names
    )
    legacy_arrival_times = evidence_arrival_times
    if legacy_arrival_times is None:
        legacy_arrival_times = dict.fromkeys(stages, 0.0)
    adjusted, weights, times, discount = _adjusted_scenario_values(
        values,
        stages,
        stage_weights,
        discount_rate,
        irreversibility_penalty,
        lock_in_penalty,
        legacy_arrival_times,
        require_strict_arrival_times=False,
    )
    flexibility = value_of_flexibility(
        adjusted[None, :, :],
        stages,
        strategies,
        dict(zip(stages, weights.tolist(), strict=True)),
        {
            "fixture_id": "dynamic-real-options-compatibility",
            "execution_mode": "deterministic",
        },
    )
    # Preserve the legacy first-in-input tie presentation for this compatibility
    # envelope; the versioned Value of Flexibility surface uses canonical ties.
    optimal_names = [
        strategies[index] for index in np.argmax(adjusted, axis=0).tolist()
    ]
    option_value = flexibility.value_of_flexibility
    waiting_value = option_value
    regret = np.maximum(0.0, np.max(adjusted, axis=0)[None, :] - adjusted)
    robust_index = int(np.argmax(np.min(adjusted, axis=1)))
    return DynamicRealOptionsResult(
        expected_net_benefits=adjusted,
        decision_stage_names=stages,
        strategy_names=strategies,
        optimal_strategy_names=optimal_names,
        waiting_value=waiting_value,
        option_value=option_value,
        policy_path_regret=regret,
        timing_sensitivity=weights * discount,
        robust_strategy_name=strategies[robust_index],
        pareto_strategy_names=_pareto_strategies(adjusted, strategies),
        method_maturity="fixture-backed",
        diagnostics={
            "discount_rate": float(discount_rate),
            "irreversibility_penalty": float(irreversibility_penalty),
            "lock_in_penalty": float(lock_in_penalty),
            "evidence_arrival_times": dict(zip(stages, times.tolist(), strict=True)),
            "exercise_rules": {},
            "commitment_baseline": flexibility.commitment_baseline,
        },
        reporting={
            "reporting_standard": "CHEERS-VOI",
            "analysis_type": "value_of_dynamic_real_options",
            "method_maturity": "fixture-backed",
            "decision_stage_names": stages,
            "adjacent_estimand": "value_of_flexibility",
        },
    )
