"""Exact finite decision-value optimization of information-source sequences."""

# pyright: reportAny=false, reportExplicitAny=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations, permutations
import json
import math
from typing import TYPE_CHECKING, Any, cast

from voiage.contracts.information_source_portfolio import (
    validate_information_source_portfolio_result,
    validate_information_source_portfolio_semantics,
)
from voiage.exceptions import raise_input_error

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


@dataclass(frozen=True)
class InformationSourcePortfolioResult:
    """Portable result envelope for exact finite source-portfolio VOI."""

    payload: dict[str, Any]

    def to_contract_dict(self) -> dict[str, Any]:
        """Return an independent JSON-compatible result copy."""
        return cast("dict[str, Any]", json.loads(json.dumps(self.payload)))


def _ties(values: Mapping[str, float], absolute: float, relative: float) -> list[str]:
    best = max(values.values())
    return sorted(
        key
        for key, value in values.items()
        if math.isclose(value, best, abs_tol=absolute, rel_tol=relative)
    )


def information_source_portfolio_value(
    specification: Mapping[str, object],
) -> InformationSourcePortfolioResult:
    """Optimize a bounded source sequence under one declared joint-world law."""
    try:
        payload = cast("dict[str, Any]", json.loads(json.dumps(specification)))
        validate_information_source_portfolio_semantics(payload)
        result = _evaluate(payload)
        validate_information_source_portfolio_result(result)
    except (ArithmeticError, KeyError, TypeError, ValueError) as error:
        raise_input_error(str(error))
    return InformationSourcePortfolioResult(result)


def _resolved(
    sequence: Sequence[str],
    states: Sequence[Mapping[str, Any]],
    actions: Sequence[str],
    baseline_tie: Sequence[str],
    absolute: float,
    relative: float,
) -> tuple[float, list[dict[str, Any]]]:
    grouped: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for state in states:
        observations = cast("Mapping[str, str]", state["source_observations"])
        grouped[tuple(observations[source] for source in sequence)].append(state)
    partitions: list[dict[str, Any]] = []
    for index, observation_tuple in enumerate(sorted(grouped), 1):
        members = grouped[observation_tuple]
        probability = math.fsum(float(member["probability"]) for member in members)
        conditional = {
            action: math.fsum(
                float(member["probability"])
                * float(cast("Mapping[str, float]", member["action_values"])[action])
                for member in members
            )
            / probability
            for action in actions
        }
        action_tie = _ties(conditional, absolute, relative)
        partitions.append(
            {
                "partition_id": f"observation-{index}",
                "observations": dict(zip(sequence, observation_tuple, strict=True)),
                "probability": probability,
                "conditional_action_values": conditional,
                "action_tie": action_tie,
                "conditional_value": max(conditional.values()),
                "switch_from_baseline": action_tie != list(baseline_tie),
            }
        )
    resolved = math.fsum(
        float(partition["probability"]) * float(partition["conditional_value"])
        for partition in partitions
    )
    return resolved, partitions


def _sequence_is_feasible(
    sequence: Sequence[str],
    source_by_id: Mapping[str, Mapping[str, Any]],
    constraints: Mapping[str, Any],
) -> tuple[bool, str | None]:
    selected = set(sequence)
    for index, source_id in enumerate(sequence):
        source = source_by_id[source_id]
        if any(excluded in selected for excluded in cast("list[str]", source["excludes"])):
            return False, "exclusivity"
        for successor in cast("list[str]", source["must_precede"]):
            if successor in selected and index >= sequence.index(successor):
                return False, "ordering"
    sources = [source_by_id[source_id] for source_id in sequence]
    if math.fsum(float(source["cost"]) for source in sources) > float(constraints["max_cost"]):
        return False, "cost"
    if math.fsum(float(source["latency"]) for source in sources) > float(constraints["max_latency"]):
        return False, "latency"
    if math.fsum(float(source["privacy_cost"]) for source in sources) > float(constraints["max_privacy_cost"]):
        return False, "privacy"
    if any(float(source["sla_probability"]) < float(constraints["min_source_sla"]) for source in sources):
        return False, "sla"
    if any(float(source["freshness_age"]) > float(constraints["max_freshness_age"]) for source in sources):
        return False, "freshness"
    coverage = set().union(*(set(cast("list[str]", source["coverage"])) for source in sources))
    if not set(cast("list[str]", constraints["required_coverage"])) <= coverage:
        return False, "coverage"
    return True, None


def _evaluate(payload: Mapping[str, Any]) -> dict[str, Any]:
    actions = sorted(cast("list[str]", payload["actions"]))
    states = sorted(cast("list[Mapping[str, Any]]", payload["states"]), key=lambda item: cast("str", item["state_id"]))
    sources = sorted(cast("list[Mapping[str, Any]]", payload["sources"]), key=lambda item: cast("str", item["source_id"]))
    source_by_id = {cast("str", source["source_id"]): source for source in sources}
    source_ids = sorted(source_by_id)
    constraints = cast("Mapping[str, Any]", payload["constraints"])
    tie_policy = cast("Mapping[str, float]", payload["tie_policy"])
    absolute = float(tie_policy["absolute_tolerance"])
    relative = float(tie_policy["relative_tolerance"])
    expected = {
        action: math.fsum(
            float(state["probability"])
            * float(cast("Mapping[str, float]", state["action_values"])[action])
            for state in states
        )
        for action in actions
    }
    baseline_value = max(expected.values())
    baseline_tie = _ties(expected, absolute, relative)
    value_context = cast("Mapping[str, Any]", payload["value_context"])
    delay_rate = float(value_context["delay_cost_per_time"])
    raw_cache: dict[tuple[str, ...], tuple[float, list[dict[str, Any]]]] = {}

    def raw(sequence: tuple[str, ...]) -> tuple[float, list[dict[str, Any]]]:
        if sequence not in raw_cache:
            raw_cache[sequence] = _resolved(sequence, states, actions, baseline_tie, absolute, relative)
        return raw_cache[sequence]

    evaluations: list[dict[str, Any]] = []
    total_candidates = 0
    prune_reasons: dict[str, int] = defaultdict(int)
    max_sources = min(int(constraints["max_sources"]), len(source_ids))
    for length in range(1, max_sources + 1):
        for sequence in permutations(source_ids, length):
            total_candidates += 1
            feasible, reason = _sequence_is_feasible(sequence, source_by_id, constraints)
            if not feasible:
                prune_reasons[cast("str", reason)] += 1
                continue
            evaluations.append(
                _sequence_evaluation(sequence, source_by_id, raw, baseline_value, delay_rate)
            )
    if not evaluations:
        raise ValueError("portfolio constraints leave no feasible non-empty source sequence")
    evaluations.sort(key=lambda item: tuple(cast("list[str]", item["source_sequence"])))
    best_net = max(float(item["net_value"]) for item in evaluations)
    tied = [
        item
        for item in evaluations
        if math.isclose(float(item["net_value"]), best_net, abs_tol=absolute, rel_tol=relative)
    ]
    tied.sort(key=lambda item: (float(item["total_source_cost"]), float(item["total_latency"]), tuple(cast("list[str]", item["source_sequence"]))))
    selected = dict(tied[0])
    selected["optimal_sequence_tie"] = [cast("list[str]", item["source_sequence"]) for item in tied]
    sequence = tuple(cast("list[str]", selected["source_sequence"]))
    attribution = _shapley(sequence, raw, baseline_value)
    switches = [partition for partition in cast("list[dict[str, Any]]", selected["partitions"]) if partition["switch_from_baseline"]]
    return {
        "schema_version": "1.0.0",
        "analysis_id": payload["analysis_id"],
        "analysis_type": "information_source_portfolio_result",
        "method_maturity": "experimental",
        "value_context": dict(value_context),
        "baseline": {"expected_action_values": expected, "action_tie": baseline_tie, "value": baseline_value},
        "evaluated_sequences": evaluations,
        "optimum": selected,
        "conditional_marginals": selected["conditional_marginals"],
        "attribution": attribution,
        "switches": switches,
        "assurance": {
            "solver": "exact_exhaustive_bounded_sequences",
            "total_candidate_sequences": total_candidates,
            "feasible_sequences": len(evaluations),
            "pruned_sequences": total_candidates - len(evaluations),
            "prune_reasons": dict(sorted(prune_reasons.items())),
            "approximation_used": False,
            "optimality_gap": 0.0,
            "independent_additive_evsi_used": False,
            "joint_world_dependence": "declared_complete_joint_world_law",
            "attribution_scope": "decision_value_not_predictive_data_shapley",
        },
        "provenance": {
            **dict(cast("Mapping[str, Any]", payload["provenance"])),
            "source_receipts": [
                {"source_id": source_id, **dict(cast("Mapping[str, Any]", source_by_id[source_id]["rights"]))}
                for source_id in source_ids
            ],
        },
        "language_dispositions": {
            "python": "experimental_exact_runtime",
            "rust": "unsupported",
            "r": "unsupported",
            "julia": "unsupported",
            "mojo": "external_boundary",
        },
        "unsupported_dispositions": [
            "adaptive_stopping_and_branching_acquisition",
            "probabilistic_observation_channels_outside_joint_worlds",
            "approximate_portfolio_solver",
            "predictive_data_shapley",
            "stable_or_polyglot_execution",
        ],
    }


def _sequence_evaluation(
    sequence: tuple[str, ...],
    source_by_id: Mapping[str, Mapping[str, Any]],
    raw: Any,
    baseline_value: float,
    delay_rate: float,
) -> dict[str, Any]:
    resolved_value, partitions = raw(sequence)
    gross = resolved_value - baseline_value
    sources = [source_by_id[source_id] for source_id in sequence]
    source_cost = math.fsum(float(source["cost"]) for source in sources)
    latency = math.fsum(float(source["latency"]) for source in sources)
    privacy = math.fsum(float(source["privacy_cost"]) for source in sources)
    delay_cost = latency * delay_rate
    marginals: list[dict[str, Any]] = []
    previous_gross = 0.0
    for position, source_id in enumerate(sequence, 1):
        prefix = sequence[:position]
        prefix_value, _ = raw(prefix)
        prefix_gross = prefix_value - baseline_value
        source = source_by_id[source_id]
        incremental_delay = float(source["latency"]) * delay_rate
        gross_marginal = prefix_gross - previous_gross
        marginals.append(
            {
                "position": position,
                "source_id": source_id,
                "conditioning_sources": list(sequence[: position - 1]),
                "gross_marginal_value": gross_marginal,
                "incremental_source_cost": float(source["cost"]),
                "incremental_delay_cost": incremental_delay,
                "net_marginal_value": gross_marginal - float(source["cost"]) - incremental_delay,
            }
        )
        previous_gross = prefix_gross
    return {
        "source_sequence": list(sequence),
        "total_source_cost": source_cost,
        "total_latency": latency,
        "total_privacy_cost": privacy,
        "delay_cost": delay_cost,
        "resolved_value": resolved_value,
        "gross_value": gross,
        "willingness_to_pay": gross - delay_cost,
        "net_value": gross - delay_cost - source_cost,
        "partitions": partitions,
        "conditional_marginals": marginals,
    }


def _shapley(
    selected: tuple[str, ...],
    raw: Any,
    baseline_value: float,
) -> list[dict[str, Any]]:
    members = tuple(sorted(selected))
    count = len(members)
    denominator = math.factorial(count)

    def value(coalition: tuple[str, ...]) -> float:
        if not coalition:
            return 0.0
        return float(raw(tuple(sorted(coalition)))[0]) - baseline_value

    result: list[dict[str, Any]] = []
    for source_id in members:
        others = tuple(item for item in members if item != source_id)
        contribution = 0.0
        for size in range(len(others) + 1):
            weight = math.factorial(size) * math.factorial(count - size - 1) / denominator
            for coalition in combinations(others, size):
                contribution += weight * (value((*coalition, source_id)) - value(coalition))
        result.append({"source_id": source_id, "gross_attribution": contribution, "attribution_method": "exact_decision_value_shapley"})
    return result
