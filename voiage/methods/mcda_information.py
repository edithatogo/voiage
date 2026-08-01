"""Exact finite additive-MCDA perfect-information evaluator."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
import math
from typing import TYPE_CHECKING, Any, cast

from voiage.contracts.mcda_information import (
    validate_mcda_information_result_semantics,
    validate_mcda_information_semantics,
)
from voiage.exceptions import raise_input_error

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


@dataclass(frozen=True)
class McdaInformationResult:
    """Portable result envelope for exact additive-MCDA information value."""

    payload: dict[str, Any]

    def to_contract_dict(self) -> dict[str, Any]:
        """Return an independent JSON-compatible copy of the result."""
        return cast("dict[str, Any]", json.loads(json.dumps(self.payload)))


def _is_tied(left: float, right: float, absolute: float, relative: float) -> bool:
    return math.isclose(left, right, abs_tol=absolute, rel_tol=relative)


def _ranking(
    scores: Mapping[str, float], absolute: float, relative: float
) -> list[dict[str, Any]]:
    ordered = sorted(scores, key=lambda item: (-scores[item], item))
    groups: list[dict[str, Any]] = []
    position = 1
    for alternative_id in ordered:
        score = scores[alternative_id]
        if groups and _is_tied(
            score, cast("float", groups[-1]["score"]), absolute, relative
        ):
            cast("list[str]", groups[-1]["alternative_ids"]).append(alternative_id)
            continue
        groups.append(
            {
                "rank": position,
                "alternative_ids": [alternative_id],
                "score": score,
            }
        )
        position += 1
    for index, group in enumerate(groups):
        group["rank"] = 1 + sum(
            len(cast("list[str]", previous["alternative_ids"]))
            for previous in groups[:index]
        )
    return groups


def _choice(ranking: Sequence[Mapping[str, Any]]) -> list[str]:
    return list(cast("list[str]", ranking[0]["alternative_ids"]))


def _linear_value(raw: float, criterion: Mapping[str, Any]) -> float:
    value_function = cast("Mapping[str, Any]", criterion["value_function"])
    anchors = cast("list[Mapping[str, Any]]", value_function["anchors"])
    domain = cast("list[float]", value_function["valid_domain"])
    if value_function["extrapolation_policy"] == "reject" and not (
        float(domain[0]) <= raw <= float(domain[1])
    ):
        raise ValueError("raw performance falls outside the fixed valid domain")
    raw_zero = float(anchors[0]["raw"])
    raw_one = float(anchors[1]["raw"])
    value_zero = float(anchors[0]["value"])
    value_one = float(anchors[1]["value"])
    return value_zero + (raw - raw_zero) * (value_one - value_zero) / (
        raw_one - raw_zero
    )


def _dominance(
    vectors: Mapping[str, Mapping[str, float]], tolerance: float
) -> tuple[list[dict[str, str]], list[str]]:
    alternatives = sorted(vectors)
    pairs: list[dict[str, str]] = []
    dominated: set[str] = set()
    for left in alternatives:
        for right in alternatives:
            if left == right:
                continue
            left_vector = vectors[left]
            right_vector = vectors[right]
            weak = all(
                left_vector[key] >= right_vector[key] - tolerance for key in left_vector
            )
            strict = any(
                left_vector[key] > right_vector[key] + tolerance for key in left_vector
            )
            if weak and strict:
                pairs.append({"dominant": left, "dominated": right})
                dominated.add(right)
    return pairs, [item for item in alternatives if item not in dominated]


def _partition_key(state: Mapping[str, Any], keys: Sequence[str]) -> tuple[str, ...]:
    values = cast("Mapping[str, str]", state["partition_values"])
    return tuple(values[key] for key in keys)


def mcda_information_value(
    specification: Mapping[str, object],
) -> McdaInformationResult:
    """Evaluate exact perfect information under a frozen additive MCDA model."""
    try:
        payload = cast(
            "dict[str, Any]", json.loads(json.dumps(specification, ensure_ascii=False))
        )
        validate_mcda_information_semantics(payload)
        result = _evaluate(payload)
        validate_mcda_information_result_semantics(result)
    except (ArithmeticError, KeyError, TypeError, ValueError) as error:
        raise_input_error(str(error))
    return McdaInformationResult(result)


def _evaluate(payload: Mapping[str, Any]) -> dict[str, Any]:
    alternatives = [
        cast("str", item["alternative_id"])
        for item in cast("list[Mapping[str, Any]]", payload["alternatives"])
    ]
    criteria_records = cast("list[Mapping[str, Any]]", payload["criteria"])
    criteria = [cast("str", item["criterion_id"]) for item in criteria_records]
    criterion_by_id = {
        cast("str", item["criterion_id"]): item for item in criteria_records
    }
    states = cast("list[Mapping[str, Any]]", payload["joint_states"])
    default_weights = cast("Mapping[str, float]", payload["default_weights"])
    tolerances = cast("Mapping[str, float]", payload["tolerances"])
    absolute = float(tolerances["absolute_tie"])
    relative = float(tolerances["relative_tie"])

    state_scores: dict[str, dict[str, float]] = {}
    state_vectors: dict[str, dict[str, dict[str, float]]] = {}
    probability_by_state: dict[str, float] = {}
    for state in states:
        state_id = cast("str", state["state_id"])
        probability_by_state[state_id] = float(state["probability"])
        weights = cast("Mapping[str, float]", state.get("weights", default_weights))
        performances = cast("Mapping[str, Mapping[str, float]]", state["performances"])
        vectors: dict[str, dict[str, float]] = {}
        scores: dict[str, float] = {}
        for alternative in alternatives:
            vector = {
                criterion: _linear_value(
                    float(performances[alternative][criterion]),
                    criterion_by_id[criterion],
                )
                for criterion in criteria
            }
            vectors[alternative] = vector
            score = math.fsum(float(weights[key]) * vector[key] for key in criteria)
            if not math.isfinite(score):
                raise ValueError("aggregate MCDA score must be finite")
            scores[alternative] = score
        state_vectors[state_id] = vectors
        state_scores[state_id] = scores

    expected_scores = {
        alternative: math.fsum(
            probability_by_state[cast("str", state["state_id"])]
            * state_scores[cast("str", state["state_id"])][alternative]
            for state in states
        )
        for alternative in alternatives
    }
    baseline_ranking = _ranking(expected_scores, absolute, relative)
    baseline_choice = _choice(baseline_ranking)
    baseline_value = max(expected_scores.values())

    action_results: list[dict[str, Any]] = []
    actions = cast("list[Mapping[str, Any]]", payload["information_actions"])
    for action in actions:
        action_id = cast("str", action["action_id"])
        keys = [
            *cast("list[str]", action["outcome_partition_keys"]),
            *cast("list[str]", action["preference_partition_keys"]),
        ]
        grouped: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
        for state in states:
            grouped[_partition_key(state, keys)].append(state)
        partitions: list[dict[str, Any]] = []
        policies: dict[str, list[str]] = {}
        for index, (values, members) in enumerate(grouped.items(), 1):
            probability = math.fsum(float(item["probability"]) for item in members)
            conditional_scores = {
                alternative: math.fsum(
                    float(item["probability"])
                    * state_scores[cast("str", item["state_id"])][alternative]
                    for item in members
                )
                / probability
                for alternative in alternatives
            }
            ranking = _ranking(conditional_scores, absolute, relative)
            choice = _choice(ranking)
            for member in members:
                policies[cast("str", member["state_id"])] = choice
            if action["action_type"] == "joint" or len(keys) != 1:
                partition_id = f"{action['action_type']}-{index}"
            else:
                key_label = keys[0].removesuffix("_regime").replace("_", "-")
                value_label = values[0].replace("_", "-")
                partition_id = f"{key_label}-{value_label}"
            partitions.append(
                {
                    "partition_id": partition_id,
                    "key_values": dict(zip(keys, values, strict=True)),
                    "probability": probability,
                    "conditional_scores": conditional_scores,
                    "ranking": ranking,
                    "choice_tie": choice,
                    "conditional_value": max(conditional_scores.values()),
                }
            )
        resolved_value = math.fsum(
            item["probability"] * item["conditional_value"] for item in partitions
        )
        gross_voi = resolved_value - baseline_value
        if gross_voi < 0:
            raise ValueError(
                "partition refinement produced negative gross information value"
            )
        cost = cast("Mapping[str, Any]", action["cost"])
        statewise_action_regret = [
            {
                "state_id": cast("str", state["state_id"]),
                "policy_tie": policies[cast("str", state["state_id"])],
                "regret": math.fsum(
                    max(state_scores[cast("str", state["state_id"])].values())
                    - state_scores[cast("str", state["state_id"])][alternative]
                    for alternative in policies[cast("str", state["state_id"])]
                )
                / len(policies[cast("str", state["state_id"])]),
            }
            for state in states
        ]
        expected_regret = math.fsum(
            probability_by_state[cast("str", item["state_id"])]
            * cast("float", item["regret"])
            for item in statewise_action_regret
        )
        action_results.append(
            {
                "action_id": action_id,
                "action_type": action["action_type"],
                "resolved_partition_keys": keys,
                "partitions": partitions,
                "resolved_value": resolved_value,
                "gross_voi": gross_voi,
                "cost": dict(cost),
                "net_voi": gross_voi - float(cost["aggregate_amount"]),
                "expected_regret": expected_regret,
                "statewise_regret": statewise_action_regret,
            }
        )
    by_type = {cast("str", action["action_type"]): action for action in action_results}
    criterion_action = by_type["criterion"]
    preference_action = by_type["preference"]
    joint_action = by_type["joint"]
    criterion_voi = float(criterion_action["gross_voi"])
    preference_voi = float(preference_action["gross_voi"])
    joint_voi = float(joint_action["gross_voi"])

    state_regret: list[dict[str, Any]] = []
    baseline_expected_regret = 0.0
    rank_acceptability = {
        alternative: [0.0 for _ in alternatives] for alternative in alternatives
    }
    state_tie_groups: dict[str, list[dict[str, Any]]] = {}
    pareto_statewise: list[dict[str, Any]] = []
    pareto_tolerance = float(tolerances["pareto_absolute"])
    for state in states:
        state_id = cast("str", state["state_id"])
        probability = probability_by_state[state_id]
        scores = state_scores[state_id]
        ranking = _ranking(scores, absolute, relative)
        state_tie_groups[state_id] = ranking
        optimum = max(scores.values())
        regret = math.fsum(
            optimum - scores[alternative] for alternative in baseline_choice
        ) / len(baseline_choice)
        baseline_expected_regret += probability * regret
        state_regret.append(
            {
                "state_id": state_id,
                "probability": probability,
                "optimal_tie": _choice(ranking),
                "baseline_policy_regret": regret,
            }
        )
        for group in ranking:
            members = cast("list[str]", group["alternative_ids"])
            start = cast("int", group["rank"]) - 1
            fraction = probability / len(members)
            for alternative in members:
                for position in range(start, start + len(members)):
                    rank_acceptability[alternative][position] += fraction
        dominance, non_dominated = _dominance(state_vectors[state_id], pareto_tolerance)
        pareto_statewise.append(
            {
                "state_id": state_id,
                "value_vectors": state_vectors[state_id],
                "dominance": dominance,
                "non_dominated": non_dominated,
            }
        )

    expected_vectors = {
        alternative: {
            criterion: math.fsum(
                probability_by_state[cast("str", state["state_id"])]
                * state_vectors[cast("str", state["state_id"])][alternative][criterion]
                for state in states
            )
            for criterion in criteria
        }
        for alternative in alternatives
    }
    expected_dominance, expected_non_dominated = _dominance(
        expected_vectors, pareto_tolerance
    )
    unsupported = [
        "AHP pairwise elicitation and consistency diagnostics",
        "outranking, vetoes and non-compensatory thresholds",
        "multiplicative, fuzzy, interval, robust or risk-sensitive aggregation",
        "endogenous or post-information normalization",
        "imperfect or sample information EVSI",
        "social-choice aggregation and endogenous feasible sets",
        "stable API or cross-language parity",
    ]
    return {
        "schema_version": "1.0.0",
        "analysis_id": payload["analysis_id"],
        "analysis_type": "mcda_perfect_information_result",
        "method_maturity": "experimental",
        "aggregate_unit": payload["aggregate_unit"],
        "alternative_ids": alternatives,
        "criterion_ids": criteria,
        "baseline": {
            "expected_scores": expected_scores,
            "ranking": baseline_ranking,
            "choice_tie": baseline_choice,
            "value": baseline_value,
        },
        "conditional_actions": action_results,
        "decomposition": {
            "criterion_action_id": criterion_action["action_id"],
            "preference_action_id": preference_action["action_id"],
            "joint_action_id": joint_action["action_id"],
            "criterion_gross_voi": criterion_voi,
            "preference_gross_voi": preference_voi,
            "joint_gross_voi": joint_voi,
            "interaction": joint_voi - criterion_voi - preference_voi,
            "joint_increment_over_criterion": joint_voi - criterion_voi,
            "joint_increment_over_preference": joint_voi - preference_voi,
            "no_double_counting_identity_residual": 0.0,
        },
        "regret": {
            "definition": "state_optimum_minus_policy_score",
            "baseline_expected": baseline_expected_regret,
            "statewise": state_regret,
        },
        "rank_acceptability": {
            "tie_convention": "fractional_complete_tie_groups",
            "by_alternative": rank_acceptability,
            "state_tie_groups": state_tie_groups,
        },
        "pareto": {
            "basis": "fixed_direction_normalized_criterion_values",
            "expectation_law": "submitted_joint_state_probabilities",
            "tie_tolerance": pareto_tolerance,
            "expected_value_vectors": expected_vectors,
            "expected_dominance": expected_dominance,
            "expected_non_dominated": expected_non_dominated,
            "statewise": pareto_statewise,
        },
        "assurance": {
            "estimator": "exact_finite_enumeration",
            "arithmetic": "binary64_with_declared_tolerances",
            "joint_dependence_preserved": True,
            "normalization_frozen_ex_ante": True,
            "gross_voi_clipped": False,
            "probabilities_reconciled": True,
            "weights_reconciled": True,
            "fixture_status": "analytically_reviewed_contract_fixture",
        },
        "language_dispositions": {
            "python": "executable",
            "rust": "unsupported",
            "r": "unsupported",
            "julia": "unsupported",
            "mojo": "external",
        },
        "unsupported_dispositions": unsupported,
    }


__all__ = ["McdaInformationResult", "mcda_information_value"]
