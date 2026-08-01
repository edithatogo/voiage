"""Exact finite static and dynamic value-of-heterogeneity decomposition."""

# pyright: reportAny=false, reportExplicitAny=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import TYPE_CHECKING, Any, cast

from voiage.contracts.heterogeneity_value import (
    validate_heterogeneity_value_result,
    validate_heterogeneity_value_semantics,
)
from voiage.exceptions import raise_input_error

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass(frozen=True)
class HeterogeneityValueDecompositionResult:
    """Portable experimental result for static/dynamic heterogeneity value."""

    payload: dict[str, Any]

    def to_contract_dict(self) -> dict[str, Any]:
        """Return an independent JSON-compatible result copy."""
        return cast("dict[str, Any]", json.loads(json.dumps(self.payload)))


def heterogeneity_value_decomposition(
    specification: Mapping[str, object],
) -> HeterogeneityValueDecompositionResult:
    """Evaluate C0, Cf, P0, Pf and an optional sample-information extension."""
    try:
        payload = cast("dict[str, Any]", json.loads(json.dumps(specification)))
        validate_heterogeneity_value_semantics(payload)
        result = _evaluate(payload)
        validate_heterogeneity_value_result(result)
    except (ArithmeticError, KeyError, TypeError, ValueError) as error:
        raise_input_error(str(error))
    return HeterogeneityValueDecompositionResult(result)


def _best(values: Mapping[str, float], direction: str) -> float:
    return max(values.values()) if direction == "maximize" else min(values.values())


def _selected(values: Mapping[str, float], direction: str) -> str:
    optimum = _best(values, direction)
    return min(key for key, value in values.items() if value == optimum)


def _ties(
    values: Mapping[str, float], direction: str, absolute: float, relative: float
) -> list[str]:
    optimum = _best(values, direction)
    return sorted(
        key
        for key, value in values.items()
        if math.isclose(value, optimum, abs_tol=absolute, rel_tol=relative)
    )


def _gain(new: float, old: float, direction: str) -> float:
    return new - old if direction == "maximize" else old - new


def _evaluate(data: dict[str, Any]) -> dict[str, Any]:
    direction = str(data["objective"]["direction"])
    absolute = float(data["tolerances"]["absolute_tie"])
    relative = float(data["tolerances"]["relative_tie"])
    groups = sorted(data["subgroups"], key=lambda item: item["subgroup_id"])
    states = sorted(data["states"], key=lambda item: item["state_id"])
    probabilities = {
        str(item["state_id"]): float(item["probability"]) for item in states
    }
    weights = {str(item["subgroup_id"]): float(item["weight"]) for item in groups}
    eligible = {
        str(item["subgroup_id"]): sorted(item["eligible_action_ids"]) for item in groups
    }
    common_actions = sorted(
        set.intersection(*(set(items) for items in eligible.values()))
    )

    def value(state: dict[str, Any], group_id: str, action_id: str) -> float:
        return float(state["subgroup_action_values"][group_id][action_id])

    current_group_values: dict[str, dict[str, float]] = {}
    for group_id, action_ids in eligible.items():
        current_group_values[group_id] = {
            action_id: math.fsum(
                probabilities[str(state["state_id"])]
                * value(state, group_id, action_id)
                for state in states
            )
            for action_id in action_ids
        }
    current_common_values = {
        action_id: math.fsum(
            weights[group_id] * current_group_values[group_id][action_id]
            for group_id in weights
        )
        for action_id in common_actions
    }
    c0 = _best(current_common_values, direction)
    cf = math.fsum(
        weights[group_id] * _best(current_group_values[group_id], direction)
        for group_id in weights
    )

    perfect_common_by_state: dict[str, dict[str, float]] = {}
    perfect_group_values: dict[str, dict[str, float]] = {
        group_id: {} for group_id in weights
    }
    perfect_state_audit: list[dict[str, Any]] = []
    for state in states:
        state_id = str(state["state_id"])
        common_values = {
            action_id: math.fsum(
                weights[group_id] * value(state, group_id, action_id)
                for group_id in weights
            )
            for action_id in common_actions
        }
        perfect_common_by_state[state_id] = common_values
        group_policies: dict[str, Any] = {}
        for group_id, action_ids in eligible.items():
            action_values = {
                action_id: value(state, group_id, action_id) for action_id in action_ids
            }
            best_value = _best(action_values, direction)
            perfect_group_values[group_id][state_id] = best_value
            group_policies[group_id] = {
                "action_values": action_values,
                "action_tie": _ties(action_values, direction, absolute, relative),
                "selected_action_id": _selected(action_values, direction),
                "value": best_value,
            }
        perfect_state_audit.append(
            {
                "state_id": state_id,
                "probability": probabilities[state_id],
                "population_common": {
                    "action_values": common_values,
                    "action_tie": _ties(common_values, direction, absolute, relative),
                    "selected_action_id": _selected(common_values, direction),
                    "value": _best(common_values, direction),
                },
                "subgroup_policies": group_policies,
            }
        )
    p0 = math.fsum(
        probabilities[state_id] * _best(values, direction)
        for state_id, values in perfect_common_by_state.items()
    )
    pf = math.fsum(
        weights[group_id]
        * math.fsum(
            probabilities[state_id] * state_value
            for state_id, state_value in state_values.items()
        )
        for group_id, state_values in perfect_group_values.items()
    )
    static = _gain(cf, c0, direction)
    dynamic = _gain(pf, p0, direction)
    evpi0 = _gain(p0, c0, direction)
    evpif = _gain(pf, cf, direction)

    subgroup_results: list[dict[str, Any]] = []
    for group in groups:
        group_id = str(group["subgroup_id"])
        perfect_value = math.fsum(
            probabilities[state_id] * state_value
            for state_id, state_value in perfect_group_values[group_id].items()
        )
        current_value = _best(current_group_values[group_id], direction)
        subgroup_results.append(
            {
                "subgroup_id": group_id,
                "label": group["label"],
                "weight": weights[group_id],
                "current_action_values": current_group_values[group_id],
                "current_action_tie": _ties(
                    current_group_values[group_id], direction, absolute, relative
                ),
                "selected_current_action_id": _selected(
                    current_group_values[group_id], direction
                ),
                "current_value": current_value,
                "perfect_information_value": perfect_value,
                "evpi": _gain(perfect_value, current_value, direction),
            }
        )

    sample_result = _sample_extension(
        data,
        groups,
        states,
        probabilities,
        weights,
        eligible,
        common_actions,
        current_group_values,
        c0,
        cf,
        direction,
        absolute,
        relative,
    )
    residual = dynamic - static - (evpif - evpi0)
    if abs(residual) <= 1e-12:  # pragma: no branch - algebraic identity
        residual = 0.0
    return {
        "schema_version": "1.0.0",
        "analysis_id": data["analysis_id"],
        "analysis_type": "heterogeneity_value_decomposition_result",
        "method_maturity": "experimental",
        "objective": data["objective"],
        "four_value_decomposition": {
            "c0": c0,
            "cf": cf,
            "p0": p0,
            "pf": pf,
            "static_value": static,
            "dynamic_value": dynamic,
            "identity_residual": residual,
        },
        "perfect_information": {
            "population_common_evpi": evpi0,
            "subgroup_policy_evpi": evpif,
            "difference_identity": evpif - evpi0,
        },
        "subgroup_results": subgroup_results,
        "sample_information": sample_result,
        "policy_audit": {
            "current_population_common": {
                "action_values": current_common_values,
                "action_tie": _ties(
                    current_common_values, direction, absolute, relative
                ),
                "selected_action_id": _selected(current_common_values, direction),
            },
            "perfect_information_states": perfect_state_audit,
            "subgroup_specification": data["subgroup_specification"],
        },
        "assurance": {
            **data["estimator_assurance"],
            "states_evaluated": len(states),
            "subgroups_evaluated": len(groups),
            "common_actions_evaluated": len(common_actions),
            "identity_verified": residual == 0.0,
            "selection_adjustment_performed": False,
            "sparse_subgroup_inference_performed": False,
        },
        "provenance": data["provenance"],
        "language_dispositions": {
            "python": "experimental_exact_execution",
            "rust": "not_implemented",
            "r": "not_implemented",
            "julia": "not_implemented",
            "mojo": "external_upstream_boundary",
        },
    }


def _sample_extension(
    data: dict[str, Any],
    groups: list[dict[str, Any]],
    states: list[dict[str, Any]],
    probabilities: dict[str, float],
    weights: dict[str, float],
    eligible: dict[str, list[str]],
    common_actions: list[str],
    current_group_values: dict[str, dict[str, float]],
    c0: float,
    cf: float,
    direction: str,
    absolute: float,
    relative: float,
) -> dict[str, Any] | None:
    sample = data["sample_information"]
    if sample is None:
        return None

    def value(state: dict[str, Any], group_id: str, action_id: str) -> float:
        return float(state["subgroup_action_values"][group_id][action_id])

    signal_audit: list[dict[str, Any]] = []
    s0 = 0.0
    sf = 0.0
    for signal in sorted(sample["signals"], key=lambda item: item["signal_id"]):
        signal_id = str(signal["signal_id"])
        joint = {
            str(state["state_id"]): probabilities[str(state["state_id"])]
            * float(signal["likelihood_by_state"][str(state["state_id"])])
            for state in states
        }
        marginal = math.fsum(joint.values())
        common_values = {
            action_id: math.fsum(
                joint[str(state["state_id"])]
                * math.fsum(
                    weights[str(group["subgroup_id"])]
                    * value(state, str(group["subgroup_id"]), action_id)
                    for group in groups
                )
                for state in states
            )
            for action_id in common_actions
        }
        common_contribution = _best(common_values, direction)
        s0 += common_contribution
        group_audit: dict[str, Any] = {}
        subgroup_contribution = 0.0
        for group in groups:
            group_id = str(group["subgroup_id"])
            action_values = {
                action_id: math.fsum(
                    joint[str(state["state_id"])] * value(state, group_id, action_id)
                    for state in states
                )
                for action_id in eligible[group_id]
            }
            selected_value = _best(action_values, direction)
            subgroup_contribution += weights[group_id] * selected_value
            group_audit[group_id] = {
                "joint_weighted_action_values": action_values,
                "action_tie": _ties(action_values, direction, absolute, relative),
                "selected_action_id": _selected(action_values, direction),
            }
        sf += subgroup_contribution
        signal_audit.append(
            {
                "signal_id": signal_id,
                "probability": marginal,
                "population_common": {
                    "joint_weighted_action_values": common_values,
                    "action_tie": _ties(common_values, direction, absolute, relative),
                    "selected_action_id": _selected(common_values, direction),
                },
                "subgroup_policies": group_audit,
            }
        )
    evsi0 = _gain(s0, c0, direction)
    evsif = _gain(sf, cf, direction)
    sample_segmentation = _gain(sf, s0, direction)
    subgroup_evsi = []
    for group_id in sorted(weights):
        sample_value = math.fsum(
            _best(
                signal["subgroup_policies"][group_id]["joint_weighted_action_values"],
                direction,
            )
            for signal in signal_audit
        )
        current_value = _best(current_group_values[group_id], direction)
        subgroup_evsi.append(
            {
                "subgroup_id": group_id,
                "weight": weights[group_id],
                "current_value": current_value,
                "sample_value": sample_value,
                "evsi": _gain(sample_value, current_value, direction),
                "weighted_evsi_contribution": weights[group_id]
                * _gain(sample_value, current_value, direction),
            }
        )
    cost = float(sample["cost"]["amount"])
    identity = sample_segmentation - _gain(cf, c0, direction) - (evsif - evsi0)
    if abs(identity) <= 1e-12:  # pragma: no branch - algebraic identity
        identity = 0.0
    return {
        "research_action_id": sample["research_action_id"],
        "s0": s0,
        "sf": sf,
        "population_common_evsi": evsi0,
        "subgroup_policy_evsi": evsif,
        "sample_informed_segmentation_value": sample_segmentation,
        "identity_residual": identity,
        "cost": sample["cost"],
        "population_common_net_evsi": evsi0 - cost,
        "subgroup_policy_net_evsi": evsif - cost,
        "signals": signal_audit,
        "subgroup_evsi": subgroup_evsi,
        "current_subgroup_action_values": current_group_values,
    }
