"""Exact finite event-localized information value and density analysis.

The experimental evaluator implements Hazen's policy-relative expected-utility
information density on a finite probability-mass support.  It also evaluates a
declared binary event under perfect revelation and a symmetric imperfect binary
channel.  Monetary buying-price information remains delegated to the governed
expected-utility pricing family.
"""

# pyright: reportAny=false, reportExplicitAny=false, reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportImplicitStringConcatenation=false, reportUnknownLambdaType=false
# pyright: reportUnusedCallResult=false

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any, cast

from voiage.contracts.event_localized_information import (
    validate_event_localized_information_result_semantics,
    validate_event_localized_information_semantics,
)
from voiage.exceptions import raise_input_error

_ROOT_KEYS = {
    "schema_version",
    "analysis_id",
    "analysis_type",
    "value_unit",
    "direction",
    "chronology",
    "actions",
    "states",
    "density",
    "event",
    "tie_tolerance",
    "integral_tolerance",
    "provenance",
}
_STATE_KEYS = {"state_id", "probability", "coordinate", "action_values"}
_DENSITY_KEYS = {
    "measure",
    "coordinate_names",
    "coordinate_units",
    "base_coordinate",
    "reference_action",
}
_EVENT_KEYS = {
    "event_id",
    "definition",
    "information_cost",
    "accuracy_grid",
}
_PROVENANCE_KEYS = {"probability_source", "value_source", "event_source"}


@dataclass(frozen=True)
class EventLocalizedInformationResult:
    """Portable result envelope for exact event-localized information value."""

    payload: dict[str, Any]

    def to_contract_dict(self) -> dict[str, Any]:
        """Return an independent JSON-compatible result copy."""
        return cast("dict[str, Any]", json.loads(json.dumps(self.payload)))


def _finite(value: object, label: str) -> float:
    number = float(cast("Any", value))
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _nonempty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _exact_keys(record: object, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(record, dict) or set(record) != expected:
        actual = set(record) if isinstance(record, dict) else set()
        raise ValueError(
            f"{label} keys must match the v1 contract; "
            f"missing={sorted(expected - actual)}, unknown={sorted(actual - expected)}"
        )
    return cast("dict[str, Any]", record)


def _ties(values: dict[str, float], tolerance: float) -> list[str]:
    maximum = max(values.values())
    return sorted(
        action
        for action, value in values.items()
        if math.isclose(value, maximum, abs_tol=tolerance, rel_tol=0.0)
    )


def _clean(value: float, tolerance: float) -> float:
    if abs(value) <= tolerance:
        return 0.0
    return value


def _event_members(
    definition: object,
    states: list[dict[str, Any]],
    dimension: int,
) -> set[str]:
    if not isinstance(definition, dict):
        raise TypeError("event.definition must be an object")
    definition = cast("dict[str, Any]", definition)
    kind = definition.get("kind")
    if kind == "state_set":
        if set(definition) != {"kind", "state_ids"}:
            raise ValueError(
                "state_set event definition keys must match the v1 contract"
            )
        raw_ids = definition["state_ids"]
        if not isinstance(raw_ids, list) or not raw_ids:
            raise ValueError("state_set state_ids must be a non-empty array")
        ids = {_nonempty_string(value, "event state id") for value in raw_ids}
        known = {cast("str", state["state_id"]) for state in states}
        if not ids <= known:
            raise ValueError("state_set event references an unknown state")
        return ids
    if kind != "threshold" or set(definition) != {
        "kind",
        "coordinate_index",
        "operator",
        "threshold",
    }:
        raise ValueError("event.definition must be a strict threshold or state_set")
    index = definition["coordinate_index"]
    if (
        not isinstance(index, int)
        or isinstance(index, bool)
        or not 0 <= index < dimension
    ):
        raise ValueError("event coordinate_index is outside the coordinate dimension")
    threshold = _finite(definition["threshold"], "event threshold")
    operator = definition["operator"]
    predicates = {
        "less_than": lambda value: value < threshold,
        "less_than_or_equal": lambda value: value <= threshold,
        "greater_than": lambda value: value > threshold,
        "greater_than_or_equal": lambda value: value >= threshold,
    }
    if operator not in predicates:
        raise ValueError("event threshold operator is unsupported")
    predicate = predicates[cast("str", operator)]
    return {
        cast("str", state["state_id"])
        for state in states
        if predicate(cast("list[float]", state["coordinate"])[index])
    }


def _conditional_summary(
    states: list[dict[str, Any]],
    actions: list[str],
    weights: dict[str, float],
    tolerance: float,
) -> tuple[float, dict[str, float], list[str]]:
    probability = math.fsum(weights.values())
    if probability <= 0.0:
        raise ValueError("conditional event or signal probability must be positive")
    values = {
        action: math.fsum(
            weights[cast("str", state["state_id"])]
            * cast("dict[str, float]", state["action_values"])[action]
            for state in states
        )
        / probability
        for action in actions
    }
    return probability, values, _ties(values, tolerance)


def event_localized_information_value(
    specification: dict[str, object],
) -> EventLocalizedInformationResult:
    """Evaluate exact finite event VOI and policy-relative information density.

    Parameters
    ----------
    specification:
        Strict ``v1`` finite probability-mass specification.  ``direction`` is
        currently restricted to ``maximize`` so the policy-relative density is
        exactly ``f(x) [max_a g_a(x) - g_a*(x)]``.

    Returns
    -------
    EventLocalizedInformationResult
        Deterministic portable result with complete ties, perfect/imperfect
        event values, density integrals, modes, directions and assurance.
    """
    try:
        payload = cast("dict[str, Any]", json.loads(json.dumps(specification)))
        result = _evaluate(payload)
        validate_event_localized_information_semantics(payload)
        validate_event_localized_information_result_semantics(
            result, tolerance=max(float(payload["integral_tolerance"]), 1e-12)
        )
    except (ArithmeticError, KeyError, TypeError, ValueError) as error:
        raise_input_error(str(error))
    return EventLocalizedInformationResult(result)


def _evaluate(payload: dict[str, Any]) -> dict[str, Any]:
    _ = _exact_keys(payload, _ROOT_KEYS, "specification")
    if payload["schema_version"] != "v1":
        raise ValueError("schema_version must be 'v1'")
    if payload["analysis_type"] != "event_localized_information_value":
        raise ValueError("analysis_type must be event_localized_information_value")
    analysis_id = _nonempty_string(payload["analysis_id"], "analysis_id")
    value_unit = _nonempty_string(payload["value_unit"], "value_unit")
    if payload["direction"] != "maximize":
        raise ValueError("direction must be maximize in the v1 density contract")
    chronology = payload["chronology"]
    if (
        not isinstance(chronology, list)
        or len(chronology) < 3
        or any(not isinstance(stage, str) or not stage for stage in chronology)
    ):
        raise ValueError("chronology must contain at least three non-empty stages")
    actions_raw = payload["actions"]
    if not isinstance(actions_raw, list):
        raise TypeError("actions must be an array")
    actions = [_nonempty_string(action, "action") for action in actions_raw]
    if len(actions) < 2 or len(set(actions)) != len(actions):
        raise ValueError("actions must contain at least two unique identifiers")
    tie_tolerance = _finite(payload["tie_tolerance"], "tie_tolerance")
    integral_tolerance = _finite(payload["integral_tolerance"], "integral_tolerance")
    if not 0.0 <= tie_tolerance <= 1e-6:
        raise ValueError("tie_tolerance must lie in [0, 1e-6]")
    if not 0.0 < integral_tolerance <= 1e-6:
        raise ValueError("integral_tolerance must lie in (0, 1e-6]")

    density = _exact_keys(payload["density"], _DENSITY_KEYS, "density")
    if density["measure"] != "probability_mass":
        raise ValueError("density.measure must be probability_mass")
    names = density["coordinate_names"]
    units = density["coordinate_units"]
    base = density["base_coordinate"]
    if not isinstance(names, list) or not names:
        raise ValueError("coordinate_names must be a non-empty array")
    dimension = len(names)
    names = [_nonempty_string(value, "coordinate name") for value in names]
    if len(set(names)) != dimension:
        raise ValueError("coordinate names must be unique")
    if not isinstance(units, list) or len(units) != dimension:
        raise ValueError("coordinate_units must match the coordinate dimension")
    units = [_nonempty_string(value, "coordinate unit") for value in units]
    if not isinstance(base, list) or len(base) != dimension:
        raise ValueError("base_coordinate must match the coordinate dimension")
    base_coordinate = [_finite(value, "base coordinate") for value in base]

    states_raw = payload["states"]
    if not isinstance(states_raw, list) or not states_raw:
        raise ValueError("states must be a non-empty array")
    states: list[dict[str, Any]] = []
    for raw_state in states_raw:
        state = _exact_keys(raw_state, _STATE_KEYS, "state")
        state_id = _nonempty_string(state["state_id"], "state_id")
        probability = _finite(state["probability"], f"{state_id} probability")
        if probability <= 0.0:
            raise ValueError("state probabilities must be positive")
        coordinate = state["coordinate"]
        if not isinstance(coordinate, list) or len(coordinate) != dimension:
            raise ValueError(
                "each state coordinate must match the coordinate dimension"
            )
        coordinate = [_finite(value, f"{state_id} coordinate") for value in coordinate]
        values = state["action_values"]
        if not isinstance(values, dict) or set(values) != set(actions):
            raise ValueError(
                "each state action_values must contain exactly all actions"
            )
        action_values = {
            action: _finite(values[action], f"{state_id}.{action}")
            for action in actions
        }
        states.append(
            {
                "state_id": state_id,
                "probability": probability,
                "coordinate": coordinate,
                "action_values": action_values,
            }
        )
    state_ids = [cast("str", state["state_id"]) for state in states]
    if len(set(state_ids)) != len(state_ids):
        raise ValueError("state identifiers must be unique")
    if not math.isclose(
        math.fsum(cast("float", state["probability"]) for state in states),
        1.0,
        abs_tol=1e-12,
        rel_tol=0.0,
    ):
        raise ValueError("state probabilities must sum to one")

    expected = {
        action: math.fsum(
            cast("float", state["probability"])
            * cast("dict[str, float]", state["action_values"])[action]
            for state in states
        )
        for action in actions
    }
    baseline_ties = _ties(expected, tie_tolerance)
    reference_action = _nonempty_string(
        density["reference_action"], "density reference_action"
    )
    if reference_action not in actions:
        raise ValueError("density reference_action must identify a declared action")
    baseline_value = max(expected.values())
    if expected[reference_action] != baseline_value:
        raise ValueError("density reference_action must be exactly baseline-optimal")

    groups: dict[tuple[float, ...], list[dict[str, Any]]] = {}
    for state in states:
        groups.setdefault(tuple(cast("list[float]", state["coordinate"])), []).append(
            state
        )
    atoms: list[dict[str, Any]] = []
    raw_policy_densities: list[float] = []
    raw_centered_densities: list[float] = []
    for coordinate in sorted(groups):
        group = groups[coordinate]
        mass = math.fsum(cast("float", state["probability"]) for state in group)
        conditional = {
            action: math.fsum(
                cast("float", state["probability"])
                * cast("dict[str, float]", state["action_values"])[action]
                for state in group
            )
            / mass
            for action in actions
        }
        maximum = max(conditional.values())
        policy_relative = mass * (maximum - conditional[reference_action])
        centered = mass * (maximum - baseline_value)
        raw_policy_densities.append(policy_relative)
        raw_centered_densities.append(centered)
        atoms.append(
            {
                "coordinate": list(coordinate),
                "probability_mass": mass,
                "conditional_action_values": conditional,
                "optimal_actions": _ties(conditional, tie_tolerance),
                "reference_policy_value": conditional[reference_action],
                "policy_relative_density": _clean(policy_relative, integral_tolerance),
                "centered_density": _clean(centered, integral_tolerance),
            }
        )
    coordinate_information_value = (
        math.fsum(
            atom["probability_mass"] * max(atom["conditional_action_values"].values())
            for atom in atoms
        )
        - baseline_value
    )
    if coordinate_information_value < -integral_tolerance:
        raise ValueError("coordinate information value must be non-negative")
    policy_integral = math.fsum(raw_policy_densities)
    centered_integral = math.fsum(raw_centered_densities)
    policy_error = policy_integral - coordinate_information_value
    centered_error = centered_integral - coordinate_information_value
    if (
        abs(policy_error) > integral_tolerance
        or abs(centered_error) > integral_tolerance
    ):
        raise ValueError("information-density integral exceeds integral_tolerance")
    maximum_density = max(
        cast("float", atom["policy_relative_density"]) for atom in atoms
    )
    modes = (
        []
        if maximum_density <= integral_tolerance
        else [
            cast("list[float]", atom["coordinate"])
            for atom in atoms
            if math.isclose(
                cast("float", atom["policy_relative_density"]),
                maximum_density,
                abs_tol=tie_tolerance,
                rel_tol=0.0,
            )
        ]
    )

    event = _exact_keys(payload["event"], _EVENT_KEYS, "event")
    event_id = _nonempty_string(event["event_id"], "event_id")
    event_members = _event_members(event["definition"], states, dimension)
    probabilities = {
        cast("str", state["state_id"]): cast("float", state["probability"])
        for state in states
    }
    event_weights = {
        state_id: probability if state_id in event_members else 0.0
        for state_id, probability in probabilities.items()
    }
    complement_weights = {
        state_id: probability if state_id not in event_members else 0.0
        for state_id, probability in probabilities.items()
    }
    if (
        math.fsum(event_weights.values()) <= 0.0
        or math.fsum(complement_weights.values()) <= 0.0
    ):
        raise ValueError("event and complement must both have positive probability")
    event_probability, event_values, event_ties = _conditional_summary(
        states, actions, event_weights, tie_tolerance
    )
    complement_probability, complement_values, complement_ties = _conditional_summary(
        states, actions, complement_weights, tie_tolerance
    )
    perfect_informed_value = event_probability * max(
        event_values.values()
    ) + complement_probability * max(complement_values.values())
    perfect_gross = _clean(perfect_informed_value - baseline_value, integral_tolerance)
    if perfect_gross < -integral_tolerance:
        raise ValueError("perfect event information value must be non-negative")
    information_cost = _finite(event["information_cost"], "information cost")
    if information_cost < 0.0:
        raise ValueError("information cost must be non-negative")
    accuracy_grid = event["accuracy_grid"]
    if not isinstance(accuracy_grid, list) or not accuracy_grid:
        raise ValueError("accuracy_grid must be a non-empty array")
    accuracies = [_finite(value, "accuracy") for value in accuracy_grid]
    if any(value < 0.0 or value > 1.0 for value in accuracies):
        raise ValueError("accuracy values must lie in [0, 1]")
    if len(set(accuracies)) != len(accuracies):
        raise ValueError("accuracy_grid values must be unique")
    curve: list[dict[str, Any]] = []
    for accuracy in sorted(accuracies):
        report_event_weights = {
            state_id: probability
            * (accuracy if state_id in event_members else 1.0 - accuracy)
            for state_id, probability in probabilities.items()
        }
        report_complement_weights = {
            state_id: probability
            * (1.0 - accuracy if state_id in event_members else accuracy)
            for state_id, probability in probabilities.items()
        }
        report_event_probability, report_event_values, report_event_ties = (
            _conditional_summary(states, actions, report_event_weights, tie_tolerance)
        )
        (
            report_complement_probability,
            report_complement_values,
            report_complement_ties,
        ) = _conditional_summary(
            states, actions, report_complement_weights, tie_tolerance
        )
        informed_value = report_event_probability * max(
            report_event_values.values()
        ) + report_complement_probability * max(report_complement_values.values())
        gross = _clean(informed_value - baseline_value, integral_tolerance)
        if gross < -integral_tolerance:
            raise ValueError("imperfect event information value must be non-negative")
        curve.append(
            {
                "accuracy": accuracy,
                "signal_probabilities": {
                    "event_reported": report_event_probability,
                    "complement_reported": report_complement_probability,
                },
                "conditional_action_values": {
                    "event_reported": report_event_values,
                    "complement_reported": report_complement_values,
                },
                "optimal_actions": {
                    "event_reported": report_event_ties,
                    "complement_reported": report_complement_ties,
                },
                "informed_value": informed_value,
                "gross_voi": gross,
                "net_voi": gross - information_cost,
            }
        )
    symmetry_errors: list[float] = []
    for index, row in enumerate(curve):
        target = 1.0 - cast("float", row["accuracy"])
        counterpart = next(
            (
                candidate
                for candidate in curve[index + 1 :]
                if math.isclose(
                    cast("float", candidate["accuracy"]),
                    target,
                    abs_tol=1e-12,
                    rel_tol=0.0,
                )
            ),
            None,
        )
        if counterpart is not None:
            symmetry_errors.append(
                abs(
                    cast("float", row["gross_voi"])
                    - cast("float", counterpart["gross_voi"])
                )
            )
    maximum_symmetry_error = max(symmetry_errors) if symmetry_errors else None
    if (
        maximum_symmetry_error is not None
        and maximum_symmetry_error > integral_tolerance
    ):
        raise ValueError("binary-channel symmetry exceeds integral_tolerance")
    half_row = next(
        (
            row
            for row in curve
            if math.isclose(
                cast("float", row["accuracy"]), 0.5, abs_tol=1e-12, rel_tol=0.0
            )
        ),
        None,
    )
    half_residual = (
        cast("float", half_row["gross_voi"]) if half_row is not None else None
    )
    if half_residual is not None and abs(half_residual) > integral_tolerance:
        raise ValueError("accuracy 0.5 residual exceeds integral_tolerance")
    provenance = _exact_keys(payload["provenance"], _PROVENANCE_KEYS, "provenance")
    for key in sorted(_PROVENANCE_KEYS):
        _ = _nonempty_string(provenance[key], f"provenance.{key}")

    return {
        "schema_version": "v1",
        "analysis_id": analysis_id,
        "analysis_type": "event_localized_information_value_result",
        "method_maturity": "experimental",
        "value_unit": value_unit,
        "direction": "maximize",
        "chronology": chronology,
        "baseline": {
            "action_expected_values": expected,
            "optimal_actions": baseline_ties,
            "reference_action": reference_action,
            "reference_value": baseline_value,
        },
        "event": {
            "event_id": event_id,
            "definition": event["definition"],
            "partition_evidence": [
                {
                    "state_id": cast("str", state["state_id"]),
                    "coordinate": cast("list[float]", state["coordinate"]),
                    "event_member": cast("str", state["state_id"]) in event_members,
                }
                for state in sorted(
                    states, key=lambda value: cast("str", value["state_id"])
                )
            ],
            "state_ids": sorted(event_members),
            "complement_state_ids": sorted(set(state_ids) - event_members),
            "probability": event_probability,
            "complement_probability": complement_probability,
            "conditional_action_values": {
                "event": event_values,
                "complement": complement_values,
            },
            "optimal_actions": {"event": event_ties, "complement": complement_ties},
            "perfect_informed_value": perfect_informed_value,
            "perfect_gross_voi": perfect_gross,
            "information_cost": information_cost,
            "perfect_net_voi": perfect_gross - information_cost,
            "imperfect_binary_channel": curve,
        },
        "density": {
            "measure": "probability_mass",
            "formula": "f(x) * (max_a g_a(x) - g_reference(x))",
            "centered_diagnostic_formula": "f(x) * (max_a g_a(x) - V0)",
            "coordinate_names": names,
            "coordinate_units": units,
            "base_coordinate": base_coordinate,
            "reference_action": reference_action,
            "atoms": atoms,
            "information_value": _clean(
                coordinate_information_value, integral_tolerance
            ),
            "policy_relative_integral": _clean(policy_integral, integral_tolerance),
            "centered_integral": _clean(centered_integral, integral_tolerance),
            "integral_errors": {
                "policy_relative": _clean(policy_error, integral_tolerance),
                "centered": _clean(centered_error, integral_tolerance),
            },
            "modes": modes,
            "directions_from_base": [
                [
                    coordinate[index] - base_coordinate[index]
                    for index in range(dimension)
                ]
                for coordinate in modes
            ],
        },
        "assurance": {
            "estimator": "exact_finite_enumeration",
            "complete_ties": True,
            "event_complement_partition": True,
            "density_integral_within_tolerance": True,
            "maximum_binary_channel_symmetry_error": maximum_symmetry_error,
            "accuracy_half_no_information_residual": half_residual,
            "policy_relative_density_nonnegative": all(
                cast("float", atom["policy_relative_density"]) >= -integral_tolerance
                for atom in atoms
            ),
            "centered_density_is_signed_diagnostic": True,
            "bpi_delegated_to_issue_595": True,
            "continuous_density_claimed": False,
        },
        "provenance": provenance,
        "references": [
            {
                "doi": "10.1287/deca.2022.0465",
                "role": "policy-relative expected-utility information density",
            },
            {
                "doi": "10.1287/deca.2024.0172",
                "role": "perfect and imperfect tail-event information",
            },
        ],
        "language_dispositions": {
            "Python": "experimental_runtime",
            "Rust": "not_implemented",
            "R": "not_implemented",
            "Julia": "not_implemented",
            "Mojo": "external_upstream_boundary",
        },
    }


__all__ = ["EventLocalizedInformationResult", "event_localized_information_value"]
