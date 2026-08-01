"""Strict portable contract for experimental event-localized information."""

# pyright: reportAny=false, reportExplicitAny=false, reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportMissingModuleSource=false
# pyright: reportUnknownLambdaType=false

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any, Final, cast

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

_TEXT: Final[dict[str, object]] = {"type": "string", "minLength": 1}
_NUMBER: Final[dict[str, object]] = {"type": "number"}
_TOLERANCE_MAX = 1e-6

_STATE: Final[dict[str, object]] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["state_id", "probability", "coordinate", "action_values"],
    "properties": {
        "state_id": _TEXT,
        "probability": {"type": "number", "exclusiveMinimum": 0, "maximum": 1},
        "coordinate": {"type": "array", "minItems": 1, "items": _NUMBER},
        "action_values": {
            "type": "object",
            "minProperties": 2,
            "additionalProperties": _NUMBER,
        },
    },
}
_THRESHOLD: Final[dict[str, object]] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["kind", "coordinate_index", "operator", "threshold"],
    "properties": {
        "kind": {"const": "threshold"},
        "coordinate_index": {"type": "integer", "minimum": 0},
        "operator": {
            "enum": [
                "less_than",
                "less_than_or_equal",
                "greater_than",
                "greater_than_or_equal",
            ]
        },
        "threshold": _NUMBER,
    },
}
_STATE_SET: Final[dict[str, object]] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["kind", "state_ids"],
    "properties": {
        "kind": {"const": "state_set"},
        "state_ids": {
            "type": "array",
            "minItems": 1,
            "uniqueItems": True,
            "items": _TEXT,
        },
    },
}
_PROVENANCE: Final[dict[str, object]] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["probability_source", "value_source", "event_source"],
    "properties": {
        "probability_source": _TEXT,
        "value_source": _TEXT,
        "event_source": _TEXT,
    },
}

EVENT_LOCALIZED_INFORMATION_INPUT_SCHEMA_V1: Final[dict[str, object]] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://voiage.dev/specs/frontier/event-localized-information/v1/input.schema.json",
    "title": "EventLocalizedInformationInputV1",
    "type": "object",
    "additionalProperties": False,
    "required": [
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
    ],
    "properties": {
        "schema_version": {"const": "v1"},
        "analysis_id": _TEXT,
        "analysis_type": {"const": "event_localized_information_value"},
        "value_unit": _TEXT,
        "direction": {"const": "maximize"},
        "chronology": {
            "type": "array",
            "minItems": 3,
            "items": _TEXT,
        },
        "actions": {
            "type": "array",
            "minItems": 2,
            "uniqueItems": True,
            "items": _TEXT,
        },
        "states": {"type": "array", "minItems": 1, "items": _STATE},
        "density": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "measure",
                "coordinate_names",
                "coordinate_units",
                "base_coordinate",
                "reference_action",
            ],
            "properties": {
                "measure": {"const": "probability_mass"},
                "coordinate_names": {
                    "type": "array",
                    "minItems": 1,
                    "uniqueItems": True,
                    "items": _TEXT,
                },
                "coordinate_units": {
                    "type": "array",
                    "minItems": 1,
                    "items": _TEXT,
                },
                "base_coordinate": {
                    "type": "array",
                    "minItems": 1,
                    "items": _NUMBER,
                },
                "reference_action": _TEXT,
            },
        },
        "event": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "event_id",
                "definition",
                "information_cost",
                "accuracy_grid",
            ],
            "properties": {
                "event_id": _TEXT,
                "definition": {"oneOf": [_THRESHOLD, _STATE_SET]},
                "information_cost": {"type": "number", "minimum": 0},
                "accuracy_grid": {
                    "type": "array",
                    "minItems": 1,
                    "uniqueItems": True,
                    "items": {"type": "number", "minimum": 0, "maximum": 1},
                },
            },
        },
        "tie_tolerance": {
            "type": "number",
            "minimum": 0,
            "maximum": _TOLERANCE_MAX,
        },
        "integral_tolerance": {
            "type": "number",
            "exclusiveMinimum": 0,
            "maximum": _TOLERANCE_MAX,
        },
        "provenance": _PROVENANCE,
    },
}

_STRING_ARRAY: Final[dict[str, object]] = {
    "type": "array",
    "minItems": 1,
    "uniqueItems": True,
    "items": _TEXT,
}
_NUMBER_ARRAY: Final[dict[str, object]] = {
    "type": "array",
    "minItems": 1,
    "items": _NUMBER,
}
_NUMBER_MAP: Final[dict[str, object]] = {
    "type": "object",
    "minProperties": 2,
    "additionalProperties": _NUMBER,
}
_POLICY_MAP: Final[dict[str, object]] = {
    "type": "object",
    "minProperties": 2,
    "additionalProperties": _STRING_ARRAY,
}
_CONDITIONAL_MAP: Final[dict[str, object]] = {
    "type": "object",
    "minProperties": 2,
    "additionalProperties": _NUMBER_MAP,
}

EVENT_LOCALIZED_INFORMATION_RESULT_SCHEMA_V1: Final[dict[str, object]] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://voiage.dev/specs/frontier/event-localized-information/v1/result.schema.json",
    "title": "EventLocalizedInformationResultV1",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema_version",
        "analysis_id",
        "analysis_type",
        "method_maturity",
        "value_unit",
        "direction",
        "chronology",
        "baseline",
        "event",
        "density",
        "assurance",
        "provenance",
        "references",
        "language_dispositions",
    ],
    "properties": {
        "schema_version": {"const": "v1"},
        "analysis_id": _TEXT,
        "analysis_type": {"const": "event_localized_information_value_result"},
        "method_maturity": {"const": "experimental"},
        "value_unit": _TEXT,
        "direction": {"const": "maximize"},
        "chronology": {"type": "array", "minItems": 3, "items": _TEXT},
        "baseline": {"$ref": "#/$defs/baseline"},
        "event": {"$ref": "#/$defs/eventResult"},
        "density": {"$ref": "#/$defs/densityResult"},
        "assurance": {"$ref": "#/$defs/assurance"},
        "provenance": _PROVENANCE,
        "references": {
            "type": "array",
            "minItems": 2,
            "items": {"$ref": "#/$defs/reference"},
        },
        "language_dispositions": {"$ref": "#/$defs/languages"},
    },
    "$defs": {
        "baseline": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "action_expected_values",
                "optimal_actions",
                "reference_action",
                "reference_value",
            ],
            "properties": {
                "action_expected_values": _NUMBER_MAP,
                "optimal_actions": _STRING_ARRAY,
                "reference_action": _TEXT,
                "reference_value": _NUMBER,
            },
        },
        "channelRow": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "accuracy",
                "signal_probabilities",
                "conditional_action_values",
                "optimal_actions",
                "informed_value",
                "gross_voi",
                "net_voi",
            ],
            "properties": {
                "accuracy": {"type": "number", "minimum": 0, "maximum": 1},
                "signal_probabilities": _NUMBER_MAP,
                "conditional_action_values": _CONDITIONAL_MAP,
                "optimal_actions": _POLICY_MAP,
                "informed_value": _NUMBER,
                "gross_voi": {"type": "number", "minimum": 0},
                "net_voi": _NUMBER,
            },
        },
        "eventResult": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "event_id",
                "definition",
                "partition_evidence",
                "state_ids",
                "complement_state_ids",
                "probability",
                "complement_probability",
                "conditional_action_values",
                "optimal_actions",
                "perfect_informed_value",
                "perfect_gross_voi",
                "information_cost",
                "perfect_net_voi",
                "imperfect_binary_channel",
            ],
            "properties": {
                "event_id": _TEXT,
                "definition": {"oneOf": [_THRESHOLD, _STATE_SET]},
                "partition_evidence": {
                    "type": "array",
                    "minItems": 1,
                    "items": {"$ref": "#/$defs/partitionEvidence"},
                },
                "state_ids": _STRING_ARRAY,
                "complement_state_ids": _STRING_ARRAY,
                "probability": {
                    "type": "number",
                    "exclusiveMinimum": 0,
                    "exclusiveMaximum": 1,
                },
                "complement_probability": {
                    "type": "number",
                    "exclusiveMinimum": 0,
                    "exclusiveMaximum": 1,
                },
                "conditional_action_values": _CONDITIONAL_MAP,
                "optimal_actions": _POLICY_MAP,
                "perfect_informed_value": _NUMBER,
                "perfect_gross_voi": {"type": "number", "minimum": 0},
                "information_cost": {"type": "number", "minimum": 0},
                "perfect_net_voi": _NUMBER,
                "imperfect_binary_channel": {
                    "type": "array",
                    "minItems": 1,
                    "items": {"$ref": "#/$defs/channelRow"},
                },
            },
        },
        "partitionEvidence": {
            "type": "object",
            "additionalProperties": False,
            "required": ["state_id", "coordinate", "event_member"],
            "properties": {
                "state_id": _TEXT,
                "coordinate": _NUMBER_ARRAY,
                "event_member": {"type": "boolean"},
            },
        },
        "atom": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "coordinate",
                "probability_mass",
                "conditional_action_values",
                "optimal_actions",
                "reference_policy_value",
                "policy_relative_density",
                "centered_density",
            ],
            "properties": {
                "coordinate": _NUMBER_ARRAY,
                "probability_mass": {
                    "type": "number",
                    "exclusiveMinimum": 0,
                    "maximum": 1,
                },
                "conditional_action_values": _NUMBER_MAP,
                "optimal_actions": _STRING_ARRAY,
                "reference_policy_value": _NUMBER,
                "policy_relative_density": {"type": "number", "minimum": 0},
                "centered_density": _NUMBER,
            },
        },
        "densityResult": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "measure",
                "formula",
                "centered_diagnostic_formula",
                "coordinate_names",
                "coordinate_units",
                "base_coordinate",
                "reference_action",
                "atoms",
                "information_value",
                "policy_relative_integral",
                "centered_integral",
                "integral_errors",
                "modes",
                "directions_from_base",
            ],
            "properties": {
                "measure": {"const": "probability_mass"},
                "formula": {"const": "f(x) * (max_a g_a(x) - g_reference(x))"},
                "centered_diagnostic_formula": {"const": "f(x) * (max_a g_a(x) - V0)"},
                "coordinate_names": _STRING_ARRAY,
                "coordinate_units": {
                    "type": "array",
                    "minItems": 1,
                    "items": _TEXT,
                },
                "base_coordinate": _NUMBER_ARRAY,
                "reference_action": _TEXT,
                "atoms": {
                    "type": "array",
                    "minItems": 1,
                    "items": {"$ref": "#/$defs/atom"},
                },
                "information_value": {"type": "number", "minimum": 0},
                "policy_relative_integral": {"type": "number", "minimum": 0},
                "centered_integral": _NUMBER,
                "integral_errors": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["policy_relative", "centered"],
                    "properties": {
                        "policy_relative": _NUMBER,
                        "centered": _NUMBER,
                    },
                },
                "modes": {"type": "array", "items": _NUMBER_ARRAY},
                "directions_from_base": {
                    "type": "array",
                    "items": _NUMBER_ARRAY,
                },
            },
        },
        "assurance": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "estimator",
                "complete_ties",
                "event_complement_partition",
                "density_integral_within_tolerance",
                "maximum_binary_channel_symmetry_error",
                "accuracy_half_no_information_residual",
                "policy_relative_density_nonnegative",
                "centered_density_is_signed_diagnostic",
                "bpi_delegated_to_issue_595",
                "continuous_density_claimed",
            ],
            "properties": {
                "estimator": {"const": "exact_finite_enumeration"},
                "complete_ties": {"const": True},
                "event_complement_partition": {"const": True},
                "density_integral_within_tolerance": {"const": True},
                "maximum_binary_channel_symmetry_error": {
                    "type": ["number", "null"],
                    "minimum": 0,
                },
                "accuracy_half_no_information_residual": {"type": ["number", "null"]},
                "policy_relative_density_nonnegative": {"const": True},
                "centered_density_is_signed_diagnostic": {"const": True},
                "bpi_delegated_to_issue_595": {"const": True},
                "continuous_density_claimed": {"const": False},
            },
        },
        "reference": {
            "type": "object",
            "additionalProperties": False,
            "required": ["doi", "role"],
            "properties": {"doi": _TEXT, "role": _TEXT},
        },
        "languages": {
            "type": "object",
            "additionalProperties": False,
            "required": ["Python", "Rust", "R", "Julia", "Mojo"],
            "properties": {
                "Python": {"const": "experimental_runtime"},
                "Rust": {"const": "not_implemented"},
                "R": {"const": "not_implemented"},
                "Julia": {"const": "not_implemented"},
                "Mojo": {"const": "external_upstream_boundary"},
            },
        },
    },
}


def _validate(
    schema: Mapping[str, object], payload: Mapping[str, Any], label: str
) -> None:
    try:
        Draft202012Validator(schema).validate(payload)
    except ValidationError as error:
        location = ".".join(str(part) for part in error.absolute_path) or "root"
        raise ValueError(
            f"{label} schema violation at {location}: {error.message}"
        ) from error


def _finite_tree(value: object, label: str = "payload") -> None:
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)) and not math.isfinite(float(value)):
        raise ValueError(f"{label} numbers must be finite")
    if isinstance(value, Mapping):
        for key, nested in value.items():
            _finite_tree(nested, f"{label}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, nested in enumerate(value):
            _finite_tree(nested, f"{label}[{index}]")


def _close(actual: float, expected: float, tolerance: float, label: str) -> None:
    if not math.isclose(actual, expected, abs_tol=tolerance, rel_tol=0.0):
        raise ValueError(f"{label} does not reconcile")


def _require_action_keys(
    values: Mapping[str, Any], actions: set[str], label: str
) -> None:
    if set(values) != actions:
        raise ValueError(f"{label} action identifiers do not reconcile")


def _ties_are_maximal(
    ties: Sequence[str], values: Mapping[str, Any], tolerance: float, label: str
) -> None:
    maximum = max(float(value) for value in values.values())
    expected = {
        key
        for key, value in values.items()
        if math.isclose(float(value), maximum, abs_tol=tolerance, rel_tol=0.0)
    }
    if set(ties) != expected:
        raise ValueError(f"{label} ties do not match maximal actions")


def validate_event_localized_information_semantics(payload: Mapping[str, Any]) -> None:
    """Validate strict schema and cross-field semantics for a v1 request."""
    _validate(EVENT_LOCALIZED_INFORMATION_INPUT_SCHEMA_V1, payload, "event input")
    _finite_tree(payload)
    states = cast("Sequence[Mapping[str, Any]]", payload["states"])
    actions = cast("Sequence[str]", payload["actions"])
    probability = math.fsum(float(state["probability"]) for state in states)
    if not math.isclose(probability, 1.0, abs_tol=1e-12, rel_tol=0.0):
        raise ValueError("state probabilities must sum to one")
    state_ids = [cast("str", state["state_id"]) for state in states]
    if len(set(state_ids)) != len(state_ids):
        raise ValueError("state identifiers must be unique")
    density = cast("Mapping[str, Any]", payload["density"])
    dimension = len(cast("Sequence[object]", density["coordinate_names"]))
    if len(cast("Sequence[object]", density["coordinate_units"])) != dimension:
        raise ValueError("coordinate_units must match the coordinate dimension")
    if len(cast("Sequence[object]", density["base_coordinate"])) != dimension:
        raise ValueError("base_coordinate must match the coordinate dimension")
    action_set = set(actions)
    for state in states:
        if len(cast("Sequence[object]", state["coordinate"])) != dimension:
            raise ValueError(
                "each state coordinate must match the coordinate dimension"
            )
        if set(cast("Mapping[str, Any]", state["action_values"])) != action_set:
            raise ValueError(
                "each state action_values must contain exactly all actions"
            )
    if density["reference_action"] not in action_set:
        raise ValueError("density reference_action must identify a declared action")
    definition = cast(
        "Mapping[str, Any]", cast("Mapping[str, Any]", payload["event"])["definition"]
    )
    if definition["kind"] == "state_set":
        if not set(cast("Sequence[str]", definition["state_ids"])) <= set(state_ids):
            raise ValueError("state_set event references an unknown state")
    elif int(definition["coordinate_index"]) >= dimension:
        raise ValueError("event coordinate_index is outside the coordinate dimension")


def validate_event_localized_information_result_semantics(
    payload: Mapping[str, Any], *, tolerance: float = 1e-9
) -> None:
    """Validate schema and numerical identities for a v1 result envelope."""
    _validate(EVENT_LOCALIZED_INFORMATION_RESULT_SCHEMA_V1, payload, "event result")
    _finite_tree(payload)
    baseline = cast("Mapping[str, Any]", payload["baseline"])
    baseline_values = cast("Mapping[str, Any]", baseline["action_expected_values"])
    actions = set(baseline_values)
    baseline_maximum = max(float(value) for value in baseline_values.values())
    _close(float(baseline["reference_value"]), baseline_maximum, tolerance, "baseline")
    reference = cast("str", baseline["reference_action"])
    if reference not in baseline_values:
        raise ValueError("baseline reference action is undeclared")
    _close(
        float(baseline_values[reference]),
        baseline_maximum,
        tolerance,
        "reference action",
    )
    _ties_are_maximal(
        cast("Sequence[str]", baseline["optimal_actions"]),
        baseline_values,
        tolerance,
        "baseline",
    )

    event = cast("Mapping[str, Any]", payload["event"])
    probability = float(event["probability"])
    complement_probability = float(event["complement_probability"])
    _close(probability + complement_probability, 1.0, tolerance, "event partition")
    if set(cast("Sequence[str]", event["state_ids"])) & set(
        cast("Sequence[str]", event["complement_state_ids"])
    ):
        raise ValueError("event and complement state identifiers must be disjoint")
    state_ids = set(cast("Sequence[str]", event["state_ids"]))
    complement_ids = set(cast("Sequence[str]", event["complement_state_ids"]))
    partition_evidence = cast(
        "Sequence[Mapping[str, Any]]", event["partition_evidence"]
    )
    evidence_ids = [cast("str", row["state_id"]) for row in partition_evidence]
    if len(set(evidence_ids)) != len(evidence_ids):
        raise ValueError("partition evidence state identifiers must be unique")
    if set(evidence_ids) != state_ids | complement_ids:
        raise ValueError("partition evidence does not cover the event partition")
    evidenced_event_ids = {
        cast("str", row["state_id"])
        for row in partition_evidence
        if bool(row["event_member"])
    }
    if evidenced_event_ids != state_ids:
        raise ValueError("partition evidence membership does not reconcile")
    definition = cast("Mapping[str, Any]", event["definition"])
    if (
        definition["kind"] == "state_set"
        and set(cast("Sequence[str]", definition["state_ids"])) != state_ids
    ):
        raise ValueError("state-set definition does not reconcile with the partition")
    conditional = cast(
        "Mapping[str, Mapping[str, Any]]", event["conditional_action_values"]
    )
    policies = cast("Mapping[str, Sequence[str]]", event["optimal_actions"])
    perfect = 0.0
    for key, weight in (("event", probability), ("complement", complement_probability)):
        _require_action_keys(conditional[key], actions, key)
        _ties_are_maximal(policies[key], conditional[key], tolerance, key)
        perfect += weight * max(float(value) for value in conditional[key].values())
    for action in actions:
        marginal = probability * float(conditional["event"][action]) + (
            complement_probability * float(conditional["complement"][action])
        )
        _close(
            float(baseline_values[action]),
            marginal,
            tolerance,
            f"event marginal for {action}",
        )
    _close(
        float(event["perfect_informed_value"]),
        perfect,
        tolerance,
        "perfect informed value",
    )
    gross = perfect - baseline_maximum
    _close(float(event["perfect_gross_voi"]), gross, tolerance, "perfect gross VOI")
    cost = float(event["information_cost"])
    _close(float(event["perfect_net_voi"]), gross - cost, tolerance, "perfect net VOI")

    curve = cast("Sequence[Mapping[str, Any]]", event["imperfect_binary_channel"])
    if len({float(row["accuracy"]) for row in curve}) != len(curve):
        raise ValueError("result channel accuracies must be unique")
    for row in curve:
        accuracy = float(row["accuracy"])
        probabilities = cast("Mapping[str, Any]", row["signal_probabilities"])
        _require_action_keys(
            probabilities,
            {"event_reported", "complement_reported"},
            "signal probabilities",
        )
        expected_report_event = accuracy * probability + (
            (1.0 - accuracy) * complement_probability
        )
        expected_report_complement = (1.0 - accuracy) * probability + (
            accuracy * complement_probability
        )
        _close(
            float(probabilities["event_reported"]),
            expected_report_event,
            tolerance,
            "event-reported signal probability",
        )
        _close(
            float(probabilities["complement_reported"]),
            expected_report_complement,
            tolerance,
            "complement-reported signal probability",
        )
        _close(
            math.fsum(float(value) for value in probabilities.values()),
            1.0,
            tolerance,
            "signal probabilities",
        )
        row_values = cast(
            "Mapping[str, Mapping[str, Any]]", row["conditional_action_values"]
        )
        row_policies = cast("Mapping[str, Sequence[str]]", row["optimal_actions"])
        informed = 0.0
        for key in ("event_reported", "complement_reported"):
            _require_action_keys(row_values[key], actions, key)
            _ties_are_maximal(row_policies[key], row_values[key], tolerance, key)
            informed += float(probabilities[key]) * max(
                float(value) for value in row_values[key].values()
            )
        for action in actions:
            event_numerator = accuracy * probability * float(
                conditional["event"][action]
            ) + (1.0 - accuracy) * complement_probability * float(
                conditional["complement"][action]
            )
            complement_numerator = (1.0 - accuracy) * probability * float(
                conditional["event"][action]
            ) + accuracy * complement_probability * float(
                conditional["complement"][action]
            )
            _close(
                float(row_values["event_reported"][action]),
                event_numerator / expected_report_event,
                tolerance,
                f"event-reported conditional value for {action}",
            )
            _close(
                float(row_values["complement_reported"][action]),
                complement_numerator / expected_report_complement,
                tolerance,
                f"complement-reported conditional value for {action}",
            )
        _close(
            float(row["informed_value"]), informed, tolerance, "channel informed value"
        )
        row_gross = informed - baseline_maximum
        _close(float(row["gross_voi"]), row_gross, tolerance, "channel gross VOI")
        _close(float(row["net_voi"]), row_gross - cost, tolerance, "channel net VOI")

    density = cast("Mapping[str, Any]", payload["density"])
    if density["reference_action"] != reference:
        raise ValueError("density and baseline reference actions do not reconcile")
    atoms = cast("Sequence[Mapping[str, Any]]", density["atoms"])
    dimension = len(cast("Sequence[object]", density["coordinate_names"]))
    if len(cast("Sequence[object]", density["coordinate_units"])) != dimension:
        raise ValueError("result coordinate units do not match the dimension")
    if len(cast("Sequence[object]", density["base_coordinate"])) != dimension:
        raise ValueError("result base coordinate does not match the dimension")
    evidence_coordinates: dict[str, Sequence[float]] = {}
    for row in partition_evidence:
        coordinate = cast("Sequence[float]", row["coordinate"])
        if len(coordinate) != dimension:
            raise ValueError(
                "partition evidence coordinate does not match the dimension"
            )
        evidence_coordinates[cast("str", row["state_id"])] = coordinate
    if definition["kind"] == "threshold":
        index = int(definition["coordinate_index"])
        if index >= dimension:
            raise ValueError("result event coordinate_index is outside the dimension")
        threshold = float(definition["threshold"])
        operator = cast("str", definition["operator"])
        predicates = {
            "less_than": lambda value: value < threshold,
            "less_than_or_equal": lambda value: value <= threshold,
            "greater_than": lambda value: value > threshold,
            "greater_than_or_equal": lambda value: value >= threshold,
        }
        expected_event_ids = {
            state_id
            for state_id, coordinate in evidence_coordinates.items()
            if predicates[operator](float(coordinate[index]))
        }
        if expected_event_ids != state_ids:
            raise ValueError(
                "threshold definition does not reconcile with the partition"
            )
    coordinates = [
        tuple(float(value) for value in cast("Sequence[Any]", atom["coordinate"]))
        for atom in atoms
    ]
    if len(set(coordinates)) != len(coordinates):
        raise ValueError("density atoms must have unique grouped coordinates")
    if set(coordinates) != {
        tuple(float(value) for value in coordinate)
        for coordinate in evidence_coordinates.values()
    }:
        raise ValueError("density atoms do not cover partition-evidence coordinates")
    mass = math.fsum(float(atom["probability_mass"]) for atom in atoms)
    _close(mass, 1.0, tolerance, "density probability mass")
    resolved = 0.0
    policy_integral = 0.0
    centered_integral = 0.0
    for atom in atoms:
        if len(cast("Sequence[object]", atom["coordinate"])) != dimension:
            raise ValueError("density atom coordinate does not match the dimension")
        values = cast("Mapping[str, Any]", atom["conditional_action_values"])
        _require_action_keys(values, actions, "atom")
        _ties_are_maximal(
            cast("Sequence[str]", atom["optimal_actions"]), values, tolerance, "atom"
        )
        atom_mass = float(atom["probability_mass"])
        resolved += atom_mass * max(float(value) for value in values.values())
        reference_value = float(values[reference])
        _close(
            float(atom["reference_policy_value"]),
            reference_value,
            tolerance,
            "atom reference value",
        )
        expected_policy_density = atom_mass * (
            max(float(value) for value in values.values()) - reference_value
        )
        expected_centered_density = atom_mass * (
            max(float(value) for value in values.values()) - baseline_maximum
        )
        _close(
            float(atom["policy_relative_density"]),
            expected_policy_density,
            tolerance,
            "policy-relative density",
        )
        _close(
            float(atom["centered_density"]),
            expected_centered_density,
            tolerance,
            "centered density",
        )
        policy_integral += float(atom["policy_relative_density"])
        centered_integral += float(atom["centered_density"])
    for action in actions:
        marginal = math.fsum(
            float(atom["probability_mass"])
            * float(
                cast("Mapping[str, Any]", atom["conditional_action_values"])[action]
            )
            for atom in atoms
        )
        _close(
            float(baseline_values[action]),
            marginal,
            tolerance,
            f"density marginal for {action}",
        )
    information_value = resolved - baseline_maximum
    _close(
        float(density["information_value"]),
        information_value,
        tolerance,
        "information value",
    )
    _close(
        float(density["policy_relative_integral"]),
        policy_integral,
        tolerance,
        "policy integral",
    )
    _close(
        float(density["centered_integral"]),
        centered_integral,
        tolerance,
        "centered integral",
    )
    errors = cast("Mapping[str, Any]", density["integral_errors"])
    _close(
        float(errors["policy_relative"]),
        policy_integral - information_value,
        tolerance,
        "policy error",
    )
    _close(
        float(errors["centered"]),
        centered_integral - information_value,
        tolerance,
        "centered error",
    )
    maximum_density = max(float(atom["policy_relative_density"]) for atom in atoms)
    expected_modes = (
        []
        if maximum_density <= tolerance
        else [
            cast("Sequence[float]", atom["coordinate"])
            for atom in atoms
            if math.isclose(
                float(atom["policy_relative_density"]),
                maximum_density,
                abs_tol=tolerance,
                rel_tol=0.0,
            )
        ]
    )
    modes = cast("Sequence[Sequence[float]]", density["modes"])
    if list(modes) != expected_modes:
        raise ValueError("density modes do not reconcile")
    base = cast("Sequence[float]", density["base_coordinate"])
    expected_directions = [
        [coordinate[index] - base[index] for index in range(dimension)]
        for coordinate in expected_modes
    ]
    if (
        cast("Sequence[Sequence[float]]", density["directions_from_base"])
        != expected_directions
    ):
        raise ValueError("density directions do not reconcile")

    symmetry_errors: list[float] = []
    for index, row in enumerate(curve):
        target = 1.0 - float(row["accuracy"])
        counterpart = next(
            (
                candidate
                for candidate in curve[index + 1 :]
                if math.isclose(
                    float(candidate["accuracy"]),
                    target,
                    abs_tol=1e-12,
                    rel_tol=0.0,
                )
            ),
            None,
        )
        if counterpart is not None:
            symmetry_errors.append(
                abs(float(row["gross_voi"]) - float(counterpart["gross_voi"]))
            )
    expected_symmetry = max(symmetry_errors) if symmetry_errors else None
    assurance = cast("Mapping[str, Any]", payload["assurance"])
    actual_symmetry = assurance["maximum_binary_channel_symmetry_error"]
    if expected_symmetry is None:
        if actual_symmetry is not None:
            raise ValueError("binary-channel symmetry must be not evaluated")
    else:
        if actual_symmetry is None:
            raise ValueError("binary-channel symmetry result is missing")
        _close(
            float(actual_symmetry),
            expected_symmetry,
            tolerance,
            "binary-channel symmetry",
        )
        if expected_symmetry > tolerance:
            raise ValueError("binary-channel symmetry exceeds tolerance")
    half = next(
        (
            row
            for row in curve
            if math.isclose(float(row["accuracy"]), 0.5, abs_tol=1e-12, rel_tol=0.0)
        ),
        None,
    )
    expected_half = float(half["gross_voi"]) if half is not None else None
    actual_half = assurance["accuracy_half_no_information_residual"]
    if expected_half is None:
        if actual_half is not None:
            raise ValueError("accuracy-half residual must be not evaluated")
    else:
        if actual_half is None:
            raise ValueError("accuracy-half residual is missing")
        _close(float(actual_half), expected_half, tolerance, "accuracy-half residual")
        if abs(expected_half) > tolerance:
            raise ValueError("accuracy-half residual exceeds tolerance")


__all__ = [
    "EVENT_LOCALIZED_INFORMATION_INPUT_SCHEMA_V1",
    "EVENT_LOCALIZED_INFORMATION_RESULT_SCHEMA_V1",
    "validate_event_localized_information_result_semantics",
    "validate_event_localized_information_semantics",
]
