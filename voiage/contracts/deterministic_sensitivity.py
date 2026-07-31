"""Installed-wheel contract for experimental deterministic sensitivity analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, cast

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from voiage.exceptions import raise_input_error

if TYPE_CHECKING:
    from collections.abc import Mapping

_NON_EMPTY_STRING: Final[dict[str, object]] = {"type": "string", "minLength": 1}
_COORDINATE_VALUE: Final[dict[str, object]] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["parameter_name", "value"],
    "properties": {
        "parameter_name": _NON_EMPTY_STRING,
        "value": {"type": "number"},
    },
}

DETERMINISTIC_SENSITIVITY_INPUT_SCHEMA_V1: Final[dict[str, object]] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://voiage.dev/schemas/frontier/deterministic-sensitivity-input.v1.json",
    "title": "DeterministicSensitivityInputV1Experimental",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema_version",
        "analysis_id",
        "baseline",
        "parameter_grids",
        "alternative_names",
        "output_unit",
        "direction",
        "tie_tolerance",
        "two_way_designs",
        "scenarios",
        "model_evaluation_records",
        "provenance",
    ],
    "properties": {
        "schema_version": {"const": "deterministic-sensitivity-input-v1"},
        "analysis_id": _NON_EMPTY_STRING,
        "baseline": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["parameter_name", "value", "unit"],
                "properties": {
                    "parameter_name": _NON_EMPTY_STRING,
                    "value": {"type": "number"},
                    "unit": _NON_EMPTY_STRING,
                },
            },
        },
        "parameter_grids": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "parameter_name",
                    "unit",
                    "values",
                    "range_provenance",
                ],
                "properties": {
                    "parameter_name": _NON_EMPTY_STRING,
                    "unit": _NON_EMPTY_STRING,
                    "values": {
                        "type": "array",
                        "minItems": 1,
                        "uniqueItems": True,
                        "items": {"type": "number"},
                    },
                    "range_provenance": _NON_EMPTY_STRING,
                },
            },
        },
        "alternative_names": {
            "type": "array",
            "minItems": 1,
            "uniqueItems": True,
            "items": _NON_EMPTY_STRING,
        },
        "output_unit": _NON_EMPTY_STRING,
        "direction": {"enum": ["maximize", "minimize"]},
        "tie_tolerance": {
            "type": "object",
            "additionalProperties": False,
            "required": ["absolute", "relative", "representative_policy"],
            "properties": {
                "absolute": {"type": "number", "minimum": 0},
                "relative": {"type": "number", "minimum": 0},
                "representative_policy": {"const": "canonical-lexicographic"},
            },
        },
        "two_way_designs": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "surface_id",
                    "first_parameter",
                    "second_parameter",
                    "feasibility_semantics",
                    "feasible_points",
                ],
                "properties": {
                    "surface_id": _NON_EMPTY_STRING,
                    "first_parameter": _NON_EMPTY_STRING,
                    "second_parameter": _NON_EMPTY_STRING,
                    "feasibility_semantics": {
                        "enum": [
                            "explicit-mask",
                            "explicit-path",
                            "full-cartesian-independent",
                        ]
                    },
                    "feasible_points": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["first", "second"],
                            "properties": {
                                "first": {"type": "number"},
                                "second": {"type": "number"},
                            },
                        },
                    },
                },
            },
        },
        "scenarios": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "scenario_id",
                    "coordinates",
                    "structural_assumption",
                ],
                "properties": {
                    "scenario_id": _NON_EMPTY_STRING,
                    "coordinates": {
                        "type": "array",
                        "minItems": 1,
                        "items": _COORDINATE_VALUE,
                    },
                    "structural_assumption": _NON_EMPTY_STRING,
                },
            },
        },
        "model_evaluation_records": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "record_id",
                    "analysis_kind",
                    "analysis_ref",
                    "coordinates",
                    "alternative_outputs",
                ],
                "properties": {
                    "record_id": _NON_EMPTY_STRING,
                    "analysis_kind": {
                        "enum": ["baseline", "one-way", "two-way", "scenario"]
                    },
                    "analysis_ref": _NON_EMPTY_STRING,
                    "coordinates": {
                        "type": "array",
                        "minItems": 1,
                        "items": _COORDINATE_VALUE,
                    },
                    "alternative_outputs": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["alternative_name", "value", "unit"],
                            "properties": {
                                "alternative_name": _NON_EMPTY_STRING,
                                "value": {"type": "number"},
                                "unit": _NON_EMPTY_STRING,
                            },
                        },
                    },
                },
            },
        },
        "provenance": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "fixture_id",
                "baseline_source",
                "execution_mode",
                "reference",
            ],
            "properties": {
                "fixture_id": _NON_EMPTY_STRING,
                "baseline_source": _NON_EMPTY_STRING,
                "execution_mode": {"const": "deterministic"},
                "reference": _NON_EMPTY_STRING,
            },
        },
    },
}


def _unique_names(entries: list[object], key: str, label: str) -> list[str]:
    names = [str(cast("Mapping[str, object]", item)[key]) for item in entries]
    if len(names) != len(set(names)):
        raise_input_error(f"{label} must use unique {key} values.")
    return names


def validate_deterministic_sensitivity_specification(
    specification: Mapping[str, object],
) -> None:
    """Validate the exact v1 schema and cross-field unit/name invariants."""
    try:
        Draft202012Validator(DETERMINISTIC_SENSITIVITY_INPUT_SCHEMA_V1).validate(
            specification
        )
    except ValidationError as error:
        location = ".".join(str(part) for part in error.absolute_path) or "root"
        raise_input_error(f"Invalid DSA specification at {location}: {error.message}")

    baseline = cast("list[object]", specification["baseline"])
    grids = cast("list[object]", specification["parameter_grids"])
    alternatives = cast("list[str]", specification["alternative_names"])
    records = cast("list[object]", specification["model_evaluation_records"])
    baseline_names = _unique_names(baseline, "parameter_name", "baseline")
    grid_names = _unique_names(grids, "parameter_name", "parameter_grids")
    if set(baseline_names) != set(grid_names):
        raise_input_error("baseline and parameter_grids must name the same parameters.")
    baseline_units = {
        str(cast("Mapping[str, object]", item)["parameter_name"]): str(
            cast("Mapping[str, object]", item)["unit"]
        )
        for item in baseline
    }
    grid_units = {
        str(cast("Mapping[str, object]", item)["parameter_name"]): str(
            cast("Mapping[str, object]", item)["unit"]
        )
        for item in grids
    }
    if baseline_units != grid_units:
        raise_input_error("baseline and parameter grid units must match exactly.")
    output_unit = str(specification["output_unit"])
    for record in records:
        record_map = cast("Mapping[str, object]", record)
        coordinates = cast("list[object]", record_map["coordinates"])
        outputs = cast("list[object]", record_map["alternative_outputs"])
        if set(
            _unique_names(coordinates, "parameter_name", "record coordinates")
        ) != set(baseline_names):
            raise_input_error(
                "Every evaluation record must contain the complete baseline coordinate set."
            )
        output_names = _unique_names(outputs, "alternative_name", "record outputs")
        if set(output_names) != set(alternatives):
            raise_input_error(
                "Every evaluation record must contain exactly alternative_names."
            )
        if any(
            str(cast("Mapping[str, object]", item)["unit"]) != output_unit
            for item in outputs
        ):
            raise_input_error("Every evaluation output unit must match output_unit.")
