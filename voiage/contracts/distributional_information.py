"""Installed v1 contract for experimental distribution-family information value."""

from __future__ import annotations

from collections.abc import Mapping
import json
import math
from typing import Final, cast

MAX_PROBABILITY_SUM_TOLERANCE: Final = 1e-6

VALUE_OF_DISTRIBUTIONAL_INFORMATION_INPUT_SCHEMA_V1: Final[dict[str, object]] = (
    json.loads(
        r"""
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://voiage.dev/schemas/frontier/value-of-distributional-information-input.v1.json",
  "title": "ValueOfDistributionalInformationInputV1Experimental",
  "type": "object",
  "required": ["schema_version", "analysis_id", "analysis_type", "method_maturity", "information_target", "conditioning_order", "direction", "value_unit", "model_ids", "model_labels", "model_definitions", "model_probabilities", "alternative_names", "conditional_values", "conditional_value_assurance", "information_cost", "tolerances", "comparability", "provenance"],
  "properties": {
    "schema_version": {"const": "1.0.0"},
    "analysis_id": {"type": "string", "minLength": 1},
    "analysis_type": {"const": "distribution_family_information_value"},
    "method_maturity": {"const": "experimental"},
    "information_target": {"const": "model_family_index"},
    "conditioning_order": {"const": "integrate_within_family_then_resolve_family_index"},
    "direction": {"enum": ["maximize", "minimize"]},
    "value_unit": {"type": "string", "minLength": 1},
    "model_ids": {"type": "array", "minItems": 1, "uniqueItems": true, "items": {"type": "string", "minLength": 1}},
    "model_labels": {"type": "object", "minProperties": 1, "additionalProperties": {"type": "string", "minLength": 1}},
    "model_definitions": {
      "type": "array",
      "minItems": 1,
      "items": {
        "type": "object",
        "required": ["model_id", "family_or_assumption", "parameterization", "within_family_integration", "definition_source", "parameter_source", "data_reference", "value_transformation"],
        "properties": {
          "model_id": {"type": "string", "minLength": 1},
          "family_or_assumption": {"type": "string", "minLength": 1},
          "parameterization": {"type": "string", "minLength": 1},
          "within_family_integration": {"type": "string", "minLength": 1},
          "definition_source": {"type": "string", "minLength": 1},
          "parameter_source": {"type": "string", "minLength": 1},
          "data_reference": {"type": "string", "minLength": 1},
          "value_transformation": {"type": "string", "minLength": 1}
        },
        "additionalProperties": false
      }
    },
    "model_probabilities": {"type": "array", "minItems": 1, "items": {"type": "number", "minimum": 0, "maximum": 1}},
    "alternative_names": {"type": "array", "minItems": 1, "uniqueItems": true, "items": {"type": "string", "minLength": 1}},
    "conditional_values": {"type": "array", "minItems": 1, "items": {"type": "array", "minItems": 1, "items": {"type": "number"}}},
    "conditional_value_assurance": {
      "type": "object",
      "required": ["input_status", "source_values_exact", "source_uncertainty", "enumeration_method", "evidence_reference"],
      "properties": {
        "input_status": {"const": "exact_enumerated_conditional_expectations"},
        "source_values_exact": {"const": true},
        "source_uncertainty": {"const": "none_by_construction"},
        "enumeration_method": {"type": "string", "minLength": 1},
        "evidence_reference": {"type": "string", "minLength": 1}
      },
      "additionalProperties": false
    },
    "information_cost": {"type": "number", "minimum": 0},
    "tolerances": {
      "type": "object",
      "required": ["absolute", "relative", "probability_sum"],
      "properties": {
        "absolute": {"type": "number", "minimum": 0},
        "relative": {"type": "number", "minimum": 0},
        "probability_sum": {"type": "number", "exclusiveMinimum": 0, "maximum": 0.000001}
      },
      "additionalProperties": false
    },
    "comparability": {
      "type": "object",
      "required": ["population_id", "horizon_id", "discounting_id", "value_semantics_id", "cost_location_id", "verified", "verification_reference"],
      "properties": {
        "population_id": {"type": "string", "minLength": 1},
        "horizon_id": {"type": "string", "minLength": 1},
        "discounting_id": {"type": "string", "minLength": 1},
        "value_semantics_id": {"type": "string", "minLength": 1},
        "cost_location_id": {"type": "string", "minLength": 1},
        "verified": {"const": true},
        "verification_reference": {"type": "string", "minLength": 1}
      },
      "additionalProperties": false
    },
    "provenance": {
      "type": "object",
      "required": ["fixture_id", "probability_source", "value_source", "family_definition_source"],
      "properties": {
        "fixture_id": {"type": "string", "minLength": 1},
        "probability_source": {"type": "string", "minLength": 1},
        "value_source": {"type": "string", "minLength": 1},
        "family_definition_source": {"type": "string", "minLength": 1}
      },
      "additionalProperties": false
    }
  },
  "additionalProperties": false
}
"""
    )
)


def _nonempty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_distributional_information_semantics(
    payload: Mapping[str, object],
) -> None:
    """Validate cross-field and finite-number rules outside JSON Schema."""
    model_ids = payload.get("model_ids")
    alternatives = payload.get("alternative_names")
    probabilities = payload.get("model_probabilities")
    values = payload.get("conditional_values")
    labels = payload.get("model_labels")
    definitions = payload.get("model_definitions")
    assurance = payload.get("conditional_value_assurance")
    comparability = payload.get("comparability")
    tolerances = payload.get("tolerances")

    if not isinstance(model_ids, list) or not model_ids:
        raise ValueError("model_ids must be a non-empty list.")
    if not isinstance(alternatives, list) or not alternatives:
        raise ValueError("alternative_names must be a non-empty list.")
    if len(set(model_ids)) != len(model_ids):
        raise ValueError("model_ids must be unique.")
    if len(set(alternatives)) != len(alternatives):
        raise ValueError("alternative_names must be unique.")
    if not isinstance(labels, Mapping) or set(labels) != set(model_ids):
        raise ValueError("model_labels keys must exactly match model_ids.")
    definition_fields = {
        "model_id",
        "family_or_assumption",
        "parameterization",
        "within_family_integration",
        "definition_source",
        "parameter_source",
        "data_reference",
        "value_transformation",
    }
    if not isinstance(definitions, list) or len(definitions) != len(model_ids):
        raise ValueError("model_definitions must align with model_ids.")
    if not all(isinstance(item, Mapping) for item in definitions):
        raise TypeError("model_definitions entries must be objects.")
    definition_records = cast("list[Mapping[str, object]]", definitions)
    if any(
        set(item) != definition_fields
        or any(not _nonempty_string(item[field]) for field in definition_fields)
        for item in definition_records
    ):
        raise ValueError("model_definitions must contain complete non-empty records.")
    if [item["model_id"] for item in definition_records] != model_ids:
        raise ValueError("model_definitions must use model_ids order exactly.")
    assurance_fields = {
        "input_status",
        "source_values_exact",
        "source_uncertainty",
        "enumeration_method",
        "evidence_reference",
    }
    if not isinstance(assurance, Mapping) or set(assurance) != assurance_fields:
        raise ValueError("conditional_value_assurance is incomplete.")
    assurance_record = cast("Mapping[str, object]", assurance)
    if (
        assurance_record["input_status"] != "exact_enumerated_conditional_expectations"
        or assurance_record["source_values_exact"] is not True
        or assurance_record["source_uncertainty"] != "none_by_construction"
        or not _nonempty_string(assurance_record["enumeration_method"])
        or not _nonempty_string(assurance_record["evidence_reference"])
    ):
        raise ValueError(
            "conditional_value_assurance must prove exact enumerated input values."
        )
    comparability_fields = {
        "population_id",
        "horizon_id",
        "discounting_id",
        "value_semantics_id",
        "cost_location_id",
        "verified",
        "verification_reference",
    }
    if (
        not isinstance(comparability, Mapping)
        or set(comparability) != comparability_fields
    ):
        raise ValueError("comparability must contain the complete verified contract.")
    comparability_record = cast("Mapping[str, object]", comparability)
    if comparability_record["verified"] is not True or any(
        not _nonempty_string(comparability_record[field])
        for field in comparability_fields - {"verified"}
    ):
        raise ValueError("comparability must be explicitly verified with common IDs.")
    if not isinstance(probabilities, list) or len(probabilities) != len(model_ids):
        raise ValueError("model_probabilities must align with model_ids.")
    if not all(
        isinstance(item, (int, float)) and math.isfinite(float(item))
        for item in probabilities
    ):
        raise ValueError("model_probabilities must contain only finite numbers.")
    numeric_probabilities = [float(cast("int | float", item)) for item in probabilities]
    if any(item < 0 for item in numeric_probabilities):
        raise ValueError("model_probabilities must be non-negative.")
    if not isinstance(tolerances, Mapping):
        raise TypeError("tolerances must be an object.")
    probability_tolerance_value = tolerances.get("probability_sum", math.nan)
    if not isinstance(probability_tolerance_value, (int, float)):
        raise TypeError("tolerances.probability_sum must be numeric.")
    probability_tolerance = float(probability_tolerance_value)
    if (
        not math.isfinite(probability_tolerance)
        or probability_tolerance <= 0
        or probability_tolerance > MAX_PROBABILITY_SUM_TOLERANCE
    ):
        raise ValueError(
            "tolerances.probability_sum must be finite, positive and at most 1e-6."
        )
    probability_sum = math.fsum(numeric_probabilities)
    if not math.isclose(
        probability_sum, 1.0, rel_tol=0.0, abs_tol=probability_tolerance
    ):
        raise ValueError("model_probabilities must sum to 1 without renormalization.")
    if not isinstance(values, list) or len(values) != len(model_ids):
        raise ValueError("conditional_values rows must align with model_ids.")
    for row in values:
        if not isinstance(row, list) or len(row) != len(alternatives):
            raise ValueError(
                "Every conditional_values row must align with alternative_names."
            )
        if not all(
            isinstance(item, (int, float)) and math.isfinite(float(item))
            for item in row
        ):
            raise ValueError("conditional_values must contain only finite numbers.")
    information_cost = payload.get("information_cost")
    if not isinstance(information_cost, (int, float)) or not math.isfinite(
        float(information_cost)
    ):
        raise ValueError("information_cost must be finite.")
