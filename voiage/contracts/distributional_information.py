"""Installed v1 contract for experimental distribution-family information value."""

from __future__ import annotations

from collections.abc import Mapping
import json
import math
from typing import Final, cast

VALUE_OF_DISTRIBUTIONAL_INFORMATION_INPUT_SCHEMA_V1: Final[dict[str, object]] = (
    json.loads(
        r"""
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://voiage.dev/schemas/frontier/value-of-distributional-information-input.v1.json",
  "title": "ValueOfDistributionalInformationInputV1Experimental",
  "type": "object",
  "required": ["schema_version", "analysis_id", "analysis_type", "method_maturity", "information_target", "conditioning_order", "direction", "value_unit", "model_ids", "model_labels", "model_probabilities", "alternative_names", "conditional_values", "information_cost", "tolerances", "comparability", "provenance"],
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
    "model_probabilities": {"type": "array", "minItems": 1, "items": {"type": "number", "minimum": 0, "maximum": 1}},
    "alternative_names": {"type": "array", "minItems": 1, "uniqueItems": true, "items": {"type": "string", "minLength": 1}},
    "conditional_values": {"type": "array", "minItems": 1, "items": {"type": "array", "minItems": 1, "items": {"type": "number"}}},
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
      "required": ["population", "horizon", "discounting", "value_semantics", "cost_location"],
      "properties": {
        "population": {"type": "string", "minLength": 1},
        "horizon": {"type": "string", "minLength": 1},
        "discounting": {"type": "string", "minLength": 1},
        "value_semantics": {"type": "string", "minLength": 1},
        "cost_location": {"type": "string", "minLength": 1}
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


def validate_distributional_information_semantics(
    payload: Mapping[str, object],
) -> None:
    """Validate cross-field and finite-number rules outside JSON Schema."""
    model_ids = payload.get("model_ids")
    alternatives = payload.get("alternative_names")
    probabilities = payload.get("model_probabilities")
    values = payload.get("conditional_values")
    labels = payload.get("model_labels")
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
    if not math.isfinite(probability_tolerance) or probability_tolerance <= 0:
        raise ValueError("tolerances.probability_sum must be finite and positive.")
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
