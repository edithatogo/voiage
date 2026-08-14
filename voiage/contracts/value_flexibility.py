"""Versioned runtime contracts for experimental Value of Flexibility."""

from typing import Final

VALUE_OF_FLEXIBILITY_INPUT_SCHEMA_V1: Final[dict[str, object]] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://voiage.dev/schemas/frontier/value-of-flexibility-input.v1.json",
    "title": "ValueOfFlexibilityInputV1Experimental",
    "type": "object",
    "required": [
        "decision_stage_names",
        "strategy_names",
        "net_benefit",
        "stage_weights",
        "value_unit",
        "stage_semantics",
        "information_value_included",
        "provenance",
    ],
    "properties": {
        "decision_stage_names": {
            "type": "array",
            "minItems": 1,
            "uniqueItems": True,
            "items": {"type": "string", "minLength": 1},
        },
        "strategy_names": {
            "type": "array",
            "minItems": 1,
            "uniqueItems": True,
            "items": {"type": "string", "minLength": 1},
        },
        "net_benefit": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "array",
                    "minItems": 1,
                    "items": {"type": "number"},
                },
            },
        },
        "stage_weights": {
            "type": "object",
            "minProperties": 1,
            "additionalProperties": {"type": "number", "minimum": 0},
        },
        "evidence_arrival_times": {
            "type": "object",
            "minProperties": 1,
            "additionalProperties": {"type": "number", "minimum": 0},
        },
        "flexible_policy_sets": {
            "type": "object",
            "minProperties": 1,
            "additionalProperties": {
                "type": "array",
                "minItems": 1,
                "uniqueItems": True,
                "items": {"type": "string", "minLength": 1},
            },
        },
        "constrained_strategy_names": {
            "type": "array",
            "minItems": 1,
            "uniqueItems": True,
            "items": {"type": "string", "minLength": 1},
        },
        "discount_rate": {"const": 0},
        "irreversibility_penalty": {"const": 0},
        "lock_in_penalty": {"const": 0},
        "value_unit": {"type": "string", "minLength": 1},
        "stage_semantics": {"const": "timing_scenarios"},
        "information_value_included": {"const": False},
        "provenance": {
            "type": "object",
            "required": ["fixture_id", "execution_mode"],
            "properties": {
                "fixture_id": {"type": "string", "minLength": 1},
                "execution_mode": {"const": "deterministic"},
            },
            "additionalProperties": False,
        },
    },
    "additionalProperties": False,
}
