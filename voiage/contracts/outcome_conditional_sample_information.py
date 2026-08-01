"""Strict contracts for outcome-conditional sample-information value."""

# pyright: reportAny=false, reportExplicitAny=false, reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportImportCycles=false, reportMissingModuleSource=false
# pyright: reportPrivateUsage=false, reportUnusedCallResult=false

from __future__ import annotations

from copy import deepcopy
import json
from typing import Any, Final, cast

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

_TEXT: Final[dict[str, object]] = {"type": "string", "minLength": 1}
_NUMBER: Final[dict[str, object]] = {"type": "number"}
_PROBABILITY: Final[dict[str, object]] = {
    "type": "number",
    "minimum": 0,
    "maximum": 1,
}
_NUMBER_MAP: Final[dict[str, object]] = {
    "type": "object",
    "minProperties": 2,
    "additionalProperties": _NUMBER,
}
_PROBABILITY_MAP: Final[dict[str, object]] = {
    "type": "object",
    "minProperties": 1,
    "additionalProperties": _PROBABILITY,
}
_STRING_ARRAY: Final[dict[str, object]] = {
    "type": "array",
    "minItems": 1,
    "uniqueItems": True,
    "items": _TEXT,
}
_LANGUAGES: Final[dict[str, object]] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["python", "rust", "r", "julia", "mojo"],
    "properties": {
        "python": {"const": "experimental_executable"},
        "rust": {"const": "unsupported"},
        "r": {"const": "unsupported"},
        "julia": {"const": "unsupported"},
        "mojo": {"const": "external"},
    },
}
_PROVENANCE: Final[dict[str, object]] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["probability_source", "likelihood_source", "value_source"],
    "properties": {
        "probability_source": _TEXT,
        "likelihood_source": _TEXT,
        "value_source": _TEXT,
    },
}

OUTCOME_CONDITIONAL_SAMPLE_INFORMATION_INPUT_SCHEMA_V1: Final[dict[str, object]] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://voiage.dev/specs/frontier/outcome-conditional-sample-information/v1/input.schema.json",
    "title": "OutcomeConditionalSampleInformationInputV1",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema_version",
        "analysis_id",
        "analysis_type",
        "method_maturity",
        "value_unit",
        "population",
        "horizon",
        "discount_basis",
        "objective",
        "chronology",
        "scope",
        "actions",
        "states",
        "outcomes",
        "reference_action",
        "information_cost",
        "cost_placement",
        "low_value_thresholds",
        "quantile_levels",
        "tie_tolerance",
        "probability_tolerance",
        "provenance",
    ],
    "properties": {
        "schema_version": {"const": "v1"},
        "analysis_id": _TEXT,
        "analysis_type": {"const": "outcome_conditional_sample_information_value"},
        "method_maturity": {"const": "experimental"},
        "value_unit": _TEXT,
        "population": _TEXT,
        "horizon": _TEXT,
        "discount_basis": _TEXT,
        "objective": {
            "type": "object",
            "additionalProperties": False,
            "required": ["measure", "direction"],
            "properties": {
                "measure": {"enum": ["utility", "loss"]},
                "direction": {"enum": ["maximize", "minimize"]},
            },
        },
        "chronology": {"const": ["prior", "measure", "observe", "update", "act"]},
        "scope": {
            "type": "object",
            "additionalProperties": False,
            "required": ["mode", "observed_outcome_id"],
            "properties": {
                "mode": {"enum": ["prospective", "retrospective"]},
                "observed_outcome_id": {"type": ["string", "null"]},
            },
        },
        "actions": {
            "type": "array",
            "minItems": 2,
            "uniqueItems": True,
            "items": _TEXT,
        },
        "states": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["state_id", "probability", "action_values"],
                "properties": {
                    "state_id": _TEXT,
                    "probability": {
                        "type": "number",
                        "exclusiveMinimum": 0,
                        "maximum": 1,
                    },
                    "action_values": _NUMBER_MAP,
                },
            },
        },
        "outcomes": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["outcome_id", "likelihood_by_state"],
                "properties": {
                    "outcome_id": _TEXT,
                    "likelihood_by_state": _PROBABILITY_MAP,
                },
            },
        },
        "reference_action": _TEXT,
        "information_cost": {"type": "number", "minimum": 0},
        "cost_placement": {"const": "subtract_after_gross_vsi"},
        "low_value_thresholds": {
            "type": "array",
            "minItems": 1,
            "uniqueItems": True,
            "items": {"type": "number", "minimum": 0},
        },
        "quantile_levels": {
            "type": "array",
            "minItems": 1,
            "uniqueItems": True,
            "items": _PROBABILITY,
        },
        "tie_tolerance": {"type": "number", "minimum": 0},
        "probability_tolerance": {
            "type": "number",
            "exclusiveMinimum": 0,
            "maximum": 1e-6,
        },
        "provenance": _PROVENANCE,
    },
}

_INPUT_CONTRACT = deepcopy(OUTCOME_CONDITIONAL_SAMPLE_INFORMATION_INPUT_SCHEMA_V1)
_ = _INPUT_CONTRACT.pop("$schema")
_ = _INPUT_CONTRACT.pop("$id")

_OUTCOME_RESULT: Final[dict[str, object]] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "outcome_id",
        "predictive_probability",
        "posterior_state_probabilities",
        "posterior_action_values",
        "optimal_actions",
        "posterior_optimal_value",
        "reference_action_value",
        "delta_ev",
        "vsi",
        "net_vsi",
        "reference_action_excluded",
        "mandatory_policy_switch",
        "complete_tie_set_changed",
    ],
    "properties": {
        "outcome_id": _TEXT,
        "predictive_probability": _PROBABILITY,
        "posterior_state_probabilities": _PROBABILITY_MAP,
        "posterior_action_values": _NUMBER_MAP,
        "optimal_actions": _STRING_ARRAY,
        "posterior_optimal_value": _NUMBER,
        "reference_action_value": _NUMBER,
        "delta_ev": _NUMBER,
        "vsi": {"type": "number", "minimum": 0},
        "net_vsi": _NUMBER,
        "reference_action_excluded": {"type": "boolean"},
        "mandatory_policy_switch": {"type": "boolean"},
        "complete_tie_set_changed": {"type": "boolean"},
    },
}

OUTCOME_CONDITIONAL_SAMPLE_INFORMATION_RESULT_SCHEMA_V1: Final[dict[str, object]] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://voiage.dev/specs/frontier/outcome-conditional-sample-information/v1/result.schema.json",
    "title": "OutcomeConditionalSampleInformationResultV1",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema_version",
        "analysis_id",
        "analysis_type",
        "method_maturity",
        "value_unit",
        "population",
        "horizon",
        "discount_basis",
        "objective",
        "chronology",
        "scope",
        "baseline",
        "outcomes",
        "aggregate",
        "retrospective_outcome",
        "assurance",
        "input_assurance",
        "provenance",
        "references",
        "language_dispositions",
    ],
    "properties": {
        "schema_version": {"const": "v1"},
        "analysis_id": _TEXT,
        "analysis_type": {
            "const": "outcome_conditional_sample_information_value_result"
        },
        "method_maturity": {"const": "experimental"},
        "value_unit": _TEXT,
        "population": _TEXT,
        "horizon": _TEXT,
        "discount_basis": _TEXT,
        "objective": {
            "type": "object",
            "additionalProperties": False,
            "required": ["measure", "direction"],
            "properties": {
                "measure": {"enum": ["utility", "loss"]},
                "direction": {"enum": ["maximize", "minimize"]},
            },
        },
        "chronology": {"const": ["prior", "measure", "observe", "update", "act"]},
        "scope": {
            "type": "object",
            "additionalProperties": False,
            "required": ["mode", "observed_outcome_id"],
            "properties": {
                "mode": {"enum": ["prospective", "retrospective"]},
                "observed_outcome_id": {"type": ["string", "null"]},
            },
        },
        "baseline": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "action_expected_values",
                "optimal_actions",
                "reference_action",
                "reference_expected_value",
            ],
            "properties": {
                "action_expected_values": _NUMBER_MAP,
                "optimal_actions": _STRING_ARRAY,
                "reference_action": _TEXT,
                "reference_expected_value": _NUMBER,
            },
        },
        "outcomes": {"type": "array", "minItems": 1, "items": _OUTCOME_RESULT},
        "aggregate": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "evsi",
                "expected_delta_ev",
                "information_cost",
                "net_evsi",
                "variance_vsi",
                "sigma_vsi",
                "minimum_vsi",
                "maximum_vsi",
                "low_value_risks",
                "weighted_quantiles",
                "lower_tail_means",
                "reference_action_excluded_probability",
                "mandatory_policy_switch_probability",
                "complete_tie_set_changed_probability",
            ],
            "properties": {
                "evsi": {"type": "number", "minimum": 0},
                "expected_delta_ev": _NUMBER,
                "information_cost": {"type": "number", "minimum": 0},
                "net_evsi": _NUMBER,
                "variance_vsi": {"type": "number", "minimum": 0},
                "sigma_vsi": {"type": "number", "minimum": 0},
                "minimum_vsi": {"type": "number", "minimum": 0},
                "maximum_vsi": {"type": "number", "minimum": 0},
                "low_value_risks": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["delta", "probability"],
                        "properties": {
                            "delta": {"type": "number", "minimum": 0},
                            "probability": _PROBABILITY,
                        },
                    },
                },
                "weighted_quantiles": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["level", "vsi"],
                        "properties": {"level": _PROBABILITY, "vsi": _NUMBER},
                    },
                },
                "lower_tail_means": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["mass", "mean_vsi"],
                        "properties": {"mass": _PROBABILITY, "mean_vsi": _NUMBER},
                    },
                },
                "reference_action_excluded_probability": _PROBABILITY,
                "mandatory_policy_switch_probability": _PROBABILITY,
                "complete_tie_set_changed_probability": _PROBABILITY,
            },
        },
        "retrospective_outcome": {"oneOf": [{"type": "null"}, _OUTCOME_RESULT]},
        "assurance": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "estimator",
                "standard_deviation_functional",
                "ddof",
                "tower_identity_scope",
                "evsi_vsi_residual",
                "evsi_delta_ev_residual",
                "predictive_probability_residual",
                "maximum_likelihood_row_residual",
                "maximum_bayes_reconstruction_error",
                "minimum_vsi",
                "threshold_monotonic",
                "r_vsi_zero_distinct_from_switch_metrics",
                "continuous_outcomes_supported",
                "result_reconstructed_from_input",
            ],
            "properties": {
                "estimator": {"const": "exact_finite_enumeration"},
                "standard_deviation_functional": {
                    "const": "predictive_probability_weighted_population"
                },
                "ddof": {"const": 0},
                "tower_identity_scope": {"const": "expectations_only"},
                "evsi_vsi_residual": _NUMBER,
                "evsi_delta_ev_residual": _NUMBER,
                "predictive_probability_residual": _NUMBER,
                "maximum_likelihood_row_residual": _NUMBER,
                "maximum_bayes_reconstruction_error": _NUMBER,
                "minimum_vsi": _NUMBER,
                "threshold_monotonic": {"type": "boolean"},
                "r_vsi_zero_distinct_from_switch_metrics": {"const": True},
                "continuous_outcomes_supported": {"const": False},
                "result_reconstructed_from_input": {"const": True},
            },
        },
        "input_assurance": {
            "type": "object",
            "additionalProperties": False,
            "required": ["input_sha256", "input_contract"],
            "properties": {
                "input_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                "input_contract": _INPUT_CONTRACT,
            },
        },
        "provenance": _PROVENANCE,
        "references": {
            "type": "array",
            "minItems": 2,
            "uniqueItems": True,
            "items": _TEXT,
        },
        "language_dispositions": _LANGUAGES,
    },
}


def _validate(schema: dict[str, object], payload: object, label: str) -> None:
    try:
        Draft202012Validator(schema).validate(payload)
    except ValidationError as error:
        location = ".".join(str(part) for part in error.absolute_path)
        prefix = f"{label}.{location}" if location else label
        raise ValueError(f"{prefix}: {error.message}") from error


def validate_outcome_conditional_sample_information_semantics(
    payload: object,
) -> None:
    """Validate the strict input schema and cross-field semantics."""
    _validate(
        OUTCOME_CONDITIONAL_SAMPLE_INFORMATION_INPUT_SCHEMA_V1,
        payload,
        "input",
    )
    from voiage.methods.outcome_conditional_sample_information import (
        _validate_and_build,
    )

    _validate_and_build(cast("dict[str, Any]", deepcopy(payload)))


def validate_outcome_conditional_sample_information_result_semantics(
    payload: object,
) -> None:
    """Validate and independently reconstruct a portable result."""
    _validate(
        OUTCOME_CONDITIONAL_SAMPLE_INFORMATION_RESULT_SCHEMA_V1,
        payload,
        "result",
    )
    result = cast("dict[str, Any]", deepcopy(payload))
    assurance = cast("dict[str, Any]", result["input_assurance"])
    input_contract = cast("dict[str, Any]", assurance["input_contract"])
    from voiage.methods.outcome_conditional_sample_information import (
        _contract_sha256,
        _evaluate_contract,
        _validate_and_build,
    )

    if _contract_sha256(input_contract) != assurance["input_sha256"]:
        raise ValueError("result input commitment digest is inconsistent")
    expected = _evaluate_contract(_validate_and_build(input_contract))
    if json.dumps(result, sort_keys=True, allow_nan=False) != json.dumps(
        expected, sort_keys=True, allow_nan=False
    ):
        raise ValueError("result is inconsistent with its committed input contract")
