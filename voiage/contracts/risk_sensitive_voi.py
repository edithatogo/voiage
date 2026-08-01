"""Portable v1 contract for exact risk-sensitive constrained VOI."""

# pyright: reportAny=false, reportExplicitAny=false, reportUnknownMemberType=false
# ruff: noqa: TRY301

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any, Final, cast

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from voiage.exceptions import InputError, raise_input_error

_ID: Final[dict[str, object]] = {
    "type": "string",
    "minLength": 1,
    "pattern": r"^[A-Za-z][A-Za-z0-9._-]*$",
}
_TEXT: Final[dict[str, object]] = {"type": "string", "minLength": 1}
_NUMBER: Final[dict[str, object]] = {"type": "number"}

RISK_SENSITIVE_VOI_INPUT_SCHEMA_V1: Final[dict[str, object]] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://voiage.dev/schemas/frontier/risk-sensitive-voi-input.v1.json",
    "title": "RiskSensitiveConstrainedVoiInputV1Experimental",
    "type": "object",
    "required": [
        "schema_version",
        "analysis_id",
        "analysis_type",
        "method_maturity",
        "planned_version",
        "objective",
        "states",
        "policies",
        "constraints",
        "information_action",
        "tolerances",
        "assurance",
        "provenance",
    ],
    "properties": {
        "schema_version": {"const": "1.0.0"},
        "analysis_id": _ID,
        "analysis_type": {"const": "risk_sensitive_constrained_perfect_information"},
        "method_maturity": {"const": "experimental"},
        "planned_version": {"const": "v1.3.0"},
        "objective": {
            "type": "object",
            "required": ["kind", "direction", "unit", "operational_definition"],
            "properties": {
                "kind": {
                    "enum": [
                        "expected_value",
                        "expected_utility",
                        "lower_tail_cvar",
                        "minimax_regret",
                    ]
                },
                "direction": {"const": "maximize"},
                "unit": _TEXT,
                "operational_definition": _TEXT,
                "confidence_level": {
                    "type": "number",
                    "exclusiveMinimum": 0,
                    "exclusiveMaximum": 1,
                },
                "regret_reference_by_state": {
                    "type": "object",
                    "minProperties": 1,
                    "additionalProperties": _NUMBER,
                },
            },
            "additionalProperties": False,
        },
        "states": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["state_id", "probability"],
                "properties": {
                    "state_id": _ID,
                    "probability": {
                        "type": "number",
                        "exclusiveMinimum": 0,
                        "maximum": 1,
                    },
                },
                "additionalProperties": False,
            },
        },
        "policies": {
            "type": "array",
            "minItems": 2,
            "items": {
                "type": "object",
                "required": [
                    "policy_id",
                    "label",
                    "objective_by_state",
                    "constraint_usage",
                    "source_reference",
                ],
                "properties": {
                    "policy_id": _ID,
                    "label": _TEXT,
                    "source_reference": _TEXT,
                    "objective_by_state": {
                        "type": "object",
                        "minProperties": 1,
                        "additionalProperties": _NUMBER,
                    },
                    "constraint_usage": {
                        "type": "object",
                        "additionalProperties": {
                            "type": "object",
                            "minProperties": 1,
                            "additionalProperties": _NUMBER,
                        },
                    },
                },
                "additionalProperties": False,
            },
        },
        "constraints": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": [
                    "constraint_id",
                    "kind",
                    "label",
                    "unit",
                    "sense",
                    "limit",
                    "enforcement",
                    "source_reference",
                ],
                "properties": {
                    "constraint_id": _ID,
                    "kind": {
                        "enum": [
                            "budget",
                            "capacity",
                            "eligibility",
                            "fairness",
                            "regulation",
                            "carbon",
                            "liquidity",
                            "service_level",
                        ]
                    },
                    "label": _TEXT,
                    "unit": _TEXT,
                    "sense": {"enum": ["less_than_or_equal", "greater_than_or_equal"]},
                    "limit": _NUMBER,
                    "enforcement": {"enum": ["deterministic", "chance"]},
                    "minimum_satisfaction_probability": {
                        "type": "number",
                        "exclusiveMinimum": 0,
                        "maximum": 1,
                    },
                    "source_reference": _TEXT,
                },
                "additionalProperties": False,
            },
        },
        "information_action": {
            "type": "object",
            "required": ["action_id", "resolution", "cost"],
            "properties": {
                "action_id": _ID,
                "resolution": {"const": "perfect_state"},
                "cost": {
                    "type": "object",
                    "required": ["amount", "unit", "placement", "scope"],
                    "properties": {
                        "amount": {"type": "number", "minimum": 0},
                        "unit": _TEXT,
                        "placement": {"const": "deduct_after_objective_comparison"},
                        "scope": {"const": "action_specific_disjoint"},
                    },
                    "additionalProperties": False,
                },
            },
            "additionalProperties": False,
        },
        "tolerances": {
            "type": "object",
            "required": ["absolute_tie", "relative_tie", "probability_sum"],
            "properties": {
                "absolute_tie": {"type": "number", "minimum": 0},
                "relative_tie": {"type": "number", "minimum": 0},
                "probability_sum": {
                    "type": "number",
                    "exclusiveMinimum": 0,
                    "maximum": 1e-6,
                },
            },
            "additionalProperties": False,
        },
        "assurance": {
            "type": "object",
            "required": [
                "estimator",
                "max_policy_mappings",
                "independent_reference",
                "fixture_provenance",
            ],
            "properties": {
                "estimator": {"const": "exact_finite_enumeration"},
                "max_policy_mappings": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100000,
                },
                "independent_reference": _TEXT,
                "fixture_provenance": _TEXT,
            },
            "additionalProperties": False,
        },
        "provenance": {
            "type": "object",
            "required": ["model_source", "probability_source", "constraint_source"],
            "properties": {
                "model_source": _TEXT,
                "probability_source": _TEXT,
                "constraint_source": _TEXT,
            },
            "additionalProperties": False,
        },
    },
    "additionalProperties": False,
}

_SCORE_RECORD: Final[dict[str, object]] = {
    "type": "object",
    "required": ["objective_score", "expected_value", "worst_case_regret"],
    "properties": {
        "objective_score": _NUMBER,
        "expected_value": _NUMBER,
        "worst_case_regret": {"type": ["number", "null"]},
    },
}
_CONSTRAINT_DIAGNOSTIC: Final[dict[str, object]] = {
    "type": "object",
    "required": [
        "constraint_id",
        "kind",
        "unit",
        "enforcement",
        "required_satisfaction_probability",
        "satisfaction_probability",
        "feasible",
        "worst_slack",
        "slack_by_state",
        "violating_state_ids",
    ],
    "properties": {
        "constraint_id": _ID,
        "kind": _TEXT,
        "unit": _TEXT,
        "enforcement": {"enum": ["deterministic", "chance"]},
        "required_satisfaction_probability": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
        },
        "satisfaction_probability": {"type": "number", "minimum": 0, "maximum": 1},
        "feasible": {"type": "boolean"},
        "worst_slack": _NUMBER,
        "slack_by_state": {
            "type": "object",
            "minProperties": 1,
            "additionalProperties": _NUMBER,
        },
        "violating_state_ids": {"type": "array", "uniqueItems": True, "items": _ID},
    },
    "additionalProperties": False,
}

RISK_SENSITIVE_VOI_RESULT_SCHEMA_V1: Final[dict[str, object]] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://voiage.dev/schemas/frontier/risk-sensitive-voi-result.v1.json",
    "title": "RiskSensitiveConstrainedVoiResultV1Experimental",
    "type": "object",
    "required": [
        "schema_version",
        "analysis_id",
        "analysis_type",
        "method_maturity",
        "planned_version",
        "objective",
        "baseline",
        "perfect_information",
        "value",
        "switches",
        "risk_diagnostics",
        "constraint_diagnostics",
        "shadow_value_evidence",
        "enumeration",
        "provenance",
    ],
    "properties": {
        "schema_version": {"const": "1.0.0"},
        "analysis_id": _ID,
        "analysis_type": {
            "const": "risk_sensitive_constrained_perfect_information_result"
        },
        "method_maturity": {"const": "experimental"},
        "planned_version": {"const": "v1.3.0"},
        "objective": cast(
            "dict[str, object]",
            cast("dict[str, object]", RISK_SENSITIVE_VOI_INPUT_SCHEMA_V1["properties"])[
                "objective"
            ],
        ),
        "baseline": {
            **_SCORE_RECORD,
            "required": [
                "selected_policy_id",
                "tied_policy_ids",
                *cast("list[str]", _SCORE_RECORD["required"]),
            ],
            "properties": {
                **cast("dict[str, object]", _SCORE_RECORD["properties"]),
                "selected_policy_id": _ID,
                "tied_policy_ids": {
                    "type": "array",
                    "minItems": 1,
                    "uniqueItems": True,
                    "items": _ID,
                },
            },
            "additionalProperties": False,
        },
        "perfect_information": {
            **_SCORE_RECORD,
            "required": [
                "selected_policy_by_state",
                "tied_policy_mappings",
                *cast("list[str]", _SCORE_RECORD["required"]),
            ],
            "properties": {
                **cast("dict[str, object]", _SCORE_RECORD["properties"]),
                "selected_policy_by_state": {
                    "type": "object",
                    "minProperties": 1,
                    "additionalProperties": _ID,
                },
                "tied_policy_mappings": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "minProperties": 1,
                        "additionalProperties": _ID,
                    },
                },
            },
            "additionalProperties": False,
        },
        "value": {
            "type": "object",
            "required": ["gross", "information_cost", "net", "unit"],
            "properties": {
                "gross": _NUMBER,
                "information_cost": {"type": "number", "minimum": 0},
                "net": _NUMBER,
                "unit": _TEXT,
            },
            "additionalProperties": False,
        },
        "switches": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["state_id", "from_policy_id", "to_policy_id"],
                "properties": {
                    "state_id": _ID,
                    "from_policy_id": _ID,
                    "to_policy_id": _ID,
                },
                "additionalProperties": False,
            },
        },
        "risk_diagnostics": {
            "type": "object",
            "required": [
                "objective_kind",
                "baseline_objective_score",
                "perfect_information_objective_score",
                "baseline_expected_value",
                "perfect_information_expected_value",
                "baseline_worst_case_regret",
                "perfect_information_worst_case_regret",
                "lower_tail_mass",
            ],
            "properties": {
                "objective_kind": _TEXT,
                "baseline_objective_score": _NUMBER,
                "perfect_information_objective_score": _NUMBER,
                "baseline_expected_value": _NUMBER,
                "perfect_information_expected_value": _NUMBER,
                "baseline_worst_case_regret": {"type": ["number", "null"]},
                "perfect_information_worst_case_regret": {"type": ["number", "null"]},
                "lower_tail_mass": {"type": ["number", "null"]},
            },
            "additionalProperties": False,
        },
        "constraint_diagnostics": {
            "type": "object",
            "required": ["baseline", "perfect_information"],
            "properties": {
                "baseline": {"type": "array", "items": _CONSTRAINT_DIAGNOSTIC},
                "perfect_information": {
                    "type": "array",
                    "items": _CONSTRAINT_DIAGNOSTIC,
                },
            },
            "additionalProperties": False,
        },
        "shadow_value_evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "required": [
                    "constraint_id",
                    "shadow_value_status",
                    "method",
                    "gross_voi_without_constraint",
                    "constraint_removal_effect_on_gross_voi",
                    "unit",
                ],
                "properties": {
                    "constraint_id": _ID,
                    "shadow_value_status": {"const": "not_a_local_shadow_price"},
                    "method": {"const": "exact_constraint_removal"},
                    "gross_voi_without_constraint": _NUMBER,
                    "constraint_removal_effect_on_gross_voi": _NUMBER,
                    "unit": _TEXT,
                },
                "additionalProperties": False,
            },
        },
        "enumeration": {
            "type": "object",
            "required": [
                "exact",
                "estimator",
                "tie_policy",
                "policy_count",
                "state_count",
                "mapping_count_evaluated",
                "feasible_baseline_count",
                "feasible_mapping_count",
            ],
            "properties": {
                "exact": {"const": True},
                "estimator": {"const": "exact_finite_enumeration"},
                "tie_policy": {
                    "const": "exact_argmax_lexicographic_with_tolerance_ties"
                },
                "policy_count": {"type": "integer", "minimum": 2},
                "state_count": {"type": "integer", "minimum": 1},
                "mapping_count_evaluated": {"type": "integer", "minimum": 1},
                "feasible_baseline_count": {"type": "integer", "minimum": 1},
                "feasible_mapping_count": {"type": "integer", "minimum": 1},
            },
            "additionalProperties": False,
        },
        "provenance": cast(
            "dict[str, object]",
            cast("dict[str, object]", RISK_SENSITIVE_VOI_INPUT_SCHEMA_V1["properties"])[
                "provenance"
            ],
        ),
    },
    "additionalProperties": False,
}

_INPUT_VALIDATOR = Draft202012Validator(RISK_SENSITIVE_VOI_INPUT_SCHEMA_V1)
_RESULT_VALIDATOR = Draft202012Validator(RISK_SENSITIVE_VOI_RESULT_SCHEMA_V1)


def _reject_non_finite(value: object, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_non_finite(item, f"{path}/{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_non_finite(item, f"{path}/{index}")
    elif (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and not math.isfinite(float(value))
    ):
        raise_input_error(f"{path}: numeric values must be finite")


def _schema_error(validator: Any, payload: Mapping[str, Any]) -> None:
    errors = sorted(validator.iter_errors(payload), key=lambda error: list(error.path))
    if errors:
        error = errors[0]
        location = "/".join(str(item) for item in error.path) or "root"
        raise_input_error(f"{location}: {error.message}")


def validate_risk_sensitive_voi_semantics(payload: Mapping[str, Any]) -> None:
    """Validate schema and cross-field semantics for the finite v1 request."""
    try:
        _schema_error(_INPUT_VALIDATOR, payload)
        _reject_non_finite(payload)
        states = cast("list[Mapping[str, Any]]", payload["states"])
        state_ids = [cast("str", state["state_id"]) for state in states]
        if len(state_ids) != len(set(state_ids)):
            raise ValueError("state IDs must be unique")
        tolerance = float(
            cast("Mapping[str, Any]", payload["tolerances"])["probability_sum"]
        )
        if not math.isclose(
            math.fsum(float(state["probability"]) for state in states),
            1.0,
            abs_tol=tolerance,
            rel_tol=0,
        ):
            raise ValueError("state probabilities must sum to 1")
        policies = cast("list[Mapping[str, Any]]", payload["policies"])
        policy_ids = [cast("str", policy["policy_id"]) for policy in policies]
        if len(policy_ids) != len(set(policy_ids)):
            raise ValueError("policy IDs must be unique")
        constraints = cast("list[Mapping[str, Any]]", payload["constraints"])
        constraint_ids = [cast("str", item["constraint_id"]) for item in constraints]
        if len(constraint_ids) != len(set(constraint_ids)):
            raise ValueError("constraint IDs must be unique")
        expected_states = set(state_ids)
        expected_constraints = set(constraint_ids)
        for policy in policies:
            if (
                set(cast("Mapping[str, Any]", policy["objective_by_state"]))
                != expected_states
            ):
                raise ValueError(
                    "policy objective state keys must exactly match states"
                )
            usage = cast("Mapping[str, Mapping[str, Any]]", policy["constraint_usage"])
            if set(usage) != expected_constraints or any(
                set(row) != expected_states for row in usage.values()
            ):
                raise ValueError(
                    "policy constraint usage keys must exactly match constraints and states"
                )
        for constraint in constraints:
            chance = constraint["enforcement"] == "chance"
            if chance != ("minimum_satisfaction_probability" in constraint):
                raise ValueError(
                    "chance constraints alone require minimum_satisfaction_probability"
                )
        objective = cast("Mapping[str, Any]", payload["objective"])
        if (objective["kind"] == "lower_tail_cvar") != (
            "confidence_level" in objective
        ):
            raise ValueError("lower_tail_cvar alone requires confidence_level")
        if (objective["kind"] == "minimax_regret") != (
            "regret_reference_by_state" in objective
        ):
            raise ValueError("minimax_regret alone requires regret_reference_by_state")
        if (
            "regret_reference_by_state" in objective
            and set(cast("Mapping[str, Any]", objective["regret_reference_by_state"]))
            != expected_states
        ):
            raise ValueError("regret reference state keys must exactly match states")
        if "regret_reference_by_state" in objective:
            reference = cast(
                "Mapping[str, float]", objective["regret_reference_by_state"]
            )
            for policy in policies:
                values = cast("Mapping[str, float]", policy["objective_by_state"])
                if any(
                    float(reference[state]) < float(values[state])
                    for state in state_ids
                ):
                    raise ValueError(
                        "regret references must weakly exceed every policy objective by state"
                    )
        cost = cast(
            "Mapping[str, Any]",
            cast("Mapping[str, Any]", payload["information_action"])["cost"],
        )
        if cost["unit"] != objective["unit"]:
            raise ValueError("information cost unit must equal objective unit")
        mappings = len(policies) ** len(states)
        limit = int(
            cast("Mapping[str, Any]", payload["assurance"])["max_policy_mappings"]
        )
        if mappings > limit:
            raise ValueError(
                f"policy mapping count {mappings} exceeds max_policy_mappings {limit}"
            )
    except (KeyError, TypeError, ValueError, ValidationError) as error:
        if isinstance(error, InputError):
            raise
        raise_input_error(str(error))


def validate_risk_sensitive_voi_result(payload: Mapping[str, Any]) -> None:
    """Validate the portable v1 result envelope."""
    _schema_error(_RESULT_VALIDATOR, payload)
    _reject_non_finite(payload)


__all__ = [
    "RISK_SENSITIVE_VOI_INPUT_SCHEMA_V1",
    "RISK_SENSITIVE_VOI_RESULT_SCHEMA_V1",
    "validate_risk_sensitive_voi_result",
    "validate_risk_sensitive_voi_semantics",
]
