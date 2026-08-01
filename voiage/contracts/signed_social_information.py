"""Strict contracts for exact finite signed and social information value."""

# pyright: reportAny=false, reportExplicitAny=false, reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportMissingModuleSource=false

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any, Final, cast

from jsonschema import Draft202012Validator

from voiage.exceptions import raise_input_error

_ID: Final[dict[str, object]] = {
    "type": "string",
    "minLength": 1,
    "pattern": r"^[A-Za-z][A-Za-z0-9._-]*$",
}
_STRING: Final[dict[str, object]] = {"type": "string", "minLength": 1}
_NUMBER: Final[dict[str, object]] = {"type": "number"}

SIGNED_SOCIAL_INFORMATION_INPUT_SCHEMA_V1: Final[dict[str, object]] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://voiage.dev/schemas/frontier/signed-social-information-input.v1.json",
    "title": "SignedSocialInformationInputV1Experimental",
    "type": "object",
    "required": [
        "schema_version",
        "analysis_id",
        "analysis_type",
        "method_maturity",
        "value_unit",
        "purpose",
        "agents",
        "welfare",
        "topology",
        "actions",
        "worlds",
        "policies",
        "receipts",
        "baseline_design_id",
        "designs",
        "tie_policy",
        "provenance",
    ],
    "properties": {
        "schema_version": {"const": "1.0.0"},
        "analysis_id": _ID,
        "analysis_type": {"const": "signed_social_information_value"},
        "method_maturity": {"const": "experimental"},
        "value_unit": _STRING,
        "purpose": _ID,
        "agents": {
            "type": "array",
            "minItems": 2,
            "items": {
                "type": "object",
                "required": ["agent_id", "roles"],
                "properties": {
                    "agent_id": _ID,
                    "roles": {
                        "type": "array",
                        "minItems": 1,
                        "uniqueItems": True,
                        "items": {
                            "enum": [
                                "decision_maker",
                                "recipient",
                                "controller",
                                "stakeholder",
                            ]
                        },
                    },
                },
                "additionalProperties": False,
            },
        },
        "welfare": {
            "type": "object",
            "required": [
                "aggregator",
                "cardinal_comparability",
                "ledger_stage",
                "weights",
                "rationale",
            ],
            "properties": {
                "aggregator": {"const": "weighted_sum"},
                "cardinal_comparability": {"const": "declared"},
                "ledger_stage": {"enum": ["pre_transfer", "post_transfer"]},
                "weights": {
                    "type": "object",
                    "minProperties": 2,
                    "additionalProperties": {"type": "number", "minimum": 0},
                },
                "rationale": _STRING,
            },
            "additionalProperties": False,
        },
        "topology": {
            "type": "object",
            "required": [
                "signal_id",
                "source_agent_id",
                "controller_agent_id",
                "information_scope",
                "eligible_recipients",
                "baseline_recipients",
            ],
            "properties": {
                "signal_id": _ID,
                "source_agent_id": _ID,
                "controller_agent_id": _ID,
                "information_scope": {"enum": ["private", "public", "team"]},
                "eligible_recipients": {
                    "type": "array",
                    "uniqueItems": True,
                    "items": _ID,
                },
                "baseline_recipients": {
                    "type": "array",
                    "uniqueItems": True,
                    "items": _ID,
                },
            },
            "additionalProperties": False,
        },
        "actions": {
            "type": "array",
            "minItems": 2,
            "uniqueItems": True,
            "items": _ID,
        },
        "worlds": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": [
                    "world_id",
                    "probability",
                    "signal",
                    "action_utilities",
                ],
                "properties": {
                    "world_id": _ID,
                    "probability": {
                        "type": "number",
                        "exclusiveMinimum": 0,
                        "maximum": 1,
                    },
                    "signal": _ID,
                    "action_utilities": {
                        "type": "object",
                        "minProperties": 2,
                        "additionalProperties": {
                            "type": "object",
                            "minProperties": 2,
                            "additionalProperties": _NUMBER,
                        },
                    },
                },
                "additionalProperties": False,
            },
        },
        "policies": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["policy_id", "decision_agent_id", "decisions"],
                "properties": {
                    "policy_id": _ID,
                    "decision_agent_id": _ID,
                    "decisions": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "required": ["observation", "action_id"],
                            "properties": {
                                "observation": _ID,
                                "action_id": _ID,
                            },
                            "additionalProperties": False,
                        },
                    },
                },
                "additionalProperties": False,
            },
        },
        "receipts": {
            "type": "array",
            "items": {
                "type": "object",
                "required": [
                    "receipt_id",
                    "subject_agent_id",
                    "consent_status",
                    "purpose",
                    "legal_basis",
                    "data_scope",
                ],
                "properties": {
                    "receipt_id": _ID,
                    "subject_agent_id": _ID,
                    "consent_status": {"enum": ["granted", "denied", "not_required"]},
                    "purpose": _ID,
                    "legal_basis": _STRING,
                    "data_scope": _STRING,
                },
                "additionalProperties": False,
            },
        },
        "baseline_design_id": _ID,
        "designs": {
            "type": "array",
            "minItems": 2,
            "items": {
                "type": "object",
                "required": [
                    "design_id",
                    "comparator_design_id",
                    "recipients",
                    "policy_ids",
                    "selection_mode",
                    "selector",
                    "selected_policy_id",
                    "transfers",
                    "costs",
                    "rights_receipt_ids",
                    "blackwell_assurance",
                    "equilibrium_receipt",
                ],
                "properties": {
                    "design_id": _ID,
                    "comparator_design_id": {"oneOf": [_ID, {"type": "null"}]},
                    "recipients": {
                        "type": "array",
                        "uniqueItems": True,
                        "items": _ID,
                    },
                    "policy_ids": {
                        "type": "array",
                        "minItems": 1,
                        "uniqueItems": True,
                        "items": _ID,
                    },
                    "selection_mode": {
                        "enum": [
                            "centralized",
                            "fixed",
                            "declared_response",
                            "verified_finite_equilibrium",
                        ]
                    },
                    "selector": _STRING,
                    "selected_policy_id": {"oneOf": [_ID, {"type": "null"}]},
                    "transfers": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": [
                                "payer_agent_id",
                                "recipient_agent_id",
                                "amount",
                            ],
                            "properties": {
                                "payer_agent_id": _ID,
                                "recipient_agent_id": _ID,
                                "amount": {"type": "number", "minimum": 0},
                            },
                            "additionalProperties": False,
                        },
                    },
                    "costs": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["agent_id", "category", "amount"],
                            "properties": {
                                "agent_id": _ID,
                                "category": {
                                    "enum": [
                                        "information",
                                        "privacy",
                                        "implementation",
                                        "other_declared",
                                    ]
                                },
                                "amount": {"type": "number", "minimum": 0},
                            },
                            "additionalProperties": False,
                        },
                    },
                    "rights_receipt_ids": {
                        "type": "array",
                        "uniqueItems": True,
                        "items": _ID,
                    },
                    "blackwell_assurance": {
                        "oneOf": [
                            {"type": "null"},
                            {
                                "type": "object",
                                "required": [
                                    "refines_design_id",
                                    "signal_refinement_verified",
                                    "same_preferences",
                                    "same_constraints",
                                    "baseline_catalog_embedded",
                                ],
                                "properties": {
                                    "refines_design_id": _ID,
                                    "signal_refinement_verified": {"type": "boolean"},
                                    "same_preferences": {"type": "boolean"},
                                    "same_constraints": {"type": "boolean"},
                                    "baseline_catalog_embedded": {"type": "boolean"},
                                },
                                "additionalProperties": False,
                            },
                        ]
                    },
                    "equilibrium_receipt": {
                        "oneOf": [
                            {"type": "null"},
                            {
                                "type": "object",
                                "required": [
                                    "solution_concept",
                                    "verification_method",
                                    "verified_policy_ids",
                                ],
                                "properties": {
                                    "solution_concept": _STRING,
                                    "verification_method": {
                                        "const": "complete_catalog_best_response_check"
                                    },
                                    "verified_policy_ids": {
                                        "type": "array",
                                        "minItems": 1,
                                        "uniqueItems": True,
                                        "items": _ID,
                                    },
                                },
                                "additionalProperties": False,
                            },
                        ]
                    },
                },
                "additionalProperties": False,
            },
        },
        "tie_policy": {
            "type": "object",
            "required": ["absolute_tolerance", "relative_tolerance"],
            "properties": {
                "absolute_tolerance": {"type": "number", "minimum": 0},
                "relative_tolerance": {"type": "number", "minimum": 0},
            },
            "additionalProperties": False,
        },
        "provenance": {
            "type": "object",
            "required": [
                "world_law_source",
                "utility_source",
                "policy_source",
                "rights_source",
            ],
            "properties": {
                "world_law_source": _STRING,
                "utility_source": _STRING,
                "policy_source": _STRING,
                "rights_source": _STRING,
            },
            "additionalProperties": False,
        },
    },
    "additionalProperties": False,
}

SIGNED_SOCIAL_INFORMATION_RESULT_SCHEMA_V1: Final[dict[str, object]] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://voiage.dev/schemas/frontier/signed-social-information-result.v1.json",
    "title": "SignedSocialInformationResultV1Experimental",
    "$defs": {
        "number_map": {"type": "object", "additionalProperties": _NUMBER},
        "ledger": {
            "type": "object",
            "required": ["pre_transfer", "transfer", "cost", "post_transfer"],
            "properties": {
                "pre_transfer": {"$ref": "#/$defs/number_map"},
                "transfer": {"$ref": "#/$defs/number_map"},
                "cost": {"$ref": "#/$defs/number_map"},
                "post_transfer": {"$ref": "#/$defs/number_map"},
            },
            "additionalProperties": False,
        },
        "blackwell": {
            "type": "object",
            "required": [
                "applicable",
                "checked_value",
                "passed",
                "reasons_not_applicable",
            ],
            "properties": {
                "applicable": {"type": "boolean"},
                "checked_value": {"oneOf": [_NUMBER, {"type": "null"}]},
                "passed": {"oneOf": [{"type": "boolean"}, {"type": "null"}]},
                "reasons_not_applicable": {
                    "type": "array",
                    "uniqueItems": True,
                    "items": _ID,
                },
            },
            "additionalProperties": False,
        },
        "signed_values": {
            "type": "object",
            "required": [
                "by_agent",
                "by_role",
                "social",
                "comparator_design_id",
                "clipped_at_zero",
            ],
            "properties": {
                "by_agent": {"$ref": "#/$defs/number_map"},
                "by_role": {"$ref": "#/$defs/number_map"},
                "social": _NUMBER,
                "comparator_design_id": {"oneOf": [_ID, {"type": "null"}]},
                "clipped_at_zero": {"const": False},
            },
            "additionalProperties": False,
        },
        "design": {
            "type": "object",
            "required": [
                "design_id",
                "comparator_design_id",
                "recipients",
                "selection_mode",
                "selector",
                "feasible",
                "infeasibility_reasons",
                "policies_evaluated",
                "policy_selector_values",
                "policy_tie",
                "selected_policy_id",
                "ledgers",
                "social_pre_transfer",
                "social_post_transfer",
                "rights_receipts",
                "equilibrium_receipt",
                "signed_values",
                "policy_switch",
                "blackwell_nonnegativity",
            ],
            "properties": {
                "design_id": _ID,
                "comparator_design_id": {"oneOf": [_ID, {"type": "null"}]},
                "recipients": {
                    "type": "array",
                    "uniqueItems": True,
                    "items": _ID,
                },
                "selection_mode": {
                    "enum": [
                        "centralized",
                        "fixed",
                        "declared_response",
                        "verified_finite_equilibrium",
                    ]
                },
                "selector": _STRING,
                "feasible": {"type": "boolean"},
                "infeasibility_reasons": {
                    "type": "array",
                    "uniqueItems": True,
                    "items": _STRING,
                },
                "policies_evaluated": {
                    "type": "array",
                    "minItems": 1,
                    "uniqueItems": True,
                    "items": _ID,
                },
                "policy_selector_values": {"$ref": "#/$defs/number_map"},
                "policy_tie": {
                    "type": "array",
                    "minItems": 1,
                    "uniqueItems": True,
                    "items": _ID,
                },
                "selected_policy_id": _ID,
                "ledgers": {"$ref": "#/$defs/ledger"},
                "social_pre_transfer": _NUMBER,
                "social_post_transfer": _NUMBER,
                "rights_receipts": {
                    "type": "array",
                    "uniqueItems": True,
                    "items": _ID,
                },
                "equilibrium_receipt": {
                    "oneOf": [
                        {"type": "null"},
                        {
                            "type": "object",
                            "required": [
                                "solution_concept",
                                "verification_method",
                                "verified_policy_ids",
                            ],
                            "properties": {
                                "solution_concept": _STRING,
                                "verification_method": {
                                    "const": "complete_catalog_best_response_check"
                                },
                                "verified_policy_ids": {
                                    "type": "array",
                                    "minItems": 1,
                                    "uniqueItems": True,
                                    "items": _ID,
                                },
                            },
                            "additionalProperties": False,
                        },
                    ]
                },
                "signed_values": {"$ref": "#/$defs/signed_values"},
                "policy_switch": {"type": "boolean"},
                "blackwell_nonnegativity": {"$ref": "#/$defs/blackwell"},
            },
            "additionalProperties": False,
        },
    },
    "type": "object",
    "required": [
        "schema_version",
        "analysis_id",
        "analysis_type",
        "method_maturity",
        "value_unit",
        "welfare_contract",
        "topology",
        "baseline",
        "designs",
        "optimum",
        "diagnostics",
        "assurance",
        "language_dispositions",
        "unsupported_dispositions",
        "provenance",
    ],
    "properties": {
        "schema_version": {"const": "1.0.0"},
        "analysis_id": _ID,
        "analysis_type": {"const": "signed_social_information_value_result"},
        "method_maturity": {"const": "experimental"},
        "value_unit": _STRING,
        "welfare_contract": {
            "type": "object",
            "required": [
                "aggregator",
                "cardinal_comparability",
                "ledger_stage",
                "weights",
                "rationale",
            ],
            "properties": {
                "aggregator": {"const": "weighted_sum"},
                "cardinal_comparability": {"const": "declared"},
                "ledger_stage": {"enum": ["pre_transfer", "post_transfer"]},
                "weights": {"$ref": "#/$defs/number_map"},
                "rationale": _STRING,
            },
            "additionalProperties": False,
        },
        "topology": {
            "type": "object",
            "required": [
                "signal_id",
                "source_agent_id",
                "controller_agent_id",
                "information_scope",
                "eligible_recipients",
                "baseline_recipients",
            ],
            "properties": {
                "signal_id": _ID,
                "source_agent_id": _ID,
                "controller_agent_id": _ID,
                "information_scope": {"enum": ["private", "public", "team"]},
                "eligible_recipients": {"type": "array", "items": _ID},
                "baseline_recipients": {"type": "array", "items": _ID},
            },
            "additionalProperties": False,
        },
        "baseline": {"$ref": "#/$defs/design"},
        "designs": {
            "type": "array",
            "minItems": 2,
            "items": {"$ref": "#/$defs/design"},
        },
        "optimum": {
            "type": "object",
            "required": [
                "feasible_design_values",
                "design_tie",
                "selected_design_id",
                "social_value",
                "tie_policy",
            ],
            "properties": {
                "feasible_design_values": {"$ref": "#/$defs/number_map"},
                "design_tie": {"type": "array", "minItems": 1, "items": _ID},
                "selected_design_id": _ID,
                "social_value": _NUMBER,
                "tie_policy": {
                    "type": "object",
                    "required": ["absolute_tolerance", "relative_tolerance"],
                    "properties": {
                        "absolute_tolerance": _NUMBER,
                        "relative_tolerance": _NUMBER,
                    },
                    "additionalProperties": False,
                },
            },
            "additionalProperties": False,
        },
        "diagnostics": {
            "type": "object",
            "required": [
                "winners",
                "losers",
                "harmful_private_designs",
                "information_avoidance",
                "policy_switches",
                "winner_loser_design_id",
                "externality_by_design",
            ],
            "properties": {
                "winners": {"type": "array", "items": _ID},
                "losers": {"type": "array", "items": _ID},
                "harmful_private_designs": {"type": "array", "items": _ID},
                "information_avoidance": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["agent_id", "design_id"],
                        "properties": {"agent_id": _ID, "design_id": _ID},
                        "additionalProperties": False,
                    },
                },
                "policy_switches": {"type": "array", "items": _ID},
                "winner_loser_design_id": _ID,
                "externality_by_design": {"$ref": "#/$defs/number_map"},
            },
            "additionalProperties": False,
        },
        "assurance": {
            "type": "object",
            "required": [
                "worlds_evaluated",
                "policies_evaluated",
                "designs_evaluated",
                "complete_joint_world_law",
                "nonanticipativity",
                "finite_catalog_only",
                "general_game_solver_used",
                "negative_values_clipped",
                "rights_consent_purpose_receipts_checked",
                "deterministic_serialization",
            ],
            "properties": {
                "worlds_evaluated": {"type": "integer", "minimum": 1},
                "policies_evaluated": {"type": "integer", "minimum": 1},
                "designs_evaluated": {"type": "integer", "minimum": 2},
                "complete_joint_world_law": {"const": True},
                "nonanticipativity": _STRING,
                "finite_catalog_only": {"const": True},
                "general_game_solver_used": {"const": False},
                "negative_values_clipped": {"const": False},
                "rights_consent_purpose_receipts_checked": {"const": True},
                "deterministic_serialization": {"const": True},
            },
            "additionalProperties": False,
        },
        "language_dispositions": {
            "type": "object",
            "required": ["python", "rust", "r", "julia", "mojo"],
            "properties": {
                "python": {"const": "experimental_exact_finite_execution"},
                "rust": {"const": "not_implemented"},
                "r": {"const": "not_implemented"},
                "julia": {"const": "not_implemented"},
                "mojo": {"const": "external_upstream_boundary"},
            },
            "additionalProperties": False,
        },
        "unsupported_dispositions": {
            "type": "object",
            "required": [
                "bayesian_persuasion",
                "mechanism_design",
                "rational_inattention",
                "general_game_solving",
                "continuous_or_incomplete_world_laws",
            ],
            "properties": {
                "bayesian_persuasion": _STRING,
                "mechanism_design": _STRING,
                "rational_inattention": _STRING,
                "general_game_solving": _STRING,
                "continuous_or_incomplete_world_laws": _STRING,
            },
            "additionalProperties": False,
        },
        "provenance": {
            "type": "object",
            "required": [
                "world_law_source",
                "utility_source",
                "policy_source",
                "rights_source",
            ],
            "properties": {
                "world_law_source": _STRING,
                "utility_source": _STRING,
                "policy_source": _STRING,
                "rights_source": _STRING,
            },
            "additionalProperties": False,
        },
    },
    "additionalProperties": False,
}

_INPUT_VALIDATOR = Draft202012Validator(SIGNED_SOCIAL_INFORMATION_INPUT_SCHEMA_V1)
_RESULT_VALIDATOR = Draft202012Validator(SIGNED_SOCIAL_INFORMATION_RESULT_SCHEMA_V1)


def _validate(validator: Draft202012Validator, payload: Mapping[str, object]) -> None:
    error = next(iter(validator.iter_errors(cast("Any", dict(payload)))), None)
    if error is not None:
        path = "/".join(str(part) for part in error.absolute_path) or "<root>"
        raise ValueError(f"{path}: {error.message}")


def _ensure_finite(value: object, path: str = "<root>") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path}: numeric values must be finite")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _ensure_finite(item, f"{path}/{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _ensure_finite(item, f"{path}/{index}")


def validate_signed_social_information_semantics(
    payload: Mapping[str, object],
) -> None:
    """Validate the strict schema and cross-record finite-model invariants."""
    _validate(_INPUT_VALIDATOR, payload)
    _ensure_finite(payload)
    data = cast("dict[str, Any]", dict(payload))
    agents = cast("list[dict[str, Any]]", data["agents"])
    agent_ids = [str(agent["agent_id"]) for agent in agents]
    if len(agent_ids) != len(set(agent_ids)):
        raise ValueError("agent identifiers must be unique")
    agent_set = set(agent_ids)
    weights = cast("dict[str, float]", data["welfare"]["weights"])
    if set(weights) != agent_set:
        raise ValueError("welfare weights must contain exactly the declared agents")
    if not any(float(weight) > 0.0 for weight in weights.values()):
        raise ValueError("at least one welfare weight must be positive")
    if data["welfare"]["cardinal_comparability"] != "declared":  # pragma: no cover
        raise ValueError("cardinal comparability must be explicitly declared")

    topology = cast("dict[str, Any]", data["topology"])
    for key in ("source_agent_id", "controller_agent_id"):
        if topology[key] not in agent_set:
            raise ValueError(f"topology {key} references an unknown agent")
    eligible = set(cast("list[str]", topology["eligible_recipients"]))
    baseline_recipients = set(cast("list[str]", topology["baseline_recipients"]))
    if not eligible.issubset(agent_set) or not baseline_recipients.issubset(eligible):
        raise ValueError("topology recipients must be declared eligible agents")

    actions = cast("list[str]", data["actions"])
    action_set = set(actions)
    worlds = cast("list[dict[str, Any]]", data["worlds"])
    world_ids = [str(world["world_id"]) for world in worlds]
    if len(world_ids) != len(set(world_ids)):
        raise ValueError("world identifiers must be unique")
    if not math.isclose(
        math.fsum(float(world["probability"]) for world in worlds),
        1.0,
        abs_tol=1e-12,
    ):
        raise ValueError("world probabilities must sum to one")
    signals = {str(world["signal"]) for world in worlds}
    for world in worlds:
        utilities = cast("dict[str, dict[str, float]]", world["action_utilities"])
        if set(utilities) != action_set:
            raise ValueError("each world must contain every declared action")
        if any(set(values) != agent_set for values in utilities.values()):
            raise ValueError("each action utility must contain every declared agent")

    policies = cast("list[dict[str, Any]]", data["policies"])
    policy_ids = [str(policy["policy_id"]) for policy in policies]
    if len(policy_ids) != len(set(policy_ids)):
        raise ValueError("policy identifiers must be unique")
    policy_set = set(policy_ids)
    policy_observations: dict[str, set[str]] = {}
    for policy in policies:
        if policy["decision_agent_id"] not in agent_set:
            raise ValueError("policy decision_agent_id references an unknown agent")
        decisions = cast("list[dict[str, str]]", policy["decisions"])
        observations = [decision["observation"] for decision in decisions]
        if len(observations) != len(set(observations)):
            raise ValueError("policy observations must be unique")
        observation_set = set(observations)
        if observation_set not in ({"unobserved"}, signals):
            raise ValueError(
                "policy observations must be exactly unobserved or every signal"
            )
        if any(decision["action_id"] not in action_set for decision in decisions):
            raise ValueError("policy decision references an unknown action")
        policy_observations[str(policy["policy_id"])] = observation_set

    receipts = cast("list[dict[str, Any]]", data["receipts"])
    receipt_ids = [str(receipt["receipt_id"]) for receipt in receipts]
    if len(receipt_ids) != len(set(receipt_ids)):
        raise ValueError("receipt identifiers must be unique")
    receipt_by_id = {str(item["receipt_id"]): item for item in receipts}
    for receipt in receipts:
        if receipt["subject_agent_id"] not in agent_set:
            raise ValueError("receipt subject references an unknown agent")
        if receipt["purpose"] != data["purpose"]:
            raise ValueError("every rights receipt must match the declared purpose")

    designs = cast("list[dict[str, Any]]", data["designs"])
    design_ids = [str(design["design_id"]) for design in designs]
    if len(design_ids) != len(set(design_ids)):
        raise ValueError("design identifiers must be unique")
    design_set = set(design_ids)
    if data["baseline_design_id"] not in design_set:
        raise ValueError("baseline_design_id references an unknown design")
    baseline = next(
        design
        for design in designs
        if design["design_id"] == data["baseline_design_id"]
    )
    if baseline["comparator_design_id"] is not None:
        raise ValueError("baseline design must not declare a comparator")
    if set(baseline["recipients"]) != baseline_recipients:
        raise ValueError("baseline design recipients must match topology")

    for design in designs:
        design_id = str(design["design_id"])
        comparator = design["comparator_design_id"]
        if design_id != data["baseline_design_id"] and comparator not in design_set:
            raise ValueError("every non-baseline design needs a known comparator")
        if comparator == design_id:
            raise ValueError("a design cannot be its own comparator")
        recipients = set(cast("list[str]", design["recipients"]))
        if not recipients.issubset(eligible):
            raise ValueError("design recipients must be eligible")
        catalog = set(cast("list[str]", design["policy_ids"]))
        if not catalog.issubset(policy_set):
            raise ValueError("design policy catalog references an unknown policy")
        if not recipients and any(
            policy_observations[policy_id] != {"unobserved"} for policy_id in catalog
        ):
            raise ValueError("unshared designs may only use unobserved policies")
        if recipients and any(  # pragma: no cover
            policy_observations[policy_id] not in ({"unobserved"}, signals)
            for policy_id in catalog
        ):  # pragma: no cover - policy validation proves this
            raise ValueError("shared policies violate nonanticipativity")
        selector = str(design["selector"])
        if selector != "social_welfare" and (
            not selector.startswith("agent:") or selector[6:] not in agent_set
        ):
            raise ValueError("selector must be social_welfare or agent:<agent_id>")
        mode = str(design["selection_mode"])
        selected = design["selected_policy_id"]
        if mode == "centralized" and selected is not None:
            raise ValueError("centralized selection must derive its selected policy")
        if mode != "centralized" and selected not in catalog:
            raise ValueError("non-centralized selection requires a catalog policy")
        equilibrium = design["equilibrium_receipt"]
        if mode == "verified_finite_equilibrium":
            if (
                equilibrium is None
                or selected not in equilibrium["verified_policy_ids"]
            ):
                raise ValueError("verified equilibrium requires a matching receipt")
            if not set(equilibrium["verified_policy_ids"]).issubset(catalog):
                raise ValueError("equilibrium receipt must reference catalog policies")
        elif equilibrium is not None:
            raise ValueError("equilibrium receipt is only valid for verified catalogs")
        for transfer in cast("list[dict[str, Any]]", design["transfers"]):
            if (
                transfer["payer_agent_id"] not in agent_set
                or transfer["recipient_agent_id"] not in agent_set
            ):
                raise ValueError("transfer references an unknown agent")
            if transfer["payer_agent_id"] == transfer["recipient_agent_id"]:
                raise ValueError("transfer payer and recipient must differ")
        for cost in cast("list[dict[str, Any]]", design["costs"]):
            if cost["agent_id"] not in agent_set:
                raise ValueError("cost references an unknown agent")
        design_receipts = set(cast("list[str]", design["rights_receipt_ids"]))
        if not design_receipts.issubset(receipt_by_id):
            raise ValueError("design references an unknown rights receipt")
        if recipients:
            covered = {
                str(receipt_by_id[receipt_id]["subject_agent_id"])
                for receipt_id in design_receipts
            }
            required = recipients | {str(topology["controller_agent_id"])}
            if not required.issubset(covered):
                raise ValueError(
                    "shared designs require recipient and controller rights receipts"
                )
        blackwell = design["blackwell_assurance"]
        if blackwell is not None:
            if blackwell["refines_design_id"] != comparator:
                raise ValueError("Blackwell assurance must name the design comparator")
            if not all(
                bool(blackwell[key])
                for key in (
                    "signal_refinement_verified",
                    "same_preferences",
                    "same_constraints",
                    "baseline_catalog_embedded",
                )
            ):
                raise ValueError("Blackwell assurance cannot assert false conditions")
            comparator_design = next(
                item for item in designs if item["design_id"] == comparator
            )
            if not set(comparator_design["policy_ids"]).issubset(catalog):
                raise ValueError("Blackwell baseline catalog must actually be embedded")


def validate_signed_social_information_result(
    payload: Mapping[str, object],
) -> None:
    """Validate the portable result envelope and signed accounting identities."""
    _validate(_RESULT_VALIDATOR, payload)
    _ensure_finite(payload)
    data = cast("dict[str, Any]", dict(payload))
    baseline = cast("dict[str, Any]", data["baseline"])
    designs = cast("list[dict[str, Any]]", data["designs"])
    design_ids = [str(design["design_id"]) for design in designs]
    if len(design_ids) != len(set(design_ids)):
        raise ValueError("result design identifiers must be unique")
    design_by_id = {str(design["design_id"]): design for design in designs}
    baseline_id = str(baseline["design_id"])
    if baseline_id not in design_by_id or baseline != design_by_id[baseline_id]:
        raise ValueError("result baseline must match one evaluated design")
    if baseline["comparator_design_id"] is not None:
        raise ValueError("result baseline must not declare a comparator")
    weights = cast("dict[str, float]", data["welfare_contract"]["weights"])
    agent_ids = set(weights)
    ledger_stage = str(data["welfare_contract"]["ledger_stage"])
    for design in designs:
        ledgers = cast("dict[str, dict[str, float]]", design["ledgers"])
        if any(set(ledger) != agent_ids for ledger in ledgers.values()):
            raise ValueError("result ledgers must contain exactly the welfare agents")
        for agent_id, pre in ledgers["pre_transfer"].items():
            expected_post = (
                float(pre)
                + float(ledgers["transfer"][agent_id])
                - float(ledgers["cost"][agent_id])
            )
            if not math.isclose(
                expected_post,
                float(ledgers["post_transfer"][agent_id]),
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "pre-transfer, transfer, cost and post-transfer ledgers disagree"
                )
        expected_pre = math.fsum(
            float(weights[agent_id]) * float(ledgers["pre_transfer"][agent_id])
            for agent_id in agent_ids
        )
        expected_post = math.fsum(
            float(weights[agent_id]) * float(ledgers["post_transfer"][agent_id])
            for agent_id in agent_ids
        )
        if not math.isclose(
            expected_pre, float(design["social_pre_transfer"]), abs_tol=1e-12
        ) or not math.isclose(
            expected_post, float(design["social_post_transfer"]), abs_tol=1e-12
        ):
            raise ValueError("social values disagree with the declared welfare ledger")
        comparator_id = design["comparator_design_id"]
        comparator = (
            design if comparator_id is None else design_by_id.get(comparator_id)
        )
        if comparator is None:
            raise ValueError("result design references an unknown comparator")
        comparator_ledgers = cast("dict[str, dict[str, float]]", comparator["ledgers"])
        signed = cast("dict[str, Any]", design["signed_values"])
        by_agent = cast("dict[str, float]", signed["by_agent"])
        if set(by_agent) != agent_ids:
            raise ValueError("signed values must contain exactly the welfare agents")
        for agent_id, value in by_agent.items():
            if not math.isclose(
                float(value),
                float(ledgers["post_transfer"][agent_id])
                - float(comparator_ledgers["post_transfer"][agent_id]),
                abs_tol=1e-12,
            ):
                raise ValueError("signed agent value disagrees with comparator ledger")
        expected_social = math.fsum(
            float(weights[agent_id])
            * (
                float(ledgers[ledger_stage][agent_id])
                - float(comparator_ledgers[ledger_stage][agent_id])
            )
            for agent_id in agent_ids
        )
        if not math.isclose(expected_social, float(signed["social"]), abs_tol=1e-12):
            raise ValueError("signed social value disagrees with comparator ledger")


def validate_signed_social_information_input_or_raise(
    payload: Mapping[str, object],
) -> None:
    """Translate strict contract failures to the public input-error boundary."""
    try:
        validate_signed_social_information_semantics(payload)
    except (ArithmeticError, KeyError, TypeError, ValueError) as error:
        raise_input_error(str(error))
