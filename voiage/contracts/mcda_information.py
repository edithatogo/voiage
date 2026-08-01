"""Portable v1 contracts for finite additive MCDA perfect information."""

# pyright: reportAny=false, reportExplicitAny=false, reportUnknownMemberType=false, reportUnusedCallResult=false, reportUnknownArgumentType=false, reportUnknownVariableType=false

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any, Final, cast

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

MAX_SUM_TOLERANCE: Final = 1e-6

_ID: Final[dict[str, object]] = {
    "type": "string",
    "minLength": 1,
    "pattern": r"^[A-Za-z][A-Za-z0-9._-]*$",
}
_NUMBER: Final[dict[str, object]] = {"type": "number"}
_STRING: Final[dict[str, object]] = {"type": "string", "minLength": 1}
_ID_ARRAY: Final[dict[str, object]] = {
    "type": "array",
    "uniqueItems": True,
    "items": _ID,
}

_COST: Final[dict[str, object]] = {
    "type": "object",
    "required": [
        "original_amount",
        "original_unit",
        "aggregate_amount",
        "conversion_reference",
        "population_basis",
        "horizon_basis",
        "discount_basis",
        "cost_scope",
    ],
    "properties": {
        "original_amount": {"type": "number", "minimum": 0},
        "original_unit": _STRING,
        "aggregate_amount": {"type": "number", "minimum": 0},
        "conversion_reference": _STRING,
        "population_basis": _STRING,
        "horizon_basis": _STRING,
        "discount_basis": _STRING,
        "cost_scope": {"const": "action_specific_disjoint"},
    },
    "additionalProperties": False,
}

MCDA_INFORMATION_INPUT_SCHEMA_V1: Final[dict[str, object]] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://voiage.dev/schemas/frontier/mcda-information-input.v1.json",
    "title": "McdaInformationInputV1Experimental",
    "type": "object",
    "required": [
        "schema_version",
        "analysis_id",
        "analysis_type",
        "method_maturity",
        "aggregation_family",
        "aggregate_direction",
        "aggregate_unit",
        "alternatives",
        "criteria",
        "default_weights",
        "latent_partitions",
        "joint_states",
        "information_actions",
        "tolerances",
        "provenance",
    ],
    "properties": {
        "schema_version": {"const": "1.0.0"},
        "analysis_id": _ID,
        "analysis_type": {"const": "mcda_perfect_information"},
        "method_maturity": {"const": "experimental"},
        "aggregation_family": {"const": "compensatory_additive_value"},
        "aggregate_direction": {"const": "maximize"},
        "aggregate_unit": _STRING,
        "alternatives": {
            "type": "array",
            "minItems": 2,
            "items": {
                "type": "object",
                "required": ["alternative_id", "label", "definition_source"],
                "properties": {
                    "alternative_id": _ID,
                    "label": _STRING,
                    "definition_source": _STRING,
                },
                "additionalProperties": False,
            },
        },
        "criteria": {
            "type": "array",
            "minItems": 2,
            "items": {
                "type": "object",
                "required": [
                    "criterion_id",
                    "label",
                    "raw_unit",
                    "direction",
                    "operational_definition",
                    "value_function",
                    "source_reference",
                ],
                "properties": {
                    "criterion_id": _ID,
                    "label": _STRING,
                    "raw_unit": _STRING,
                    "direction": {"enum": ["higher_is_better", "lower_is_better"]},
                    "operational_definition": _STRING,
                    "source_reference": _STRING,
                    "value_function": {
                        "type": "object",
                        "required": [
                            "family",
                            "normalization_scope",
                            "anchors",
                            "valid_domain",
                            "extrapolation_policy",
                            "elicitation_source",
                        ],
                        "properties": {
                            "family": {"const": "linear_fixed_anchors"},
                            "normalization_scope": {"const": "fixed_ex_ante"},
                            "anchors": {
                                "type": "array",
                                "minItems": 2,
                                "maxItems": 2,
                                "items": {
                                    "type": "object",
                                    "required": ["raw", "value"],
                                    "properties": {
                                        "raw": _NUMBER,
                                        "value": _NUMBER,
                                    },
                                    "additionalProperties": False,
                                },
                            },
                            "valid_domain": {
                                "type": "array",
                                "minItems": 2,
                                "maxItems": 2,
                                "items": _NUMBER,
                            },
                            "extrapolation_policy": {"enum": ["reject", "linear"]},
                            "elicitation_source": _STRING,
                        },
                        "additionalProperties": False,
                    },
                },
                "additionalProperties": False,
            },
        },
        "default_weights": {
            "type": "object",
            "minProperties": 2,
            "additionalProperties": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "latent_partitions": {
            "type": "object",
            "required": ["outcome_keys", "preference_keys", "dependence_assumption"],
            "properties": {
                "outcome_keys": {**_ID_ARRAY, "minItems": 1},
                "preference_keys": {**_ID_ARRAY, "minItems": 1},
                "dependence_assumption": _STRING,
            },
            "additionalProperties": False,
        },
        "joint_states": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": [
                    "state_id",
                    "probability",
                    "partition_values",
                    "performances",
                ],
                "properties": {
                    "state_id": _ID,
                    "probability": {
                        "type": "number",
                        "exclusiveMinimum": 0,
                        "maximum": 1,
                    },
                    "partition_values": {
                        "type": "object",
                        "minProperties": 1,
                        "additionalProperties": _STRING,
                    },
                    "performances": {
                        "type": "object",
                        "minProperties": 2,
                        "additionalProperties": {
                            "type": "object",
                            "minProperties": 2,
                            "additionalProperties": _NUMBER,
                        },
                    },
                    "weights": {
                        "type": "object",
                        "minProperties": 2,
                        "additionalProperties": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                        },
                    },
                },
                "additionalProperties": False,
            },
        },
        "information_actions": {
            "type": "array",
            "minItems": 3,
            "maxItems": 3,
            "items": {
                "type": "object",
                "required": [
                    "action_id",
                    "action_type",
                    "outcome_partition_keys",
                    "preference_partition_keys",
                    "cost",
                ],
                "properties": {
                    "action_id": _ID,
                    "action_type": {"enum": ["criterion", "preference", "joint"]},
                    "outcome_partition_keys": _ID_ARRAY,
                    "preference_partition_keys": _ID_ARRAY,
                    "cost": _COST,
                },
                "additionalProperties": False,
            },
        },
        "tolerances": {
            "type": "object",
            "required": [
                "absolute_tie",
                "relative_tie",
                "probability_sum",
                "weight_sum",
                "pareto_absolute",
            ],
            "properties": {
                "absolute_tie": {"type": "number", "minimum": 0},
                "relative_tie": {"type": "number", "minimum": 0},
                "probability_sum": {
                    "type": "number",
                    "exclusiveMinimum": 0,
                    "maximum": MAX_SUM_TOLERANCE,
                },
                "weight_sum": {
                    "type": "number",
                    "exclusiveMinimum": 0,
                    "maximum": MAX_SUM_TOLERANCE,
                },
                "pareto_absolute": {"type": "number", "minimum": 0},
            },
            "additionalProperties": False,
        },
        "provenance": {
            "type": "object",
            "required": [
                "decision_revision",
                "model_revision",
                "data_sources",
                "transformation_sources",
                "weight_elicitation_source",
                "joint_probability_source",
                "normalization_anchor_source",
                "partition_source",
                "cost_source",
                "tie_policy_source",
                "evaluator",
                "software_version",
            ],
            "properties": {
                "decision_revision": _STRING,
                "model_revision": _STRING,
                "data_sources": {"type": "array", "minItems": 1, "items": _STRING},
                "transformation_sources": {
                    "type": "array",
                    "minItems": 1,
                    "items": _STRING,
                },
                "weight_elicitation_source": _STRING,
                "joint_probability_source": _STRING,
                "normalization_anchor_source": _STRING,
                "partition_source": _STRING,
                "cost_source": _STRING,
                "tie_policy_source": _STRING,
                "evaluator": _STRING,
                "software_version": _STRING,
            },
            "additionalProperties": False,
        },
    },
    "additionalProperties": False,
}

_TIE_GROUP: Final[dict[str, object]] = {
    "type": "object",
    "required": ["rank", "alternative_ids", "score"],
    "properties": {
        "rank": {"type": "integer", "minimum": 1},
        "alternative_ids": {**_ID_ARRAY, "minItems": 1},
        "score": _NUMBER,
    },
    "additionalProperties": False,
}
_SCORES: Final[dict[str, object]] = {
    "type": "object",
    "minProperties": 2,
    "additionalProperties": _NUMBER,
}

MCDA_INFORMATION_RESULT_SCHEMA_V1: Final[dict[str, object]] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://voiage.dev/schemas/frontier/mcda-information-result.v1.json",
    "title": "McdaInformationResultV1Experimental",
    "type": "object",
    "required": [
        "schema_version",
        "analysis_id",
        "analysis_type",
        "method_maturity",
        "aggregate_unit",
        "alternative_ids",
        "criterion_ids",
        "baseline",
        "conditional_actions",
        "decomposition",
        "regret",
        "rank_acceptability",
        "pareto",
        "assurance",
        "language_dispositions",
        "unsupported_dispositions",
    ],
    "properties": {
        "schema_version": {"const": "1.0.0"},
        "analysis_id": _ID,
        "analysis_type": {"const": "mcda_perfect_information_result"},
        "method_maturity": {"const": "experimental"},
        "aggregate_unit": _STRING,
        "alternative_ids": {**_ID_ARRAY, "minItems": 2},
        "criterion_ids": {**_ID_ARRAY, "minItems": 2},
        "baseline": {
            "type": "object",
            "required": ["expected_scores", "ranking", "choice_tie", "value"],
            "properties": {
                "expected_scores": _SCORES,
                "ranking": {"type": "array", "minItems": 1, "items": _TIE_GROUP},
                "choice_tie": {**_ID_ARRAY, "minItems": 1},
                "value": _NUMBER,
            },
            "additionalProperties": False,
        },
        "conditional_actions": {
            "type": "array",
            "minItems": 3,
            "maxItems": 3,
            "items": {
                "type": "object",
                "required": [
                    "action_id",
                    "action_type",
                    "resolved_partition_keys",
                    "partitions",
                    "resolved_value",
                    "gross_voi",
                    "cost",
                    "net_voi",
                    "expected_regret",
                    "statewise_regret",
                ],
                "properties": {
                    "action_id": _ID,
                    "action_type": {"enum": ["criterion", "preference", "joint"]},
                    "resolved_partition_keys": {**_ID_ARRAY, "minItems": 1},
                    "partitions": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "required": [
                                "partition_id",
                                "key_values",
                                "probability",
                                "conditional_scores",
                                "ranking",
                                "choice_tie",
                                "conditional_value",
                            ],
                            "properties": {
                                "partition_id": _ID,
                                "key_values": {
                                    "type": "object",
                                    "minProperties": 1,
                                    "additionalProperties": _STRING,
                                },
                                "probability": {
                                    "type": "number",
                                    "minimum": 0,
                                    "maximum": 1,
                                },
                                "conditional_scores": _SCORES,
                                "ranking": {
                                    "type": "array",
                                    "minItems": 1,
                                    "items": _TIE_GROUP,
                                },
                                "choice_tie": {**_ID_ARRAY, "minItems": 1},
                                "conditional_value": _NUMBER,
                            },
                            "additionalProperties": False,
                        },
                    },
                    "resolved_value": _NUMBER,
                    "gross_voi": {"type": "number", "minimum": 0},
                    "cost": _COST,
                    "net_voi": _NUMBER,
                    "expected_regret": {"type": "number", "minimum": 0},
                    "statewise_regret": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "required": ["state_id", "policy_tie", "regret"],
                            "properties": {
                                "state_id": _ID,
                                "policy_tie": {**_ID_ARRAY, "minItems": 1},
                                "regret": {"type": "number", "minimum": 0},
                            },
                            "additionalProperties": False,
                        },
                    },
                },
                "additionalProperties": False,
            },
        },
        "decomposition": {
            "type": "object",
            "required": [
                "criterion_action_id",
                "preference_action_id",
                "joint_action_id",
                "criterion_gross_voi",
                "preference_gross_voi",
                "joint_gross_voi",
                "interaction",
                "joint_increment_over_criterion",
                "joint_increment_over_preference",
                "no_double_counting_identity_residual",
            ],
            "properties": {
                "criterion_action_id": _ID,
                "preference_action_id": _ID,
                "joint_action_id": _ID,
                "criterion_gross_voi": {"type": "number", "minimum": 0},
                "preference_gross_voi": {"type": "number", "minimum": 0},
                "joint_gross_voi": {"type": "number", "minimum": 0},
                "interaction": _NUMBER,
                "joint_increment_over_criterion": {"type": "number", "minimum": 0},
                "joint_increment_over_preference": {"type": "number", "minimum": 0},
                "no_double_counting_identity_residual": _NUMBER,
            },
            "additionalProperties": False,
        },
        "regret": {
            "type": "object",
            "required": ["definition", "baseline_expected", "statewise"],
            "properties": {
                "definition": {"const": "state_optimum_minus_policy_score"},
                "baseline_expected": {"type": "number", "minimum": 0},
                "statewise": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "required": [
                            "state_id",
                            "probability",
                            "optimal_tie",
                            "baseline_policy_regret",
                        ],
                        "properties": {
                            "state_id": _ID,
                            "probability": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1,
                            },
                            "optimal_tie": {**_ID_ARRAY, "minItems": 1},
                            "baseline_policy_regret": {
                                "type": "number",
                                "minimum": 0,
                            },
                        },
                        "additionalProperties": False,
                    },
                },
            },
            "additionalProperties": False,
        },
        "rank_acceptability": {
            "type": "object",
            "required": ["tie_convention", "by_alternative", "state_tie_groups"],
            "properties": {
                "tie_convention": {"const": "fractional_complete_tie_groups"},
                "by_alternative": {
                    "type": "object",
                    "minProperties": 2,
                    "additionalProperties": {
                        "type": "array",
                        "minItems": 2,
                        "items": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                },
                "state_tie_groups": {
                    "type": "object",
                    "minProperties": 1,
                    "additionalProperties": {
                        "type": "array",
                        "minItems": 1,
                        "items": _TIE_GROUP,
                    },
                },
            },
            "additionalProperties": False,
        },
        "pareto": {
            "type": "object",
            "required": [
                "basis",
                "expectation_law",
                "tie_tolerance",
                "expected_value_vectors",
                "expected_dominance",
                "expected_non_dominated",
                "statewise",
            ],
            "properties": {
                "basis": {"const": "fixed_direction_normalized_criterion_values"},
                "expectation_law": {"const": "submitted_joint_state_probabilities"},
                "tie_tolerance": {"type": "number", "minimum": 0},
                "expected_value_vectors": {
                    "type": "object",
                    "minProperties": 2,
                    "additionalProperties": {
                        "type": "object",
                        "minProperties": 2,
                        "additionalProperties": _NUMBER,
                    },
                },
                "expected_dominance": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["dominant", "dominated"],
                        "properties": {"dominant": _ID, "dominated": _ID},
                        "additionalProperties": False,
                    },
                },
                "expected_non_dominated": {**_ID_ARRAY, "minItems": 1},
                "statewise": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "required": [
                            "state_id",
                            "value_vectors",
                            "dominance",
                            "non_dominated",
                        ],
                        "properties": {
                            "state_id": _ID,
                            "value_vectors": {
                                "type": "object",
                                "minProperties": 2,
                                "additionalProperties": {
                                    "type": "object",
                                    "minProperties": 2,
                                    "additionalProperties": _NUMBER,
                                },
                            },
                            "dominance": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "required": ["dominant", "dominated"],
                                    "properties": {
                                        "dominant": _ID,
                                        "dominated": _ID,
                                    },
                                    "additionalProperties": False,
                                },
                            },
                            "non_dominated": {**_ID_ARRAY, "minItems": 1},
                        },
                        "additionalProperties": False,
                    },
                },
            },
            "additionalProperties": False,
        },
        "assurance": {
            "type": "object",
            "required": [
                "estimator",
                "arithmetic",
                "joint_dependence_preserved",
                "normalization_frozen_ex_ante",
                "gross_voi_clipped",
                "probabilities_reconciled",
                "weights_reconciled",
                "fixture_status",
            ],
            "properties": {
                "estimator": {"const": "exact_finite_enumeration"},
                "arithmetic": {"const": "binary64_with_declared_tolerances"},
                "joint_dependence_preserved": {"const": True},
                "normalization_frozen_ex_ante": {"const": True},
                "gross_voi_clipped": {"const": False},
                "probabilities_reconciled": {"const": True},
                "weights_reconciled": {"const": True},
                "fixture_status": {"const": "analytically_reviewed_contract_fixture"},
            },
            "additionalProperties": False,
        },
        "language_dispositions": {
            "type": "object",
            "required": ["python", "rust", "r", "julia", "mojo"],
            "properties": {
                "python": {"const": "executable"},
                "rust": {"const": "unsupported"},
                "r": {"const": "unsupported"},
                "julia": {"const": "unsupported"},
                "mojo": {"const": "external"},
            },
            "additionalProperties": False,
        },
        "unsupported_dispositions": {
            "type": "array",
            "minItems": 1,
            "uniqueItems": True,
            "items": _STRING,
        },
    },
    "additionalProperties": False,
}


def _ids(records: Sequence[Mapping[str, Any]], key: str, label: str) -> list[str]:
    values = [cast("str", record[key]) for record in records]
    if len(values) != len(set(values)):
        raise ValueError(f"{label} must be unique")
    return values


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be finite")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _normalized_weights(
    weights: Mapping[str, Any], criterion_ids: set[str], tolerance: float
) -> None:
    if set(weights) != criterion_ids:
        raise ValueError("weight keys must exactly match criterion IDs")
    values = [_finite(value, "weights") for value in weights.values()]
    if any(value < 0 for value in values):
        raise ValueError("weights must be non-negative")
    if not math.isclose(sum(values), 1.0, abs_tol=tolerance, rel_tol=0.0):
        raise ValueError("weights must sum to 1")


def validate_mcda_information_semantics(specification: Mapping[str, object]) -> None:
    """Validate finite-law alignment, fixed scales and information partitions."""
    try:
        Draft202012Validator(MCDA_INFORMATION_INPUT_SCHEMA_V1).validate(specification)
    except ValidationError as error:
        path = "/" + "/".join(str(part) for part in error.absolute_path)
        raise ValueError(
            f"invalid MCDA information input at {path} (constraint: {error.validator})"
        ) from error

    payload = cast("Mapping[str, Any]", specification)
    alternatives = cast("list[Mapping[str, Any]]", payload["alternatives"])
    criteria = cast("list[Mapping[str, Any]]", payload["criteria"])
    states = cast("list[Mapping[str, Any]]", payload["joint_states"])
    actions = cast("list[Mapping[str, Any]]", payload["information_actions"])
    alternative_ids = set(_ids(alternatives, "alternative_id", "alternative IDs"))
    criterion_ids = set(_ids(criteria, "criterion_id", "criterion IDs"))
    _ids(states, "state_id", "joint-state IDs")
    _ids(actions, "action_id", "information-action IDs")
    action_types = [cast("str", action["action_type"]) for action in actions]
    if sorted(action_types) != ["criterion", "joint", "preference"]:
        raise ValueError(
            "v1 requires exactly one criterion, preference and joint action"
        )

    tolerances = cast("Mapping[str, Any]", payload["tolerances"])
    probability_tolerance = _finite(
        tolerances["probability_sum"], "probability tolerance"
    )
    weight_tolerance = _finite(tolerances["weight_sum"], "weight tolerance")
    if probability_tolerance <= 0 or probability_tolerance > MAX_SUM_TOLERANCE:
        raise ValueError("probability tolerance must be in (0, 1e-6]")
    if weight_tolerance <= 0 or weight_tolerance > MAX_SUM_TOLERANCE:
        raise ValueError("weight tolerance must be in (0, 1e-6]")

    _normalized_weights(
        cast("Mapping[str, Any]", payload["default_weights"]),
        criterion_ids,
        weight_tolerance,
    )
    criterion_domains: dict[str, tuple[float, float, str]] = {}
    for criterion in criteria:
        value_function = cast("Mapping[str, Any]", criterion["value_function"])
        anchors = cast("list[Mapping[str, Any]]", value_function["anchors"])
        raw = [_finite(anchor["raw"], "value-function anchor") for anchor in anchors]
        values = [
            _finite(anchor["value"], "value-function anchor") for anchor in anchors
        ]
        domain = [
            _finite(item, "value-function valid domain")
            for item in cast("list[object]", value_function["valid_domain"])
        ]
        if not raw[0] < raw[1] or not domain[0] < domain[1]:
            raise ValueError("value-function raw anchors and domain must increase")
        if raw[0] < domain[0] or raw[1] > domain[1]:
            raise ValueError("value-function anchors must lie inside the valid domain")
        expected_increase = criterion["direction"] == "higher_is_better"
        if (expected_increase and values[1] <= values[0]) or (
            not expected_increase and values[1] >= values[0]
        ):
            raise ValueError("value-function anchors must follow criterion direction")
        criterion_domains[cast("str", criterion["criterion_id"])] = (
            domain[0],
            domain[1],
            cast("str", value_function["extrapolation_policy"]),
        )

    latent = cast("Mapping[str, Any]", payload["latent_partitions"])
    outcome_keys = set(cast("list[str]", latent["outcome_keys"]))
    preference_keys = set(cast("list[str]", latent["preference_keys"]))
    all_keys = outcome_keys | preference_keys
    if outcome_keys & preference_keys:
        raise ValueError("outcome and preference partition keys must be disjoint")

    probabilities: list[float] = []
    has_state_weights = False
    for state in states:
        probabilities.append(_finite(state["probability"], "state probability"))
        partition_values = cast("Mapping[str, Any]", state["partition_values"])
        if set(partition_values) != all_keys:
            raise ValueError("every joint state must define every partition key")
        performances = cast("Mapping[str, Any]", state["performances"])
        if set(performances) != alternative_ids:
            raise ValueError("state performances must exactly match alternatives")
        for row in performances.values():
            if not isinstance(row, Mapping) or set(row) != criterion_ids:
                raise ValueError("performance rows must exactly match criteria")
            for criterion_id, value in row.items():
                raw_performance = _finite(value, "raw performance")
                lower, upper, extrapolation = criterion_domains[criterion_id]
                if extrapolation == "reject" and not lower <= raw_performance <= upper:
                    raise ValueError(
                        "raw performance lies outside a reject-extrapolation domain"
                    )
        if "weights" in state:
            has_state_weights = True
            _normalized_weights(
                cast("Mapping[str, Any]", state["weights"]),
                criterion_ids,
                weight_tolerance,
            )
    if any(probability < 0 for probability in probabilities):
        raise ValueError("state probabilities must be non-negative")
    if not math.isclose(
        sum(probabilities), 1.0, abs_tol=probability_tolerance, rel_tol=0.0
    ):
        raise ValueError("joint-state probabilities must sum to 1")

    for action in actions:
        outcome = set(cast("list[str]", action["outcome_partition_keys"]))
        preference = set(cast("list[str]", action["preference_partition_keys"]))
        if not outcome <= outcome_keys or not preference <= preference_keys:
            raise ValueError("information action references an unknown partition key")
        action_type = action["action_type"]
        valid_shape = (
            (action_type == "criterion" and bool(outcome) and not preference)
            or (action_type == "preference" and bool(preference) and not outcome)
            or (action_type == "joint" and bool(outcome) and bool(preference))
        )
        if not valid_shape:
            raise ValueError("information action keys must match its declared type")
        if preference and not has_state_weights:
            raise ValueError("preference information requires state-specific weights")
        cost = cast("Mapping[str, Any]", action["cost"])
        for field in ("original_amount", "aggregate_amount"):
            if _finite(cost[field], f"action cost {field}") < 0:
                raise ValueError("information costs must be non-negative")

    by_type = {cast("str", action["action_type"]): action for action in actions}
    criterion_action = by_type["criterion"]
    preference_action = by_type["preference"]
    joint_action = by_type["joint"]
    if set(cast("list[str]", joint_action["outcome_partition_keys"])) != set(
        cast("list[str]", criterion_action["outcome_partition_keys"])
    ) or set(cast("list[str]", joint_action["preference_partition_keys"])) != set(
        cast("list[str]", preference_action["preference_partition_keys"])
    ):
        raise ValueError(
            "joint action must exactly refine the declared criterion and preference actions"
        )


def validate_mcda_information_result_semantics(result: Mapping[str, object]) -> None:
    """Validate result identities and complete finite probability diagnostics."""
    try:
        Draft202012Validator(MCDA_INFORMATION_RESULT_SCHEMA_V1).validate(result)
    except ValidationError as error:
        path = "/" + "/".join(str(part) for part in error.absolute_path)
        raise ValueError(
            f"invalid MCDA information result at {path} (constraint: {error.validator})"
        ) from error

    payload = cast("Mapping[str, Any]", result)
    alternatives = set(cast("list[str]", payload["alternative_ids"]))
    actions = cast("list[Mapping[str, Any]]", payload["conditional_actions"])
    action_ids = _ids(actions, "action_id", "result action IDs")
    if sorted(cast("str", action["action_type"]) for action in actions) != [
        "criterion",
        "joint",
        "preference",
    ]:
        raise ValueError(
            "result requires exactly one criterion, preference and joint action"
        )
    baseline = cast("Mapping[str, Any]", payload["baseline"])
    baseline_value = _finite(baseline["value"], "baseline value")
    baseline_scores = cast("Mapping[str, Any]", baseline["expected_scores"])
    if set(baseline_scores) != alternatives:
        raise ValueError("baseline scores must exactly match alternatives")
    for score in baseline_scores.values():
        _finite(score, "baseline score")
    tolerance = 1e-9
    by_action: dict[str, Mapping[str, Any]] = {}
    for action in actions:
        by_action[cast("str", action["action_id"])] = action
        probabilities = [
            _finite(partition["probability"], "conditional probability")
            for partition in cast("list[Mapping[str, Any]]", action["partitions"])
        ]
        for partition in cast("list[Mapping[str, Any]]", action["partitions"]):
            scores = cast("Mapping[str, Any]", partition["conditional_scores"])
            if set(scores) != alternatives:
                raise ValueError("conditional scores must exactly match alternatives")
            for score in scores.values():
                _finite(score, "conditional score")
            _finite(partition["conditional_value"], "conditional value")
        if not math.isclose(sum(probabilities), 1.0, abs_tol=tolerance, rel_tol=0.0):
            raise ValueError("conditional partition probabilities must sum to 1")
        gross = _finite(action["gross_voi"], "gross VOI")
        resolved = _finite(action["resolved_value"], "resolved value")
        cost = cast("Mapping[str, Any]", action["cost"])
        aggregate_cost = _finite(cost["aggregate_amount"], "aggregate cost")
        if not math.isclose(gross, resolved - baseline_value, abs_tol=tolerance):
            raise ValueError("gross VOI must equal resolved value minus baseline")
        if not math.isclose(
            _finite(action["net_voi"], "net VOI"),
            gross - aggregate_cost,
            abs_tol=tolerance,
        ):
            raise ValueError("net VOI must subtract the action-specific aggregate cost")
    if set(action_ids) != set(by_action):
        raise ValueError("result action IDs must be unique")

    decomposition = cast("Mapping[str, Any]", payload["decomposition"])
    criterion = by_action.get(cast("str", decomposition["criterion_action_id"]))
    preference = by_action.get(cast("str", decomposition["preference_action_id"]))
    joint = by_action.get(cast("str", decomposition["joint_action_id"]))
    if criterion is None or preference is None or joint is None:
        raise ValueError("decomposition action IDs must identify result actions")
    if (
        criterion["action_type"] != "criterion"
        or preference["action_type"] != "preference"
        or joint["action_type"] != "joint"
    ):
        raise ValueError("decomposition action IDs must match their action types")
    c_value = _finite(criterion["gross_voi"], "criterion gross VOI")
    p_value = _finite(preference["gross_voi"], "preference gross VOI")
    j_value = _finite(joint["gross_voi"], "joint gross VOI")
    expected = {
        "criterion_gross_voi": c_value,
        "preference_gross_voi": p_value,
        "joint_gross_voi": j_value,
        "interaction": j_value - c_value - p_value,
        "joint_increment_over_criterion": j_value - c_value,
        "joint_increment_over_preference": j_value - p_value,
        "no_double_counting_identity_residual": 0.0,
    }
    if any(
        not math.isclose(
            _finite(decomposition[field], f"decomposition {field}"),
            value,
            abs_tol=tolerance,
        )
        for field, value in expected.items()
    ):
        raise ValueError("decomposition and interaction identities must reconcile")

    acceptability = cast(
        "Mapping[str, list[Any]]",
        cast("Mapping[str, Any]", payload["rank_acceptability"])["by_alternative"],
    )
    if set(acceptability) != alternatives:
        raise ValueError("rank acceptability must exactly match alternatives")
    for probabilities in acceptability.values():
        values = [_finite(value, "rank acceptability") for value in probabilities]
        if len(values) != len(alternatives) or not math.isclose(
            sum(values), 1.0, abs_tol=tolerance, rel_tol=0.0
        ):
            raise ValueError("rank acceptability must reconcile across every rank")


__all__ = [
    "MCDA_INFORMATION_INPUT_SCHEMA_V1",
    "MCDA_INFORMATION_RESULT_SCHEMA_V1",
    "validate_mcda_information_result_semantics",
    "validate_mcda_information_semantics",
]
