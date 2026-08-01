"""Strict v1 contract for finite uncertainty-modelling value diagnostics."""

# pyright: reportAny=false, reportExplicitAny=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false

from __future__ import annotations

from collections.abc import Mapping
from itertools import pairwise
import math
from typing import TYPE_CHECKING, Any, Final, cast

from jsonschema import Draft202012Validator

if TYPE_CHECKING:
    from jsonschema.exceptions import ValidationError

_ID: Final[dict[str, object]] = {
    "type": "string",
    "minLength": 1,
    "pattern": r"^[A-Za-z][A-Za-z0-9._-]*$",
}
_STRING: Final[dict[str, object]] = {"type": "string", "minLength": 1}
_NUMBER: Final[dict[str, object]] = {"type": "number"}
_ID_ARRAY: Final[dict[str, object]] = {
    "type": "array",
    "uniqueItems": True,
    "items": _ID,
}

UNCERTAINTY_MODELLING_VALUE_INPUT_SCHEMA_V1: Final[dict[str, object]] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://voiage.dev/schemas/frontier/uncertainty-modelling-value-input.v1.json",
    "title": "UncertaintyModellingValueInputV1Experimental",
    "type": "object",
    "required": [
        "schema_version",
        "analysis_id",
        "analysis_type",
        "method_maturity",
        "objective",
        "point_estimate",
        "stages",
        "states",
        "histories",
        "policies",
        "deterministic_candidates",
        "tie_policy",
        "solver_assurance",
        "provenance",
    ],
    "properties": {
        "schema_version": {"const": "1.0.0"},
        "analysis_id": _ID,
        "analysis_type": {"const": "uncertainty_modelling_value"},
        "method_maturity": {"const": "experimental"},
        "objective": {
            "type": "object",
            "required": [
                "direction",
                "value_unit",
                "population_basis",
                "horizon_basis",
                "discount_basis",
                "risk_criterion",
            ],
            "properties": {
                "direction": {"enum": ["minimize", "maximize"]},
                "value_unit": _STRING,
                "population_basis": _STRING,
                "horizon_basis": _STRING,
                "discount_basis": _STRING,
                "risk_criterion": {"const": "expected_value"},
            },
            "additionalProperties": False,
        },
        "point_estimate": {
            "type": "object",
            "required": [
                "functional",
                "parameter_unit",
                "value",
                "deterministic_model_revision",
            ],
            "properties": {
                "functional": {"enum": ["expectation", "median", "declared_custom"]},
                "parameter_unit": _STRING,
                "value": _NUMBER,
                "deterministic_model_revision": _STRING,
            },
            "additionalProperties": False,
        },
        "stages": {
            "type": "array",
            "minItems": 2,
            "items": {
                "type": "object",
                "required": ["stage_id", "order", "information_available"],
                "properties": {
                    "stage_id": _ID,
                    "order": {"type": "integer", "minimum": 1},
                    "information_available": _ID_ARRAY,
                },
                "additionalProperties": False,
            },
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
        "histories": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["history_id", "stage_id", "reachable_states"],
                "properties": {
                    "history_id": _ID,
                    "stage_id": _ID,
                    "reachable_states": {**_ID_ARRAY, "minItems": 1},
                },
                "additionalProperties": False,
            },
        },
        "policies": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": [
                    "policy_id",
                    "first_stage_decision",
                    "history_decisions",
                    "state_outcomes",
                ],
                "properties": {
                    "policy_id": _ID,
                    "first_stage_decision": _STRING,
                    "history_decisions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["history_id", "decision"],
                            "properties": {
                                "history_id": _ID,
                                "decision": _STRING,
                            },
                            "additionalProperties": False,
                        },
                    },
                    "state_outcomes": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "required": [
                                "state_id",
                                "feasible",
                                "objective_value",
                                "recourse_status",
                            ],
                            "properties": {
                                "state_id": _ID,
                                "feasible": {"type": "boolean"},
                                "objective_value": {
                                    "oneOf": [_NUMBER, {"type": "null"}]
                                },
                                "recourse_status": {"enum": ["feasible", "infeasible"]},
                            },
                            "additionalProperties": False,
                        },
                    },
                },
                "additionalProperties": False,
            },
        },
        "deterministic_candidates": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": [
                    "candidate_id",
                    "first_stage_decision",
                    "point_objective_value",
                    "induced_policy_id",
                ],
                "properties": {
                    "candidate_id": _ID,
                    "first_stage_decision": _STRING,
                    "point_objective_value": _NUMBER,
                    "induced_policy_id": _ID,
                },
                "additionalProperties": False,
            },
        },
        "tie_policy": {
            "type": "object",
            "required": [
                "absolute_tolerance",
                "relative_tolerance",
                "selection",
            ],
            "properties": {
                "absolute_tolerance": {"type": "number", "minimum": 0},
                "relative_tolerance": {"type": "number", "minimum": 0},
                "selection": {"const": "complete_ties_then_lexical"},
            },
            "additionalProperties": False,
        },
        "solver_assurance": {
            "type": "object",
            "required": [
                "solver_type",
                "candidate_space_complete",
                "objective_bound_tolerance",
                "feasibility_tolerance",
                "model_revision",
            ],
            "properties": {
                "solver_type": {"const": "exact_enumeration"},
                "candidate_space_complete": {"const": True},
                "objective_bound_tolerance": {"type": "number", "minimum": 0},
                "feasibility_tolerance": {"type": "number", "minimum": 0},
                "model_revision": _STRING,
            },
            "additionalProperties": False,
        },
        "provenance": {
            "type": "object",
            "required": [
                "scenario_source",
                "policy_generator",
                "independent_reference",
                "software_version",
            ],
            "properties": {
                "scenario_source": _STRING,
                "policy_generator": _STRING,
                "independent_reference": _STRING,
                "software_version": _STRING,
            },
            "additionalProperties": False,
        },
    },
    "additionalProperties": False,
}

_OPTIONAL_NUMBER: Final[dict[str, object]] = {"oneOf": [_NUMBER, {"type": "null"}]}
_HISTORY_DECISION: Final[dict[str, object]] = {
    "type": "object",
    "required": ["history_id", "decision"],
    "properties": {"history_id": _ID, "decision": _STRING},
    "additionalProperties": False,
}

UNCERTAINTY_MODELLING_VALUE_RESULT_SCHEMA_V1: Final[dict[str, object]] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://voiage.dev/schemas/frontier/uncertainty-modelling-value-result.v1.json",
    "title": "UncertaintyModellingValueResultV1Experimental",
    "type": "object",
    "required": [
        "schema_version",
        "analysis_id",
        "analysis_type",
        "method_maturity",
        "objective",
        "point_estimate",
        "scenario_structure",
        "expected_value_problem",
        "expected_result_of_ev_solution",
        "recourse_problem",
        "wait_and_see",
        "decomposition",
        "policy_audit",
        "assurance",
        "language_dispositions",
        "unsupported_dispositions",
        "provenance",
    ],
    "properties": {
        "schema_version": {"const": "1.0.0"},
        "analysis_id": _ID,
        "analysis_type": {"const": "uncertainty_modelling_value_result"},
        "method_maturity": {"const": "experimental"},
        "objective": cast(
            "dict[str, object]",
            cast(
                "dict[str, object]",
                UNCERTAINTY_MODELLING_VALUE_INPUT_SCHEMA_V1["properties"],
            )["objective"],
        ),
        "point_estimate": cast(
            "dict[str, object]",
            cast(
                "dict[str, object]",
                UNCERTAINTY_MODELLING_VALUE_INPUT_SCHEMA_V1["properties"],
            )["point_estimate"],
        ),
        "scenario_structure": {
            "type": "object",
            "required": ["stages", "states", "histories"],
            "properties": {
                "stages": cast(
                    "dict[str, object]",
                    cast(
                        "dict[str, object]",
                        UNCERTAINTY_MODELLING_VALUE_INPUT_SCHEMA_V1["properties"],
                    )["stages"],
                ),
                "states": cast(
                    "dict[str, object]",
                    cast(
                        "dict[str, object]",
                        UNCERTAINTY_MODELLING_VALUE_INPUT_SCHEMA_V1["properties"],
                    )["states"],
                ),
                "histories": cast(
                    "dict[str, object]",
                    cast(
                        "dict[str, object]",
                        UNCERTAINTY_MODELLING_VALUE_INPUT_SCHEMA_V1["properties"],
                    )["histories"],
                ),
            },
            "additionalProperties": False,
        },
        "expected_value_problem": {
            "type": "object",
            "required": [
                "candidate_values",
                "candidate_tie",
                "selected_candidate_id",
                "selected_first_stage_decision",
                "point_objective_value",
                "induced_policy_id",
            ],
            "properties": {
                "candidate_values": {
                    "type": "object",
                    "minProperties": 1,
                    "additionalProperties": _NUMBER,
                },
                "candidate_tie": {**_ID_ARRAY, "minItems": 1},
                "selected_candidate_id": _ID,
                "selected_first_stage_decision": _STRING,
                "point_objective_value": _NUMBER,
                "induced_policy_id": _ID,
            },
            "additionalProperties": False,
        },
        "expected_result_of_ev_solution": {
            "type": "object",
            "required": ["status", "value", "infeasible_states"],
            "properties": {
                "status": {"enum": ["feasible", "infeasible_recourse"]},
                "value": _OPTIONAL_NUMBER,
                "infeasible_states": _ID_ARRAY,
            },
            "additionalProperties": False,
        },
        "recourse_problem": {
            "type": "object",
            "required": ["value", "policy_tie", "selected_policy_id"],
            "properties": {
                "value": _NUMBER,
                "policy_tie": {**_ID_ARRAY, "minItems": 1},
                "selected_policy_id": _ID,
            },
            "additionalProperties": False,
        },
        "wait_and_see": {
            "type": "object",
            "required": ["value", "state_solutions"],
            "properties": {
                "value": _NUMBER,
                "state_solutions": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "required": [
                            "state_id",
                            "probability",
                            "policy_tie",
                            "selected_policy_id",
                            "objective_value",
                        ],
                        "properties": {
                            "state_id": _ID,
                            "probability": {
                                "type": "number",
                                "exclusiveMinimum": 0,
                                "maximum": 1,
                            },
                            "policy_tie": {**_ID_ARRAY, "minItems": 1},
                            "selected_policy_id": _ID,
                            "objective_value": _NUMBER,
                        },
                        "additionalProperties": False,
                    },
                },
            },
            "additionalProperties": False,
        },
        "decomposition": {
            "type": "object",
            "required": [
                "vss",
                "eviu",
                "evpi",
                "eviu_comparator",
                "eviu_equals_vss_under_v1_contract",
                "identity_status",
            ],
            "properties": {
                "vss": _OPTIONAL_NUMBER,
                "eviu": _OPTIONAL_NUMBER,
                "evpi": {"type": "number", "minimum": 0},
                "eviu_comparator": {"const": "declared_point_estimate_ev_solution"},
                "eviu_equals_vss_under_v1_contract": {"const": True},
                "identity_status": {
                    "enum": ["verified", "not_estimable_infeasible_eev"]
                },
            },
            "additionalProperties": False,
        },
        "policy_audit": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": [
                    "policy_id",
                    "first_stage_decision",
                    "history_decisions",
                    "expected_value",
                    "feasible_all_states",
                    "infeasible_states",
                    "state_outcomes",
                ],
                "properties": {
                    "policy_id": _ID,
                    "first_stage_decision": _STRING,
                    "history_decisions": {"type": "array", "items": _HISTORY_DECISION},
                    "expected_value": _OPTIONAL_NUMBER,
                    "feasible_all_states": {"type": "boolean"},
                    "infeasible_states": _ID_ARRAY,
                    "state_outcomes": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "required": [
                                "state_id",
                                "feasible",
                                "objective_value",
                                "recourse_status",
                            ],
                            "properties": {
                                "state_id": _ID,
                                "feasible": {"type": "boolean"},
                                "objective_value": _OPTIONAL_NUMBER,
                                "recourse_status": {"enum": ["feasible", "infeasible"]},
                            },
                            "additionalProperties": False,
                        },
                    },
                },
                "additionalProperties": False,
            },
        },
        "assurance": {
            "type": "object",
            "required": [
                "solver_type",
                "candidate_space_complete",
                "objective_bound_tolerance",
                "feasibility_tolerance",
                "model_revision",
                "states_evaluated",
                "policies_evaluated",
                "deterministic_candidates_evaluated",
                "nonanticipativity_representation",
                "recourse_feasibility_checked",
                "information_acquisition_modelled",
                "information_acquisition_separate_from_uncertainty_modelling",
                "deterministic_serialization",
                "objective_bound",
                "optimality_gap",
                "feasible_policies",
                "infeasible_policies",
            ],
            "properties": {
                "solver_type": {"const": "exact_enumeration"},
                "candidate_space_complete": {"const": True},
                "objective_bound_tolerance": {"type": "number", "minimum": 0},
                "feasibility_tolerance": {"type": "number", "minimum": 0},
                "model_revision": _STRING,
                "states_evaluated": {"type": "integer", "minimum": 1},
                "policies_evaluated": {"type": "integer", "minimum": 1},
                "deterministic_candidates_evaluated": {"type": "integer", "minimum": 1},
                "nonanticipativity_representation": {
                    "const": "one_decision_per_shared_history"
                },
                "recourse_feasibility_checked": {"const": True},
                "information_acquisition_modelled": {"const": False},
                "information_acquisition_separate_from_uncertainty_modelling": {
                    "const": True
                },
                "deterministic_serialization": {"const": True},
                "objective_bound": _NUMBER,
                "optimality_gap": {"type": "number", "minimum": 0},
                "feasible_policies": {"type": "integer", "minimum": 1},
                "infeasible_policies": {"type": "integer", "minimum": 0},
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
                "dvss",
                "vms",
                "approximate_or_external_solvers",
                "risk_criteria_beyond_expected_value",
            ],
            "properties": {
                "dvss": {
                    "const": "deferred_pending_separate_multistage_reference_contract"
                },
                "vms": {
                    "const": "deferred_pending_separate_multistage_reference_contract"
                },
                "approximate_or_external_solvers": {"const": "not_supported_in_v1"},
                "risk_criteria_beyond_expected_value": {"const": "not_supported_in_v1"},
            },
            "additionalProperties": False,
        },
        "provenance": cast(
            "dict[str, object]",
            cast(
                "dict[str, object]",
                UNCERTAINTY_MODELLING_VALUE_INPUT_SCHEMA_V1["properties"],
            )["provenance"],
        ),
    },
    "additionalProperties": False,
}


def _validate(schema: Mapping[str, object], instance: Mapping[str, object]) -> None:
    validator = Draft202012Validator(schema)
    errors: list[ValidationError] = sorted(
        validator.iter_errors(cast("Any", instance)),
        key=lambda error: list(error.absolute_path),
    )
    if errors:
        first = errors[0]
        location = "/".join(str(part) for part in first.absolute_path) or "<root>"
        raise ValueError(f"schema validation failed at {location}: {first.message}")


def validate_uncertainty_modelling_value_semantics(
    payload: Mapping[str, object],
) -> None:
    """Validate cross-record identities and finite exact-policy semantics."""
    _validate(UNCERTAINTY_MODELLING_VALUE_INPUT_SCHEMA_V1, payload)
    data = cast("dict[str, Any]", payload)
    states = cast("list[dict[str, Any]]", data["states"])
    state_ids = [str(state["state_id"]) for state in states]
    if len(state_ids) != len(set(state_ids)):
        raise ValueError("state identifiers must be unique")
    probabilities = [float(state["probability"]) for state in states]
    if not all(math.isfinite(value) for value in probabilities) or not math.isclose(
        math.fsum(probabilities), 1.0, abs_tol=1e-12, rel_tol=1e-12
    ):
        raise ValueError("state probabilities must be finite and sum to one")

    stages = cast("list[dict[str, Any]]", data["stages"])
    stage_ids = [str(stage["stage_id"]) for stage in stages]
    orders = [int(stage["order"]) for stage in stages]
    if len(stage_ids) != len(set(stage_ids)) or sorted(orders) != list(
        range(1, len(stages) + 1)
    ):
        raise ValueError("stages require unique identifiers and contiguous order")
    first_stage = stage_ids[orders.index(1)]
    ordered_stages = sorted(stages, key=lambda stage: int(stage["order"]))
    previous_information: set[str] = set()
    for stage in ordered_stages:
        available = set(cast("list[str]", stage["information_available"]))
        if not previous_information.issubset(available):
            raise ValueError("information available must be cumulative across stages")
        previous_information = available
    histories = cast("list[dict[str, Any]]", data["histories"])
    history_ids = [str(history["history_id"]) for history in histories]
    if len(history_ids) != len(set(history_ids)):
        raise ValueError("history identifiers must be unique")
    state_set = set(state_ids)
    for history in histories:
        if history["stage_id"] not in stage_ids or history["stage_id"] == first_stage:
            raise ValueError("histories must reference a recourse stage")
        if not set(history["reachable_states"]).issubset(state_set):
            raise ValueError("history references an unknown state")
    for stage_id in stage_ids:
        if stage_id == first_stage:
            continue
        reachable = [
            state
            for history in histories
            if history["stage_id"] == stage_id
            for state in history["reachable_states"]
        ]
        if sorted(reachable) != sorted(state_ids):
            raise ValueError(
                "histories must partition states once at each recourse stage"
            )
    recourse_stage_ids = [
        str(stage["stage_id"])
        for stage in ordered_stages
        if stage["stage_id"] != first_stage
    ]
    for prior_stage_id, later_stage_id in pairwise(recourse_stage_ids):
        prior_parts = [
            set(cast("list[str]", history["reachable_states"]))
            for history in histories
            if history["stage_id"] == prior_stage_id
        ]
        for later in (
            set(cast("list[str]", history["reachable_states"]))
            for history in histories
            if history["stage_id"] == later_stage_id
        ):
            if sum(later.issubset(prior) for prior in prior_parts) != 1:
                raise ValueError(
                    "later-stage histories must refine the prior-stage partition"
                )

    policies = cast("list[dict[str, Any]]", data["policies"])
    policy_ids = [str(policy["policy_id"]) for policy in policies]
    if len(policy_ids) != len(set(policy_ids)):
        raise ValueError("policy identifiers must be unique")
    for policy in policies:
        decisions = cast("list[dict[str, Any]]", policy["history_decisions"])
        decision_histories = [str(decision["history_id"]) for decision in decisions]
        if sorted(decision_histories) != sorted(history_ids):
            raise ValueError(
                "every policy must declare one decision per shared history"
            )
        outcomes = cast("list[dict[str, Any]]", policy["state_outcomes"])
        outcome_ids = [str(outcome["state_id"]) for outcome in outcomes]
        if sorted(outcome_ids) != sorted(state_ids):
            raise ValueError("every policy must declare one outcome per state")
        for outcome in outcomes:
            value = outcome["objective_value"]
            feasible = bool(outcome["feasible"])
            if feasible != (outcome["recourse_status"] == "feasible"):
                raise ValueError("feasibility and recourse status disagree")
            if feasible and (
                not isinstance(value, (int, float)) or not math.isfinite(float(value))
            ):
                raise ValueError("feasible outcomes require finite objective values")
            if not feasible and value is not None:
                raise ValueError("infeasible outcomes require a null objective value")

    candidates = cast("list[dict[str, Any]]", data["deterministic_candidates"])
    candidate_ids = [str(candidate["candidate_id"]) for candidate in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("deterministic candidate identifiers must be unique")
    policy_set = set(policy_ids)
    for candidate in candidates:
        if candidate["induced_policy_id"] not in policy_set:
            raise ValueError("deterministic candidate references an unknown policy")
        induced = next(
            policy
            for policy in policies
            if policy["policy_id"] == candidate["induced_policy_id"]
        )
        if candidate["first_stage_decision"] != induced["first_stage_decision"]:
            raise ValueError(
                "deterministic candidate and induced policy first-stage decisions disagree"
            )
        if not math.isfinite(float(candidate["point_objective_value"])):
            raise ValueError("deterministic objectives must be finite")
    numeric_paths = [
        data["point_estimate"]["value"],
        data["tie_policy"]["absolute_tolerance"],
        data["tie_policy"]["relative_tolerance"],
        data["solver_assurance"]["objective_bound_tolerance"],
        data["solver_assurance"]["feasibility_tolerance"],
    ]
    if not all(math.isfinite(float(value)) for value in numeric_paths):
        raise ValueError("numeric contract values must be finite")


def validate_uncertainty_modelling_value_result(
    payload: Mapping[str, object],
) -> None:
    """Validate the portable result envelope."""
    _validate(UNCERTAINTY_MODELLING_VALUE_RESULT_SCHEMA_V1, payload)
    _ensure_finite_json(payload, "<root>")


def _ensure_finite_json(value: object, path: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"result contains a non-finite number at {path}")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _ensure_finite_json(item, f"{path}/{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _ensure_finite_json(item, f"{path}/{index}")
