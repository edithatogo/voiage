"""Strict contract for experimental static/dynamic heterogeneity value."""

# pyright: reportAny=false, reportExplicitAny=false, reportMissingModuleSource=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false

from __future__ import annotations

import hashlib
import importlib
import json
import math
from typing import Any, Final, cast

from jsonschema import Draft202012Validator

_ID: Final[dict[str, object]] = {
    "type": "string",
    "minLength": 1,
    "pattern": r"^[A-Za-z][A-Za-z0-9._-]*$",
}
_STRING: Final[dict[str, object]] = {"type": "string", "minLength": 1}
_PROBABILITY: Final[dict[str, object]] = {
    "type": "number",
    "minimum": 0,
    "maximum": 1,
}

HETEROGENEITY_VALUE_INPUT_SCHEMA_V1: Final[dict[str, object]] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://voiage.dev/schemas/frontier/heterogeneity-value-input.v1.json",
    "title": "HeterogeneityValueInputV1Experimental",
    "type": "object",
    "required": [
        "schema_version",
        "analysis_id",
        "analysis_type",
        "method_maturity",
        "objective",
        "subgroup_specification",
        "actions",
        "subgroups",
        "states",
        "sample_information",
        "tolerances",
        "estimator_assurance",
        "provenance",
    ],
    "properties": {
        "schema_version": {"const": "1.0.0"},
        "analysis_id": _ID,
        "analysis_type": {"const": "heterogeneity_value_decomposition"},
        "method_maturity": {"const": "experimental"},
        "objective": {
            "type": "object",
            "required": [
                "direction",
                "value_unit",
                "population_basis",
                "horizon_basis",
                "discount_basis",
            ],
            "properties": {
                "direction": {"enum": ["maximize", "minimize"]},
                "value_unit": _STRING,
                "population_basis": _STRING,
                "horizon_basis": _STRING,
                "discount_basis": _STRING,
            },
            "additionalProperties": False,
        },
        "subgroup_specification": {
            "type": "object",
            "required": [
                "specification_id",
                "covariates",
                "selection_policy",
                "multiplicity_policy",
                "fairness_constraints",
                "privacy_constraints",
            ],
            "properties": {
                "specification_id": _ID,
                "covariates": {
                    "type": "array",
                    "minItems": 1,
                    "uniqueItems": True,
                    "items": _STRING,
                },
                "selection_policy": _STRING,
                "multiplicity_policy": _STRING,
                "fairness_constraints": _STRING,
                "privacy_constraints": _STRING,
            },
            "additionalProperties": False,
        },
        "actions": {
            "type": "array",
            "minItems": 2,
            "items": {
                "type": "object",
                "required": ["action_id", "label"],
                "properties": {"action_id": _ID, "label": _STRING},
                "additionalProperties": False,
            },
        },
        "subgroups": {
            "type": "array",
            "minItems": 2,
            "items": {
                "type": "object",
                "required": [
                    "subgroup_id",
                    "label",
                    "weight",
                    "eligible_action_ids",
                ],
                "properties": {
                    "subgroup_id": _ID,
                    "label": _STRING,
                    "weight": {"type": "number", "exclusiveMinimum": 0, "maximum": 1},
                    "eligible_action_ids": {
                        "type": "array",
                        "minItems": 1,
                        "uniqueItems": True,
                        "items": _ID,
                    },
                },
                "additionalProperties": False,
            },
        },
        "states": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["state_id", "probability", "subgroup_action_values"],
                "properties": {
                    "state_id": _ID,
                    "probability": {
                        "type": "number",
                        "exclusiveMinimum": 0,
                        "maximum": 1,
                    },
                    "subgroup_action_values": {
                        "type": "object",
                        "minProperties": 2,
                        "additionalProperties": {
                            "type": "object",
                            "minProperties": 1,
                            "additionalProperties": {"type": "number"},
                        },
                    },
                },
                "additionalProperties": False,
            },
        },
        "sample_information": {
            "oneOf": [
                {"type": "null"},
                {
                    "type": "object",
                    "required": ["research_action_id", "cost", "signals"],
                    "properties": {
                        "research_action_id": _ID,
                        "cost": {
                            "type": "object",
                            "required": ["amount", "unit", "basis"],
                            "properties": {
                                "amount": {"type": "number", "minimum": 0},
                                "unit": _STRING,
                                "basis": _STRING,
                            },
                            "additionalProperties": False,
                        },
                        "signals": {
                            "type": "array",
                            "minItems": 1,
                            "items": {
                                "type": "object",
                                "required": ["signal_id", "likelihood_by_state"],
                                "properties": {
                                    "signal_id": _ID,
                                    "likelihood_by_state": {
                                        "type": "object",
                                        "minProperties": 1,
                                        "additionalProperties": _PROBABILITY,
                                    },
                                },
                                "additionalProperties": False,
                            },
                        },
                    },
                    "additionalProperties": False,
                },
            ]
        },
        "tolerances": {
            "type": "object",
            "required": ["probability_sum", "absolute_tie", "relative_tie"],
            "properties": {
                "probability_sum": {
                    "type": "number",
                    "exclusiveMinimum": 0,
                    "maximum": 1e-6,
                },
                "absolute_tie": {"type": "number", "minimum": 0, "maximum": 1e-6},
                "relative_tie": {"type": "number", "minimum": 0, "maximum": 1e-6},
            },
            "additionalProperties": False,
        },
        "estimator_assurance": {
            "type": "object",
            "required": ["estimator", "candidate_space_complete", "model_revision"],
            "properties": {
                "estimator": {"const": "exact_enumeration"},
                "candidate_space_complete": {"const": True},
                "model_revision": _STRING,
            },
            "additionalProperties": False,
        },
        "provenance": {
            "type": "object",
            "required": [
                "subgroup_source",
                "effect_source",
                "research_model_source",
                "software_version",
            ],
            "properties": {
                "subgroup_source": _STRING,
                "effect_source": _STRING,
                "research_model_source": _STRING,
                "software_version": _STRING,
            },
            "additionalProperties": False,
        },
    },
    "additionalProperties": False,
}

_NUMBER_MAP: Final[dict[str, object]] = {
    "type": "object",
    "minProperties": 1,
    "additionalProperties": {"type": "number"},
}
_POLICY_RESULT: Final[dict[str, object]] = {
    "type": "object",
    "required": ["action_values", "action_tie", "selected_action_id", "value"],
    "properties": {
        "action_values": _NUMBER_MAP,
        "action_tie": {
            "type": "array",
            "minItems": 1,
            "uniqueItems": True,
            "items": _ID,
        },
        "selected_action_id": _ID,
        "value": {"type": "number"},
    },
    "additionalProperties": False,
}
_INPUT_PROPERTIES = cast(
    "dict[str, Any]", HETEROGENEITY_VALUE_INPUT_SCHEMA_V1["properties"]
)
_OBJECTIVE_RESULT: Final[dict[str, object]] = cast(
    "dict[str, object]", _INPUT_PROPERTIES["objective"]
)
_SUBGROUP_SPEC_RESULT: Final[dict[str, object]] = cast(
    "dict[str, object]",
    _INPUT_PROPERTIES["subgroup_specification"],
)
_PROVENANCE_RESULT: Final[dict[str, object]] = cast(
    "dict[str, object]", _INPUT_PROPERTIES["provenance"]
)


def canonical_heterogeneity_value_input_sha256(payload: dict[str, Any]) -> str:
    """Return the canonical SHA-256 commitment for a strict input contract."""
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


HETEROGENEITY_VALUE_RESULT_SCHEMA_V1: Final[dict[str, object]] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://voiage.dev/schemas/frontier/heterogeneity-value-result.v1.json",
    "title": "HeterogeneityValueResultV1Experimental",
    "type": "object",
    "required": [
        "schema_version",
        "analysis_id",
        "analysis_type",
        "method_maturity",
        "objective",
        "four_value_decomposition",
        "perfect_information",
        "subgroup_results",
        "sample_information",
        "policy_audit",
        "assurance",
        "provenance",
        "language_dispositions",
    ],
    "properties": {
        "schema_version": {"const": "1.0.0"},
        "analysis_id": _ID,
        "analysis_type": {"const": "heterogeneity_value_decomposition_result"},
        "method_maturity": {"const": "experimental"},
        "objective": _OBJECTIVE_RESULT,
        "four_value_decomposition": {
            "type": "object",
            "required": [
                "c0",
                "cf",
                "p0",
                "pf",
                "static_value",
                "dynamic_value",
                "identity_residual",
            ],
            "properties": {
                key: {"type": "number"}
                for key in (
                    "c0",
                    "cf",
                    "p0",
                    "pf",
                    "static_value",
                    "dynamic_value",
                    "identity_residual",
                )
            },
            "additionalProperties": False,
        },
        "perfect_information": {
            "type": "object",
            "required": [
                "population_common_evpi",
                "subgroup_policy_evpi",
                "difference_identity",
            ],
            "properties": {
                key: {"type": "number"}
                for key in (
                    "population_common_evpi",
                    "subgroup_policy_evpi",
                    "difference_identity",
                )
            },
            "additionalProperties": False,
        },
        "subgroup_results": {
            "type": "array",
            "minItems": 2,
            "items": {
                "type": "object",
                "required": [
                    "subgroup_id",
                    "label",
                    "weight",
                    "current_action_values",
                    "current_action_tie",
                    "selected_current_action_id",
                    "current_value",
                    "perfect_information_value",
                    "evpi",
                ],
                "properties": {
                    "subgroup_id": _ID,
                    "label": _STRING,
                    "weight": {"type": "number", "exclusiveMinimum": 0, "maximum": 1},
                    "current_action_values": _NUMBER_MAP,
                    "current_action_tie": {
                        "type": "array",
                        "minItems": 1,
                        "uniqueItems": True,
                        "items": _ID,
                    },
                    "selected_current_action_id": _ID,
                    "current_value": {"type": "number"},
                    "perfect_information_value": {"type": "number"},
                    "evpi": {"type": "number"},
                },
                "additionalProperties": False,
            },
        },
        "sample_information": {
            "oneOf": [
                {"type": "null"},
                {
                    "type": "object",
                    "required": [
                        "research_action_id",
                        "s0",
                        "sf",
                        "population_common_evsi",
                        "subgroup_policy_evsi",
                        "sample_informed_segmentation_value",
                        "identity_residual",
                        "cost",
                        "population_common_net_evsi",
                        "subgroup_policy_net_evsi",
                        "signals",
                        "subgroup_evsi",
                        "current_subgroup_action_values",
                    ],
                    "properties": {
                        "research_action_id": _ID,
                        "s0": {"type": "number"},
                        "sf": {"type": "number"},
                        "population_common_evsi": {"type": "number"},
                        "subgroup_policy_evsi": {"type": "number"},
                        "sample_informed_segmentation_value": {"type": "number"},
                        "identity_residual": {"type": "number"},
                        "cost": {
                            "type": "object",
                            "required": ["amount", "unit", "basis"],
                            "properties": {
                                "amount": {"type": "number", "minimum": 0},
                                "unit": _STRING,
                                "basis": _STRING,
                            },
                            "additionalProperties": False,
                        },
                        "population_common_net_evsi": {"type": "number"},
                        "subgroup_policy_net_evsi": {"type": "number"},
                        "signals": {
                            "type": "array",
                            "minItems": 1,
                            "items": {
                                "type": "object",
                                "required": [
                                    "signal_id",
                                    "probability",
                                    "population_common",
                                    "subgroup_policies",
                                ],
                                "properties": {
                                    "signal_id": _ID,
                                    "probability": _PROBABILITY,
                                    "population_common": {
                                        "type": "object",
                                        "required": [
                                            "joint_weighted_action_values",
                                            "action_tie",
                                            "selected_action_id",
                                        ],
                                        "properties": {
                                            "joint_weighted_action_values": _NUMBER_MAP,
                                            "action_tie": {
                                                "type": "array",
                                                "minItems": 1,
                                                "uniqueItems": True,
                                                "items": _ID,
                                            },
                                            "selected_action_id": _ID,
                                        },
                                        "additionalProperties": False,
                                    },
                                    "subgroup_policies": {
                                        "type": "object",
                                        "minProperties": 2,
                                        "additionalProperties": {
                                            "type": "object",
                                            "required": [
                                                "joint_weighted_action_values",
                                                "action_tie",
                                                "selected_action_id",
                                            ],
                                            "properties": {
                                                "joint_weighted_action_values": _NUMBER_MAP,
                                                "action_tie": {
                                                    "type": "array",
                                                    "minItems": 1,
                                                    "uniqueItems": True,
                                                    "items": _ID,
                                                },
                                                "selected_action_id": _ID,
                                            },
                                            "additionalProperties": False,
                                        },
                                    },
                                },
                                "additionalProperties": False,
                            },
                        },
                        "subgroup_evsi": {
                            "type": "array",
                            "minItems": 2,
                            "items": {
                                "type": "object",
                                "required": [
                                    "subgroup_id",
                                    "weight",
                                    "current_value",
                                    "sample_value",
                                    "evsi",
                                    "weighted_evsi_contribution",
                                ],
                                "properties": {
                                    "subgroup_id": _ID,
                                    "weight": {
                                        "type": "number",
                                        "exclusiveMinimum": 0,
                                        "maximum": 1,
                                    },
                                    "current_value": {"type": "number"},
                                    "sample_value": {"type": "number"},
                                    "evsi": {"type": "number"},
                                    "weighted_evsi_contribution": {"type": "number"},
                                },
                                "additionalProperties": False,
                            },
                        },
                        "current_subgroup_action_values": {
                            "type": "object",
                            "minProperties": 2,
                            "additionalProperties": _NUMBER_MAP,
                        },
                    },
                    "additionalProperties": False,
                },
            ]
        },
        "policy_audit": {
            "type": "object",
            "required": [
                "current_population_common",
                "perfect_information_states",
                "subgroup_specification",
            ],
            "properties": {
                "current_population_common": {
                    "type": "object",
                    "required": ["action_values", "action_tie", "selected_action_id"],
                    "properties": {
                        "action_values": _NUMBER_MAP,
                        "action_tie": {
                            "type": "array",
                            "minItems": 1,
                            "uniqueItems": True,
                            "items": _ID,
                        },
                        "selected_action_id": _ID,
                    },
                    "additionalProperties": False,
                },
                "perfect_information_states": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "required": [
                            "state_id",
                            "probability",
                            "population_common",
                            "subgroup_policies",
                        ],
                        "properties": {
                            "state_id": _ID,
                            "probability": {
                                "type": "number",
                                "exclusiveMinimum": 0,
                                "maximum": 1,
                            },
                            "population_common": _POLICY_RESULT,
                            "subgroup_policies": {
                                "type": "object",
                                "minProperties": 2,
                                "additionalProperties": _POLICY_RESULT,
                            },
                        },
                        "additionalProperties": False,
                    },
                },
                "subgroup_specification": _SUBGROUP_SPEC_RESULT,
            },
            "additionalProperties": False,
        },
        "assurance": {
            "type": "object",
            "required": [
                "estimator",
                "candidate_space_complete",
                "model_revision",
                "input_sha256",
                "input_contract",
                "states_evaluated",
                "subgroups_evaluated",
                "common_actions_evaluated",
                "identity_verified",
                "selection_adjustment_performed",
                "sparse_subgroup_inference_performed",
            ],
            "properties": {
                "estimator": {"const": "exact_enumeration"},
                "candidate_space_complete": {"const": True},
                "model_revision": _STRING,
                "input_sha256": {
                    "type": "string",
                    "pattern": "^[0-9a-f]{64}$",
                },
                "input_contract": HETEROGENEITY_VALUE_INPUT_SCHEMA_V1,
                "states_evaluated": {"type": "integer", "minimum": 1},
                "subgroups_evaluated": {"type": "integer", "minimum": 2},
                "common_actions_evaluated": {"type": "integer", "minimum": 1},
                "identity_verified": {"const": True},
                "selection_adjustment_performed": {"const": False},
                "sparse_subgroup_inference_performed": {"const": False},
            },
            "additionalProperties": False,
        },
        "provenance": _PROVENANCE_RESULT,
        "language_dispositions": {
            "type": "object",
            "required": ["python", "rust", "r", "julia", "mojo"],
            "properties": {
                "python": {"const": "experimental_exact_execution"},
                "rust": {"const": "not_implemented"},
                "r": {"const": "not_implemented"},
                "julia": {"const": "not_implemented"},
                "mojo": {"const": "external_upstream_boundary"},
            },
            "additionalProperties": False,
        },
    },
    "additionalProperties": False,
}


def _validate_schema(payload: dict[str, Any], schema: dict[str, object]) -> None:
    errors = sorted(
        Draft202012Validator(schema).iter_errors(payload),
        key=lambda item: list(item.path),
    )
    if errors:
        error = errors[0]
        location = ".".join(str(part) for part in error.path) or "root"
        raise ValueError(f"{location}: {error.message}")


def validate_heterogeneity_value_semantics(payload: dict[str, Any]) -> None:
    """Validate cross-field semantics for the finite decomposition input."""
    _validate_schema(payload, HETEROGENEITY_VALUE_INPUT_SCHEMA_V1)
    tolerance = float(payload["tolerances"]["probability_sum"])
    actions = [str(item["action_id"]) for item in payload["actions"]]
    groups = [str(item["subgroup_id"]) for item in payload["subgroups"]]
    states = [str(item["state_id"]) for item in payload["states"]]
    if len(actions) != len(set(actions)):
        raise ValueError("action identifiers must be unique")
    if len(groups) != len(set(groups)):
        raise ValueError("subgroup identifiers must be unique")
    if len(states) != len(set(states)):
        raise ValueError("state identifiers must be unique")
    if not math.isclose(
        math.fsum(float(item["weight"]) for item in payload["subgroups"]),
        1.0,
        abs_tol=tolerance,
    ):
        raise ValueError("subgroup weights must sum to one")
    if not math.isclose(
        math.fsum(float(item["probability"]) for item in payload["states"]),
        1.0,
        abs_tol=tolerance,
    ):
        raise ValueError("state probabilities must sum to one")
    action_set = set(actions)
    common = action_set.copy()
    eligible_by_group: dict[str, set[str]] = {}
    for subgroup in payload["subgroups"]:
        group_id = str(subgroup["subgroup_id"])
        eligible = {str(value) for value in subgroup["eligible_action_ids"]}
        if not eligible <= action_set:
            raise ValueError(f"subgroup {group_id} refers to an unknown action")
        eligible_by_group[group_id] = eligible
        common &= eligible
    if not common:
        raise ValueError("at least one action must be eligible for every subgroup")
    for state in payload["states"]:
        state_id = str(state["state_id"])
        values = cast("dict[str, dict[str, object]]", state["subgroup_action_values"])
        if set(values) != set(groups):
            raise ValueError(f"state {state_id} must cover every subgroup exactly")
        for group_id, action_values in values.items():
            if set(action_values) != eligible_by_group[group_id]:
                raise ValueError(
                    f"state {state_id} subgroup {group_id} must cover eligible actions exactly"
                )
            if not all(
                math.isfinite(float(cast("float", value)))
                for value in action_values.values()
            ):
                raise ValueError("all subgroup action values must be finite")
    sample = payload["sample_information"]
    if sample is None:
        return
    if sample["cost"]["unit"] != payload["objective"]["value_unit"]:
        raise ValueError("sample cost unit must match the objective value unit")
    signals = sample["signals"]
    signal_ids = [str(item["signal_id"]) for item in signals]
    if len(signal_ids) != len(set(signal_ids)):
        raise ValueError("signal identifiers must be unique")
    for signal in signals:
        if set(signal["likelihood_by_state"]) != set(states):
            raise ValueError("each signal likelihood must cover every state exactly")
    for state_id in states:
        total = math.fsum(
            float(signal["likelihood_by_state"][state_id]) for signal in signals
        )
        if not math.isclose(total, 1.0, abs_tol=tolerance):
            raise ValueError(f"signal likelihoods for state {state_id} must sum to one")


def validate_heterogeneity_value_result(payload: dict[str, Any]) -> None:
    """Validate result shape and exact decomposition identities."""
    _validate_schema(payload, HETEROGENEITY_VALUE_RESULT_SCHEMA_V1)
    values = payload["four_value_decomposition"]
    pi = payload["perfect_information"]
    direction = payload["objective"]["direction"]
    sign = 1.0 if direction == "maximize" else -1.0
    static = sign * (float(values["cf"]) - float(values["c0"]))
    dynamic = sign * (float(values["pf"]) - float(values["p0"]))
    evpi0 = sign * (float(values["p0"]) - float(values["c0"]))
    evpif = sign * (float(values["pf"]) - float(values["cf"]))
    expected = (static, dynamic, evpi0, evpif, dynamic - static - (evpif - evpi0))
    observed = (
        float(values["static_value"]),
        float(values["dynamic_value"]),
        float(pi["population_common_evpi"]),
        float(pi["subgroup_policy_evpi"]),
        float(values["identity_residual"]),
    )
    if any(
        not math.isclose(left, right, abs_tol=1e-10)
        for left, right in zip(expected, observed, strict=True)
    ):
        raise ValueError("result violates the static/dynamic heterogeneity identity")
    if min(observed[:4]) < -1e-10:
        raise ValueError("optimized gross information values must be nonnegative")
    audit = payload["policy_audit"]
    current_common = audit["current_population_common"]
    optimize = max if direction == "maximize" else min
    reconstructed_c0 = optimize(
        float(value) for value in current_common["action_values"].values()
    )
    subgroup_rows = payload["subgroup_results"]
    reconstructed_cf = math.fsum(
        float(row["weight"])
        * optimize(float(value) for value in row["current_action_values"].values())
        for row in subgroup_rows
    )
    perfect_states = audit["perfect_information_states"]
    reconstructed_p0 = math.fsum(
        float(state["probability"])
        * optimize(
            float(value)
            for value in state["population_common"]["action_values"].values()
        )
        for state in perfect_states
    )
    weights = {str(row["subgroup_id"]): float(row["weight"]) for row in subgroup_rows}
    reconstructed_pf = math.fsum(
        float(state["probability"])
        * math.fsum(
            weights[group_id]
            * optimize(float(value) for value in group["action_values"].values())
            for group_id, group in state["subgroup_policies"].items()
        )
        for state in perfect_states
    )
    reconstructed_subgroup_perfect = {
        group_id: math.fsum(
            float(state["probability"])
            * optimize(
                float(value)
                for value in state["subgroup_policies"][group_id][
                    "action_values"
                ].values()
            )
            for state in perfect_states
        )
        for group_id in weights
    }
    reconstructed = (
        reconstructed_c0,
        reconstructed_cf,
        reconstructed_p0,
        reconstructed_pf,
    )
    reported = tuple(float(values[key]) for key in ("c0", "cf", "p0", "pf"))
    if any(
        not math.isclose(left, right, abs_tol=1e-10)
        for left, right in zip(reconstructed, reported, strict=True)
    ):
        raise ValueError("policy audit does not reconstruct C0, Cf, P0 and Pf")
    for row in subgroup_rows:
        group_id = str(row["subgroup_id"])
        current_value = optimize(
            float(value) for value in row["current_action_values"].values()
        )
        perfect_value = reconstructed_subgroup_perfect[group_id]
        if not (
            math.isclose(float(row["current_value"]), current_value, abs_tol=1e-10)
            and math.isclose(
                float(row["perfect_information_value"]),
                perfect_value,
                abs_tol=1e-10,
            )
            and math.isclose(
                float(row["evpi"]),
                sign * (perfect_value - current_value),
                abs_tol=1e-10,
            )
        ):
            raise ValueError("subgroup result does not reconstruct its EVPI")
    weighted_subgroup_evpi = math.fsum(
        float(row["weight"]) * float(row["evpi"]) for row in subgroup_rows
    )
    if not math.isclose(weighted_subgroup_evpi, evpif, abs_tol=1e-10):
        raise ValueError(  # pragma: no cover - implied by exact per-group audit
            "subgroup EVPI contributions do not reconstruct EVPIf"
        )
    sample = payload["sample_information"]
    if sample is not None:
        s0 = float(sample["s0"])
        sf = float(sample["sf"])
        evsi0 = sign * (s0 - float(values["c0"]))
        evsif = sign * (sf - float(values["cf"]))
        segmentation = sign * (sf - s0)
        sample_identity = segmentation - static - (evsif - evsi0)
        cost = float(sample["cost"]["amount"])
        expected_sample = (
            evsi0,
            evsif,
            segmentation,
            sample_identity,
            evsi0 - cost,
            evsif - cost,
        )
        observed_sample = tuple(
            float(sample[key])
            for key in (
                "population_common_evsi",
                "subgroup_policy_evsi",
                "sample_informed_segmentation_value",
                "identity_residual",
                "population_common_net_evsi",
                "subgroup_policy_net_evsi",
            )
        )
        if any(
            not math.isclose(left, right, abs_tol=1e-10)
            for left, right in zip(expected_sample, observed_sample, strict=True)
        ):
            raise ValueError("sample-information decomposition identity is violated")
        signals = sample["signals"]
        if not math.isclose(
            math.fsum(float(signal["probability"]) for signal in signals),
            1.0,
            abs_tol=1e-10,
        ):
            raise ValueError("reported signal probabilities must sum to one")
        reconstructed_s0 = math.fsum(
            optimize(
                float(value)
                for value in signal["population_common"][
                    "joint_weighted_action_values"
                ].values()
            )
            for signal in signals
        )
        reconstructed_sf = math.fsum(
            math.fsum(
                weights[group_id]
                * optimize(
                    float(value)
                    for value in group["joint_weighted_action_values"].values()
                )
                for group_id, group in signal["subgroup_policies"].items()
            )
            for signal in signals
        )
        if not (
            math.isclose(reconstructed_s0, s0, abs_tol=1e-10)
            and math.isclose(reconstructed_sf, sf, abs_tol=1e-10)
        ):
            raise ValueError("signal policy audit does not reconstruct S0 and Sf")
        subgroup_evsi = sample["subgroup_evsi"]
        for row in subgroup_evsi:
            group_id = str(row["subgroup_id"])
            reconstructed_sample_value = math.fsum(
                optimize(
                    float(value)
                    for value in signal["subgroup_policies"][group_id][
                        "joint_weighted_action_values"
                    ].values()
                )
                for signal in signals
            )
            row_evsi = sign * (float(row["sample_value"]) - float(row["current_value"]))
            if not (
                math.isclose(
                    float(row["sample_value"]),
                    reconstructed_sample_value,
                    abs_tol=1e-10,
                )
                and math.isclose(float(row["evsi"]), row_evsi, abs_tol=1e-10)
                and math.isclose(
                    float(row["weighted_evsi_contribution"]),
                    float(row["weight"]) * row_evsi,
                    abs_tol=1e-10,
                )
            ):
                raise ValueError("subgroup sample result does not reconstruct its EVSI")
        if not math.isclose(
            math.fsum(
                float(row["weighted_evsi_contribution"]) for row in subgroup_evsi
            ),
            evsif,
            abs_tol=1e-10,
        ):
            raise ValueError(  # pragma: no cover - implied by exact per-group audit
                "subgroup EVSI contributions do not reconstruct EVSIf"
            )

    assurance = cast("dict[str, Any]", payload["assurance"])
    input_contract = cast("dict[str, Any]", assurance["input_contract"])
    validate_heterogeneity_value_semantics(input_contract)
    if canonical_heterogeneity_value_input_sha256(input_contract) != str(
        assurance["input_sha256"]
    ):
        raise ValueError("result input contract commitment is invalid")

    # Import lazily to preserve the contract/method dependency direction at
    # module import time. _evaluate is deterministic and does not call the
    # public validator, so this is an exact standalone reconstruction rather
    # than a recursive validation path.
    evaluator_module = importlib.import_module("voiage.methods.heterogeneity_value")
    evaluate = evaluator_module._evaluate
    expected_result = cast("dict[str, Any]", evaluate(input_contract))
    if json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ) != json.dumps(
        expected_result,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ):
        raise ValueError("result does not exactly reproduce its committed input")
