"""Strict v1 contracts for finite information-source portfolio VOI."""

# pyright: reportAny=false, reportExplicitAny=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false

from __future__ import annotations

from copy import deepcopy
import math
from typing import TYPE_CHECKING, Any, Final, cast

from jsonschema import Draft202012Validator

if TYPE_CHECKING:
    from collections.abc import Mapping

    from jsonschema.exceptions import ValidationError

_ID: Final[dict[str, object]] = {
    "type": "string",
    "minLength": 1,
    "pattern": r"^[A-Za-z][A-Za-z0-9._-]*$",
}
_STRING: Final[dict[str, object]] = {"type": "string", "minLength": 1}
_NUMBER: Final[dict[str, object]] = {"type": "number"}
_NONNEGATIVE: Final[dict[str, object]] = {"type": "number", "minimum": 0}
_ID_ARRAY: Final[dict[str, object]] = {
    "type": "array",
    "uniqueItems": True,
    "items": _ID,
}

_RIGHTS: Final[dict[str, object]] = {
    "type": "object",
    "required": ["status", "license_id", "basis", "source_uri", "content_sha256"],
    "properties": {
        "status": {"const": "cleared"},
        "license_id": _STRING,
        "basis": _STRING,
        "source_uri": {"type": "string", "format": "uri", "minLength": 1},
        "content_sha256": {
            "type": "string",
            "pattern": "^[0-9a-f]{64}$",
        },
    },
    "additionalProperties": False,
}

INFORMATION_SOURCE_PORTFOLIO_INPUT_SCHEMA_V1: Final[dict[str, object]] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://voiage.dev/schemas/frontier/information-source-portfolio-input.v1.json",
    "title": "InformationSourcePortfolioInputV1Experimental",
    "type": "object",
    "required": [
        "schema_version",
        "analysis_id",
        "analysis_type",
        "method_maturity",
        "value_context",
        "actions",
        "states",
        "sources",
        "constraints",
        "tie_policy",
        "provenance",
    ],
    "properties": {
        "schema_version": {"const": "1.0.0"},
        "analysis_id": _ID,
        "analysis_type": {"const": "information_source_portfolio"},
        "method_maturity": {"const": "experimental"},
        "value_context": {
            "type": "object",
            "required": [
                "value_unit",
                "cost_unit",
                "time_unit",
                "population_basis",
                "horizon_basis",
                "discount_basis",
                "delay_cost_per_time",
            ],
            "properties": {
                "value_unit": _STRING,
                "cost_unit": _STRING,
                "time_unit": _STRING,
                "population_basis": _STRING,
                "horizon_basis": _STRING,
                "discount_basis": _STRING,
                "delay_cost_per_time": _NONNEGATIVE,
            },
            "additionalProperties": False,
        },
        "actions": {"type": "array", "minItems": 2, "uniqueItems": True, "items": _ID},
        "states": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": [
                    "state_id",
                    "probability",
                    "action_values",
                    "source_observations",
                ],
                "properties": {
                    "state_id": _ID,
                    "probability": {
                        "type": "number",
                        "exclusiveMinimum": 0,
                        "maximum": 1,
                    },
                    "action_values": {
                        "type": "object",
                        "minProperties": 2,
                        "additionalProperties": _NUMBER,
                    },
                    "source_observations": {
                        "type": "object",
                        "minProperties": 1,
                        "additionalProperties": _STRING,
                    },
                },
                "additionalProperties": False,
            },
        },
        "sources": {
            "type": "array",
            "minItems": 1,
            "maxItems": 7,
            "items": {
                "type": "object",
                "required": [
                    "source_id",
                    "label",
                    "cost",
                    "cost_unit",
                    "latency",
                    "privacy_cost",
                    "freshness_age",
                    "sla_probability",
                    "coverage",
                    "excludes",
                    "must_precede",
                    "rights",
                ],
                "properties": {
                    "source_id": _ID,
                    "label": _STRING,
                    "cost": _NONNEGATIVE,
                    "cost_unit": _STRING,
                    "latency": _NONNEGATIVE,
                    "privacy_cost": _NONNEGATIVE,
                    "freshness_age": _NONNEGATIVE,
                    "sla_probability": {"type": "number", "minimum": 0, "maximum": 1},
                    "coverage": _ID_ARRAY,
                    "excludes": _ID_ARRAY,
                    "must_precede": _ID_ARRAY,
                    "rights": _RIGHTS,
                },
                "additionalProperties": False,
            },
        },
        "constraints": {
            "type": "object",
            "required": [
                "max_cost",
                "max_latency",
                "max_privacy_cost",
                "max_sources",
                "min_source_sla",
                "max_freshness_age",
                "required_coverage",
            ],
            "properties": {
                "max_cost": _NONNEGATIVE,
                "max_latency": _NONNEGATIVE,
                "max_privacy_cost": _NONNEGATIVE,
                "max_sources": {"type": "integer", "minimum": 1, "maximum": 7},
                "min_source_sla": {"type": "number", "minimum": 0, "maximum": 1},
                "max_freshness_age": _NONNEGATIVE,
                "required_coverage": _ID_ARRAY,
            },
            "additionalProperties": False,
        },
        "tie_policy": {
            "type": "object",
            "required": [
                "absolute_tolerance",
                "relative_tolerance",
                "sequence_selection",
            ],
            "properties": {
                "absolute_tolerance": _NONNEGATIVE,
                "relative_tolerance": _NONNEGATIVE,
                "sequence_selection": {
                    "const": "highest_net_then_lower_cost_latency_lexical"
                },
            },
            "additionalProperties": False,
        },
        "provenance": {
            "type": "object",
            "required": [
                "decision_revision",
                "joint_world_source",
                "constraint_source",
                "evaluator",
                "software_version",
            ],
            "properties": {
                "decision_revision": _STRING,
                "joint_world_source": _STRING,
                "constraint_source": _STRING,
                "evaluator": _STRING,
                "software_version": _STRING,
            },
            "additionalProperties": False,
        },
    },
    "additionalProperties": False,
}

_PARTITION: Final[dict[str, object]] = {
    "type": "object",
    "required": [
        "partition_id",
        "observations",
        "probability",
        "conditional_action_values",
        "action_tie",
        "conditional_value",
        "switch_from_baseline",
    ],
    "properties": {
        "partition_id": _ID,
        "observations": {"type": "object", "additionalProperties": _STRING},
        "probability": {"type": "number", "exclusiveMinimum": 0, "maximum": 1},
        "conditional_action_values": {
            "type": "object",
            "minProperties": 2,
            "additionalProperties": _NUMBER,
        },
        "action_tie": {**_ID_ARRAY, "minItems": 1},
        "conditional_value": _NUMBER,
        "switch_from_baseline": {"type": "boolean"},
    },
    "additionalProperties": False,
}
_MARGINAL: Final[dict[str, object]] = {
    "type": "object",
    "required": [
        "position",
        "source_id",
        "conditioning_sources",
        "gross_marginal_value",
        "incremental_source_cost",
        "incremental_delay_cost",
        "net_marginal_value",
    ],
    "properties": {
        "position": {"type": "integer", "minimum": 1},
        "source_id": _ID,
        "conditioning_sources": _ID_ARRAY,
        "gross_marginal_value": _NUMBER,
        "incremental_source_cost": _NONNEGATIVE,
        "incremental_delay_cost": _NONNEGATIVE,
        "net_marginal_value": _NUMBER,
    },
    "additionalProperties": False,
}
_EVALUATION: Final[dict[str, object]] = {
    "type": "object",
    "required": [
        "source_sequence",
        "total_source_cost",
        "total_latency",
        "total_privacy_cost",
        "delay_cost",
        "resolved_value",
        "gross_value",
        "willingness_to_pay",
        "net_value",
        "partitions",
        "conditional_marginals",
    ],
    "properties": {
        "source_sequence": _ID_ARRAY,
        "total_source_cost": _NONNEGATIVE,
        "total_latency": _NONNEGATIVE,
        "total_privacy_cost": _NONNEGATIVE,
        "delay_cost": _NONNEGATIVE,
        "resolved_value": _NUMBER,
        "gross_value": _NUMBER,
        "willingness_to_pay": _NUMBER,
        "net_value": _NUMBER,
        "partitions": {"type": "array", "minItems": 1, "items": _PARTITION},
        "conditional_marginals": {"type": "array", "items": _MARGINAL},
    },
    "additionalProperties": False,
}
_OPTIMUM: Final[dict[str, object]] = {
    **_EVALUATION,
    "required": [*cast("list[str]", _EVALUATION["required"]), "optimal_sequence_tie"],
    "properties": {
        **cast("dict[str, object]", _EVALUATION["properties"]),
        "optimal_sequence_tie": {
            "type": "array",
            "minItems": 1,
            "items": _ID_ARRAY,
        },
    },
}

INFORMATION_SOURCE_PORTFOLIO_RESULT_SCHEMA_V1: Final[dict[str, object]] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://voiage.dev/schemas/frontier/information-source-portfolio-result.v1.json",
    "title": "InformationSourcePortfolioResultV1Experimental",
    "type": "object",
    "required": [
        "schema_version",
        "analysis_id",
        "analysis_type",
        "method_maturity",
        "value_context",
        "baseline",
        "evaluated_sequences",
        "optimum",
        "conditional_marginals",
        "attribution",
        "switches",
        "assurance",
        "provenance",
        "language_dispositions",
        "unsupported_dispositions",
    ],
    "properties": {
        "schema_version": {"const": "1.0.0"},
        "analysis_id": _ID,
        "analysis_type": {"const": "information_source_portfolio_result"},
        "method_maturity": {"const": "experimental"},
        "value_context": {"type": "object"},
        "baseline": {
            "type": "object",
            "required": ["expected_action_values", "action_tie", "value"],
            "properties": {
                "expected_action_values": {
                    "type": "object",
                    "minProperties": 2,
                    "additionalProperties": _NUMBER,
                },
                "action_tie": {**_ID_ARRAY, "minItems": 1},
                "value": _NUMBER,
            },
            "additionalProperties": False,
        },
        "evaluated_sequences": {"type": "array", "minItems": 1, "items": _EVALUATION},
        "optimum": _OPTIMUM,
        "conditional_marginals": {"type": "array", "items": _MARGINAL},
        "attribution": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["source_id", "gross_attribution", "attribution_method"],
                "properties": {
                    "source_id": _ID,
                    "gross_attribution": _NUMBER,
                    "attribution_method": {"const": "exact_decision_value_shapley"},
                },
                "additionalProperties": False,
            },
        },
        "switches": {"type": "array", "items": _PARTITION},
        "assurance": {"type": "object"},
        "provenance": {"type": "object"},
        "language_dispositions": {
            "type": "object",
            "additionalProperties": {"type": "string"},
        },
        "unsupported_dispositions": {"type": "array", "minItems": 1, "items": _STRING},
    },
    "additionalProperties": False,
}


def _v2_schemas() -> tuple[dict[str, object], dict[str, object]]:
    """Build versioned schemas without changing the established v1 contract."""
    input_schema = deepcopy(INFORMATION_SOURCE_PORTFOLIO_INPUT_SCHEMA_V1)
    input_schema["$id"] = (
        "https://voiage.dev/schemas/frontier/information-source-portfolio-input.v2.json"
    )
    input_schema["title"] = "InformationSourcePortfolioInputV2Experimental"
    properties = cast("dict[str, Any]", input_schema["properties"])
    properties["schema_version"] = {"const": "2.0.0"}
    source = properties["sources"]["items"]
    source["properties"]["asset_ids"] = _ID_ARRAY
    source["required"].append("asset_ids")
    properties["shared_costs"] = {
        "type": "object",
        "required": ["assets", "existing_assets"],
        "properties": {
            "assets": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["asset_id", "cost", "cost_unit"],
                    "properties": {
                        "asset_id": _ID,
                        "cost": _NONNEGATIVE,
                        "cost_unit": _STRING,
                    },
                    "additionalProperties": False,
                },
            },
            "existing_assets": _ID_ARRAY,
        },
        "additionalProperties": False,
    }
    cast("list[str]", input_schema["required"]).append("shared_costs")
    result_schema = deepcopy(INFORMATION_SOURCE_PORTFOLIO_RESULT_SCHEMA_V1)
    result_schema["$id"] = (
        "https://voiage.dev/schemas/frontier/information-source-portfolio-result.v2.json"
    )
    result_schema["title"] = "InformationSourcePortfolioResultV2Experimental"
    result_properties = cast("dict[str, Any]", result_schema["properties"])
    result_properties["schema_version"] = {"const": "2.0.0"}
    result_properties["cost_basis"] = {
        "type": "object",
        "required": ["shared_costs", "sources"],
        "properties": {
            "shared_costs": properties["shared_costs"],
            "sources": {
                "type": "object",
                "additionalProperties": {
                    "type": "object",
                    "required": ["cost", "cost_unit", "asset_ids"],
                    "properties": {
                        "cost": _NONNEGATIVE,
                        "cost_unit": _STRING,
                        "asset_ids": _ID_ARRAY,
                    },
                    "additionalProperties": False,
                },
            },
        },
        "additionalProperties": False,
    }
    cast("list[str]", result_schema["required"]).append("cost_basis")
    return input_schema, result_schema


(
    INFORMATION_SOURCE_PORTFOLIO_INPUT_SCHEMA_V2,
    INFORMATION_SOURCE_PORTFOLIO_RESULT_SCHEMA_V2,
) = _v2_schemas()


def _validation_message(error: ValidationError) -> str:
    path = "/".join(str(item) for item in error.absolute_path)
    return f"contract validation failed at {path or '<root>'}: {error.message}"


def _validate_schema(payload: Mapping[str, Any], schema: Mapping[str, object]) -> None:
    errors = sorted(
        Draft202012Validator(schema).iter_errors(payload),
        key=lambda item: list(item.absolute_path),
    )
    if errors:
        raise ValueError(_validation_message(errors[0]))


def validate_information_source_portfolio_semantics(payload: Mapping[str, Any]) -> None:
    """Validate cross-field input semantics beyond JSON Schema."""
    version = payload.get("schema_version")
    schema = (
        INFORMATION_SOURCE_PORTFOLIO_INPUT_SCHEMA_V2
        if version == "2.0.0"
        else INFORMATION_SOURCE_PORTFOLIO_INPUT_SCHEMA_V1
    )
    _validate_schema(payload, schema)
    actions = cast("list[str]", payload["actions"])
    sources = cast("list[Mapping[str, Any]]", payload["sources"])
    states = cast("list[Mapping[str, Any]]", payload["states"])
    action_ids = set(actions)
    source_ids = [cast("str", source["source_id"]) for source in sources]
    if len(action_ids) != len(actions):  # pragma: no cover - schema uniqueItems
        raise ValueError("action IDs must be unique")
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("source IDs must be unique")
    state_ids = [cast("str", state["state_id"]) for state in states]
    if len(set(state_ids)) != len(state_ids):
        raise ValueError("state IDs must be unique")
    probability = math.fsum(float(state["probability"]) for state in states)
    if not math.isclose(probability, 1.0, abs_tol=1e-9, rel_tol=0.0):
        raise ValueError("state probabilities must sum to one")
    expected_sources = set(source_ids)
    for state in states:
        if set(cast("Mapping[str, Any]", state["action_values"])) != action_ids:
            raise ValueError("every state must define every action value")
        if (
            set(cast("Mapping[str, Any]", state["source_observations"]))
            != expected_sources
        ):
            raise ValueError("every state must define every source observation")
    context = cast("Mapping[str, Any]", payload["value_context"])
    cost_unit = context["cost_unit"]
    if context["value_unit"] != cost_unit:
        raise ValueError("value and cost units must be directly commensurate")
    if any(source["cost_unit"] != cost_unit for source in sources):
        raise ValueError(
            "every source cost unit must match the value context cost unit"
        )
    if version == "2.0.0":
        shared = cast("Mapping[str, Any]", payload["shared_costs"])
        assets = cast("list[Mapping[str, Any]]", shared["assets"])
        asset_ids = [cast("str", asset["asset_id"]) for asset in assets]
        if len(set(asset_ids)) != len(asset_ids):
            raise ValueError("duplicate shared asset IDs")
        if not set(cast("list[str]", shared["existing_assets"])) <= set(asset_ids):
            raise ValueError("unknown existing asset")
        if any(
            not set(cast("list[str]", source["asset_ids"])) <= set(asset_ids)
            for source in sources
        ):
            raise ValueError("unknown source asset")
        if any(
            asset["cost_unit"] != cost_unit or not math.isfinite(float(asset["cost"]))
            for asset in assets
        ):
            raise ValueError(
                "shared asset cost must be finite and match the value context unit"
            )
    numeric_values = [
        float(context["delay_cost_per_time"]),
        *(float(state["probability"]) for state in states),
        *(
            float(value)
            for state in states
            for value in cast("Mapping[str, float]", state["action_values"]).values()
        ),
        *(
            float(source[field])
            for source in sources
            for field in (
                "cost",
                "latency",
                "privacy_cost",
                "freshness_age",
                "sla_probability",
            )
        ),
        *(
            float(cast("Mapping[str, Any]", payload["constraints"])[field])
            for field in (
                "max_cost",
                "max_latency",
                "max_privacy_cost",
                "min_source_sla",
                "max_freshness_age",
            )
        ),
        float(cast("Mapping[str, Any]", payload["tie_policy"])["absolute_tolerance"]),
        float(cast("Mapping[str, Any]", payload["tie_policy"])["relative_tolerance"]),
    ]
    if not all(math.isfinite(value) for value in numeric_values):
        raise ValueError(
            "all probabilities, values, costs and constraints must be finite"
        )
    references = set(source_ids)
    graph: dict[str, set[str]] = {}
    for source in sources:
        source_id = cast("str", source["source_id"])
        excludes = set(cast("list[str]", source["excludes"]))
        precedes = set(cast("list[str]", source["must_precede"]))
        if source_id in excludes | precedes or not (excludes | precedes) <= references:
            raise ValueError(
                "source order and exclusion references must name other declared sources"
            )
        graph[source_id] = precedes
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in visiting:
            raise ValueError("source order constraint cycle detected")
        if node in visited:
            return
        visiting.add(node)
        for successor in graph[node]:
            visit(successor)
        visiting.remove(node)
        visited.add(node)

    for source_id in sorted(graph):
        visit(source_id)


def _validate_v2_cost_basis(payload: Mapping[str, Any]) -> None:
    """Recompute all v2 source and asset costs from the retained cost basis."""
    basis = cast("Mapping[str, Any]", payload["cost_basis"])
    shared = cast("Mapping[str, Any]", basis["shared_costs"])
    sources = cast("Mapping[str, Mapping[str, Any]]", basis["sources"])
    assets = cast("list[Mapping[str, Any]]", shared["assets"])
    asset_by_id = {asset["asset_id"]: asset for asset in assets}
    owned = set(cast("list[str]", shared["existing_assets"]))
    unit = cast("Mapping[str, Any]", payload["value_context"])["cost_unit"]
    if len(asset_by_id) != len(assets) or not owned <= set(asset_by_id):
        raise ValueError("invalid cost basis assets")
    if any(
        asset["cost_unit"] != unit or not math.isfinite(float(asset["cost"]))
        for asset in assets
    ):
        raise ValueError("invalid cost basis asset cost or unit")
    for source in sources.values():
        if (
            source["cost_unit"] != unit
            or not math.isfinite(float(source["cost"]))
            or not set(source["asset_ids"]) <= set(asset_by_id)
        ):
            raise ValueError("invalid cost basis source")
    for evaluation in cast("list[Mapping[str, Any]]", payload["evaluated_sequences"]):
        sequence = cast("list[str]", evaluation["source_sequence"])
        if len(sequence) != len(set(sequence)) or not set(sequence) <= set(sources):
            raise ValueError("invalid cost basis sequence")
        seen = set(owned)
        increments: list[float] = []
        for source_id in sequence:
            source = sources[source_id]
            new_assets = set(source["asset_ids"]) - seen
            increments.append(
                math.fsum(
                    [
                        float(source["cost"]),
                        *(float(asset_by_id[a]["cost"]) for a in sorted(new_assets)),
                    ]
                )
            )
            seen.update(new_assets)
        marginals = cast("list[Mapping[str, Any]]", evaluation["conditional_marginals"])
        if (
            len(marginals) != len(sequence)
            or any(
                item["source_id"] != source_id
                or not math.isclose(
                    float(item["incremental_source_cost"]),
                    cost,
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                )
                for item, source_id, cost in zip(
                    marginals, sequence, increments, strict=True
                )
            )
            or not math.isclose(
                float(evaluation["total_source_cost"]),
                math.fsum(increments),
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
        ):
            raise ValueError("result costs do not match cost basis")


def validate_information_source_portfolio_result(payload: Mapping[str, Any]) -> None:
    """Validate the portable result envelope."""
    schema = (
        INFORMATION_SOURCE_PORTFOLIO_RESULT_SCHEMA_V2
        if payload.get("schema_version") == "2.0.0"
        else INFORMATION_SOURCE_PORTFOLIO_RESULT_SCHEMA_V1
    )
    _validate_schema(payload, schema)
    if payload.get("schema_version") == "2.0.0":
        _validate_v2_cost_basis(payload)
    baseline = float(cast("Mapping[str, Any]", payload["baseline"])["value"])
    evaluations = cast("list[Mapping[str, Any]]", payload["evaluated_sequences"])
    for evaluation in evaluations:
        resolved = float(evaluation["resolved_value"])
        gross = float(evaluation["gross_value"])
        delay = float(evaluation["delay_cost"])
        source_cost = float(evaluation["total_source_cost"])
        if not math.isclose(resolved - baseline, gross, abs_tol=1e-9):
            raise ValueError("resolved and gross decision-value identity failed")
        if not math.isclose(
            gross - delay,
            float(evaluation["willingness_to_pay"]),
            abs_tol=1e-9,
        ):
            raise ValueError("willingness-to-pay identity failed")
        if not math.isclose(
            gross - delay - source_cost,
            float(evaluation["net_value"]),
            abs_tol=1e-9,
        ):
            raise ValueError("net decision-value identity failed")
        marginals = cast("list[Mapping[str, Any]]", evaluation["conditional_marginals"])
        if not math.isclose(
            math.fsum(float(item["gross_marginal_value"]) for item in marginals),
            gross,
            abs_tol=1e-9,
        ):
            raise ValueError("conditional marginal values must recover gross value")
    optimum = cast("Mapping[str, Any]", payload["optimum"])
    if payload.get("schema_version") == "2.0.0":
        selected = {
            key: value
            for key, value in optimum.items()
            if key != "optimal_sequence_tie"
        }
        if (
            selected not in evaluations
            or payload["conditional_marginals"] != optimum["conditional_marginals"]
        ):
            raise ValueError("reported optimum must match an evaluated sequence")
    if (
        float(optimum["net_value"])
        < max(float(item["net_value"]) for item in evaluations) - 1e-9
    ):
        raise ValueError("reported optimum is not maximal")
    attribution = cast("list[Mapping[str, Any]]", payload["attribution"])
    if not math.isclose(
        math.fsum(float(item["gross_attribution"]) for item in attribution),
        float(optimum["gross_value"]),
        abs_tol=1e-9,
    ):
        raise ValueError("decision-value attribution must recover selected gross value")
