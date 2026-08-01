"""Strict v1 contract for experimental forecast-signal decision value."""

# pyright: reportAny=false, reportExplicitAny=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Final, cast

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

if TYPE_CHECKING:
    from collections.abc import Mapping

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
_ID_ARRAY: Final[dict[str, object]] = {
    "type": "array",
    "minItems": 1,
    "uniqueItems": True,
    "items": _ID,
}

FORECAST_SIGNAL_INFORMATION_INPUT_SCHEMA_V1: Final[dict[str, object]] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://voiage.dev/schemas/frontier/forecast-signal-information-input.v1.json",
    "title": "ForecastSignalInformationInputV1Experimental",
    "type": "object",
    "required": [
        "schema_version",
        "analysis_id",
        "analysis_type",
        "method_maturity",
        "objective",
        "timing",
        "outcomes",
        "actions",
        "feasible_action_ids",
        "signals",
        "signal_cost",
        "tolerances",
        "provenance",
    ],
    "properties": {
        "schema_version": {"const": "1.0.0"},
        "analysis_id": _ID,
        "analysis_type": {"const": "forecast_signal_information"},
        "method_maturity": {"const": "experimental"},
        "objective": {
            "type": "object",
            "required": [
                "direction",
                "value_unit",
                "population_basis",
                "discount_basis",
            ],
            "properties": {
                "direction": {"const": "maximize"},
                "value_unit": _STRING,
                "population_basis": _STRING,
                "discount_basis": _STRING,
            },
            "additionalProperties": False,
        },
        "timing": {
            "type": "object",
            "required": [
                "time_unit",
                "forecast_origin",
                "information_available",
                "decision_time",
                "outcome_time",
                "maximum_freshness",
                "dependence_assumption",
            ],
            "properties": {
                "time_unit": _STRING,
                "forecast_origin": {"type": "number"},
                "information_available": {"type": "number"},
                "decision_time": {"type": "number"},
                "outcome_time": {"type": "number"},
                "maximum_freshness": {"type": "number", "minimum": 0},
                "dependence_assumption": _STRING,
            },
            "additionalProperties": False,
        },
        "outcomes": {
            "type": "array",
            "minItems": 2,
            "items": {
                "type": "object",
                "required": ["outcome_id", "label", "probability"],
                "properties": {
                    "outcome_id": _ID,
                    "label": _STRING,
                    "probability": _PROBABILITY,
                },
                "additionalProperties": False,
            },
        },
        "actions": {
            "type": "array",
            "minItems": 2,
            "items": {
                "type": "object",
                "required": [
                    "action_id",
                    "label",
                    "outcome_values",
                    "constraint_basis",
                ],
                "properties": {
                    "action_id": _ID,
                    "label": _STRING,
                    "outcome_values": {
                        "type": "object",
                        "minProperties": 2,
                        "additionalProperties": {"type": "number"},
                    },
                    "constraint_basis": _STRING,
                },
                "additionalProperties": False,
            },
        },
        "feasible_action_ids": _ID_ARRAY,
        "signals": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": [
                    "signal_id",
                    "label",
                    "likelihood_by_outcome",
                    "reported_outcome_probabilities",
                ],
                "properties": {
                    "signal_id": _ID,
                    "label": _STRING,
                    "likelihood_by_outcome": {
                        "type": "object",
                        "minProperties": 2,
                        "additionalProperties": _PROBABILITY,
                    },
                    "reported_outcome_probabilities": {
                        "type": "object",
                        "minProperties": 2,
                        "additionalProperties": _PROBABILITY,
                    },
                },
                "additionalProperties": False,
            },
        },
        "signal_cost": {
            "type": "object",
            "required": ["amount", "unit", "cost_scope", "source"],
            "properties": {
                "amount": {"type": "number", "minimum": 0},
                "unit": _STRING,
                "cost_scope": {"const": "acquisition_before_signal"},
                "source": _STRING,
            },
            "additionalProperties": False,
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
        "provenance": {
            "type": "object",
            "required": [
                "forecast_artifact",
                "payoff_source",
                "created_at",
                "training_performed",
            ],
            "properties": {
                "forecast_artifact": _STRING,
                "payoff_source": _STRING,
                "created_at": {"type": "string", "format": "date-time"},
                "training_performed": {"const": False},
            },
            "additionalProperties": False,
        },
    },
    "additionalProperties": False,
}

_SCORE_MAP: Final[dict[str, object]] = {
    "type": "object",
    "minProperties": 1,
    "additionalProperties": {"type": "number"},
}
_PROBABILITY_MAP: Final[dict[str, object]] = {
    "type": "object",
    "minProperties": 1,
    "additionalProperties": _PROBABILITY,
}

FORECAST_SIGNAL_INFORMATION_RESULT_SCHEMA_V1: Final[dict[str, object]] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://voiage.dev/schemas/frontier/forecast-signal-information-result.v1.json",
    "title": "ForecastSignalInformationResultV1Experimental",
    "type": "object",
    "required": [
        "schema_version",
        "analysis_id",
        "analysis_type",
        "method_maturity",
        "value_unit",
        "timing",
        "baseline",
        "signal_partitions",
        "value",
        "regret",
        "diagnostics",
        "assurance",
        "language_dispositions",
        "unsupported_dispositions",
    ],
    "properties": {
        "schema_version": {"const": "1.0.0"},
        "analysis_id": _ID,
        "analysis_type": {"const": "forecast_signal_information_result"},
        "method_maturity": {"const": "experimental"},
        "value_unit": _STRING,
        "timing": {
            "type": "object",
            "required": [
                "time_unit",
                "horizon",
                "freshness",
                "latency",
                "lead_time",
                "operationally_usable",
                "reason",
            ],
            "properties": {
                "time_unit": _STRING,
                "horizon": {"type": "number", "minimum": 0},
                "freshness": {"type": "number", "minimum": 0},
                "latency": {"type": "number", "minimum": 0},
                "lead_time": {"type": "number", "minimum": 0},
                "operationally_usable": {"type": "boolean"},
                "reason": _STRING,
            },
            "additionalProperties": False,
        },
        "baseline": {
            "type": "object",
            "required": ["expected_action_values", "choice_tie", "value"],
            "properties": {
                "expected_action_values": _SCORE_MAP,
                "choice_tie": _ID_ARRAY,
                "value": {"type": "number"},
            },
            "additionalProperties": False,
        },
        "signal_partitions": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": [
                    "signal_id",
                    "probability",
                    "posterior_outcomes",
                    "reported_outcomes",
                    "oracle_choice_tie",
                    "deployed_choice_tie",
                    "oracle_conditional_value",
                    "deployed_conditional_value",
                    "value_contribution",
                    "calibration_l1",
                ],
                "properties": {
                    "signal_id": _ID,
                    "probability": _PROBABILITY,
                    "posterior_outcomes": _PROBABILITY_MAP,
                    "reported_outcomes": _PROBABILITY_MAP,
                    "oracle_choice_tie": _ID_ARRAY,
                    "deployed_choice_tie": _ID_ARRAY,
                    "oracle_conditional_value": {"type": "number"},
                    "deployed_conditional_value": {"type": "number"},
                    "value_contribution": {"type": "number"},
                    "calibration_l1": {"type": "number", "minimum": 0},
                },
                "additionalProperties": False,
            },
        },
        "value": {
            "type": "object",
            "required": [
                "counterfactual_timely_oracle",
                "gross_deployed",
                "calibration_loss",
                "cost",
                "net_deployed",
                "maximum_price",
            ],
            "properties": {
                "counterfactual_timely_oracle": {"type": "number", "minimum": 0},
                "gross_deployed": {"type": "number"},
                "calibration_loss": {"type": "number", "minimum": 0},
                "cost": {"type": "number", "minimum": 0},
                "net_deployed": {"type": "number"},
                "maximum_price": {"type": "number", "minimum": 0},
            },
            "additionalProperties": False,
        },
        "regret": {
            "type": "object",
            "required": ["baseline_expected", "deployed_expected", "avoided"],
            "properties": {
                "baseline_expected": {"type": "number", "minimum": 0},
                "deployed_expected": {"type": "number", "minimum": 0},
                "avoided": {"type": "number"},
            },
            "additionalProperties": False,
        },
        "diagnostics": {
            "type": "object",
            "required": [
                "weighted_calibration_l1",
                "reported_brier",
                "posterior_reference_brier",
                "excess_brier",
                "signal_probability_coverage",
            ],
            "properties": {
                "weighted_calibration_l1": {"type": "number", "minimum": 0},
                "reported_brier": {"type": "number", "minimum": 0},
                "posterior_reference_brier": {"type": "number", "minimum": 0},
                "excess_brier": {"type": "number", "minimum": 0},
                "signal_probability_coverage": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                },
            },
            "additionalProperties": False,
        },
        "assurance": {
            "type": "object",
            "required": [
                "estimator",
                "joint_law",
                "decision_policy",
                "accuracy_is_value",
                "training_performed",
            ],
            "properties": {
                "estimator": {"const": "exact_finite_enumeration"},
                "joint_law": {"const": "outcome_prior_times_signal_likelihood"},
                "decision_policy": {
                    "const": "reported_probabilities_evaluated_under_joint_law"
                },
                "accuracy_is_value": {"const": False},
                "training_performed": {"const": False},
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
        "unsupported_dispositions": {"type": "array", "items": _STRING},
    },
    "additionalProperties": False,
}


def _validate(
    schema: Mapping[str, object], payload: Mapping[str, Any], label: str
) -> None:
    try:
        Draft202012Validator(schema).validate(payload)
    except ValidationError as error:
        path = "/".join(str(item) for item in error.absolute_path) or "$"
        raise ValueError(
            f"invalid {label} at {path}: constraint {error.validator}"
        ) from error


def _ids(records: list[Mapping[str, Any]], key: str, label: str) -> list[str]:
    values = [cast("str", item[key]) for item in records]
    if len(values) != len(set(values)):
        raise ValueError(f"{label} IDs must be unique")
    return values


def _finite_sum(values: Mapping[str, Any], label: str, tolerance: float) -> None:
    numeric = [float(value) for value in values.values()]
    if not all(math.isfinite(value) for value in numeric):
        raise ValueError(f"{label} must be finite")
    if not math.isclose(math.fsum(numeric), 1.0, abs_tol=tolerance, rel_tol=0.0):
        raise ValueError(f"{label} must sum to 1")


def validate_forecast_signal_information_semantics(payload: Mapping[str, Any]) -> None:
    """Validate schema and cross-record invariants for a v1 request."""
    _validate(
        FORECAST_SIGNAL_INFORMATION_INPUT_SCHEMA_V1, payload, "forecast-signal input"
    )
    tolerance = float(
        cast("Mapping[str, Any]", payload["tolerances"])["probability_sum"]
    )
    tolerances = cast("Mapping[str, Any]", payload["tolerances"])
    if not all(math.isfinite(float(value)) for value in tolerances.values()):
        raise ValueError("tolerances must be finite")
    outcomes = cast("list[Mapping[str, Any]]", payload["outcomes"])
    actions = cast("list[Mapping[str, Any]]", payload["actions"])
    signals = cast("list[Mapping[str, Any]]", payload["signals"])
    outcome_ids = _ids(outcomes, "outcome_id", "outcome")
    action_ids = _ids(actions, "action_id", "action")
    _ = _ids(signals, "signal_id", "signal")
    _finite_sum(
        {cast("str", item["outcome_id"]): item["probability"] for item in outcomes},
        "outcome probabilities",
        tolerance,
    )
    feasible = cast("list[str]", payload["feasible_action_ids"])
    if not feasible or not set(feasible) <= set(action_ids):
        raise ValueError("feasible action IDs must be a non-empty subset of actions")
    for action in actions:
        values = cast("Mapping[str, Any]", action["outcome_values"])
        if set(values) != set(outcome_ids):
            raise ValueError(
                "every action outcome-value map must exactly match outcomes"
            )
        if not all(math.isfinite(float(value)) for value in values.values()):
            raise ValueError("action outcome values must be finite")
    likelihood_sums = dict.fromkeys(outcome_ids, 0.0)
    priors = {
        cast("str", item["outcome_id"]): float(item["probability"]) for item in outcomes
    }
    for signal in signals:
        likelihood = cast("Mapping[str, Any]", signal["likelihood_by_outcome"])
        reported = cast("Mapping[str, Any]", signal["reported_outcome_probabilities"])
        if set(likelihood) != set(outcome_ids) or set(reported) != set(outcome_ids):
            raise ValueError("signal probability maps must exactly match outcomes")
        _finite_sum(reported, "reported outcome probabilities", tolerance)
        for outcome_id in outcome_ids:
            likelihood_sums[outcome_id] += float(likelihood[outcome_id])
        marginal = math.fsum(
            priors[outcome_id] * float(likelihood[outcome_id])
            for outcome_id in outcome_ids
        )
        if marginal <= 0:
            raise ValueError(
                "every declared signal must have positive marginal probability"
            )
    for outcome_id, total in likelihood_sums.items():
        if not math.isclose(total, 1.0, abs_tol=tolerance, rel_tol=0.0):
            raise ValueError(
                f"signal likelihoods for outcome {outcome_id} must sum to 1"
            )
    timing = cast("Mapping[str, Any]", payload["timing"])
    origin = float(timing["forecast_origin"])
    available = float(timing["information_available"])
    decision = float(timing["decision_time"])
    outcome = float(timing["outcome_time"])
    maximum_freshness = float(timing["maximum_freshness"])
    if not all(
        math.isfinite(value)
        for value in (origin, available, decision, outcome, maximum_freshness)
    ):
        raise ValueError("timing values must be finite")
    if not origin <= available <= outcome or not origin <= decision <= outcome:
        raise ValueError("timing must satisfy origin <= available/decision <= outcome")
    objective = cast("Mapping[str, Any]", payload["objective"])
    cost = cast("Mapping[str, Any]", payload["signal_cost"])
    if not math.isfinite(float(cost["amount"])):
        raise ValueError("signal cost amount must be finite")
    if cost["unit"] != objective["value_unit"]:
        raise ValueError("signal cost unit must match objective value unit")


def validate_forecast_signal_information_result_semantics(
    payload: Mapping[str, Any],
) -> None:
    """Validate the strict v1 result envelope."""
    _validate(
        FORECAST_SIGNAL_INFORMATION_RESULT_SCHEMA_V1, payload, "forecast-signal result"
    )
    tolerance = 1e-9
    baseline = cast("Mapping[str, Any]", payload["baseline"])
    baseline_scores = cast("Mapping[str, Any]", baseline["expected_action_values"])
    if not set(cast("list[str]", baseline["choice_tie"])) <= set(baseline_scores):
        raise ValueError("baseline choices must identify declared action scores")
    partitions = cast("list[Mapping[str, Any]]", payload["signal_partitions"])
    _finite_sum(
        {cast("str", item["signal_id"]): item["probability"] for item in partitions},
        "result signal probabilities",
        tolerance,
    )
    for partition in partitions:
        _finite_sum(
            cast("Mapping[str, Any]", partition["posterior_outcomes"]),
            "posterior outcome probabilities",
            tolerance,
        )
        _finite_sum(
            cast("Mapping[str, Any]", partition["reported_outcomes"]),
            "reported result outcome probabilities",
            tolerance,
        )
    value = cast("Mapping[str, Any]", payload["value"])
    gross = float(value["gross_deployed"])
    cost = float(value["cost"])
    identities = (
        (float(value["net_deployed"]), gross - cost, "net deployed value"),
        (float(value["maximum_price"]), max(0.0, gross), "maximum price"),
        (
            math.fsum(float(item["value_contribution"]) for item in partitions),
            gross,
            "signal value contributions",
        ),
    )
    for actual, expected, label in identities:
        if not math.isclose(actual, expected, abs_tol=tolerance, rel_tol=tolerance):
            raise ValueError(f"{label} does not reconcile")
    regret = cast("Mapping[str, Any]", payload["regret"])
    if not math.isclose(
        float(regret["avoided"]), gross, abs_tol=tolerance, rel_tol=tolerance
    ):
        raise ValueError("regret avoided must equal gross deployed value")
