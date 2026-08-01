"""Exact finite evaluator for experimental forecast and signal information."""

# pyright: reportAny=false, reportExplicitAny=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import TYPE_CHECKING, Any, cast

from voiage.contracts.forecast_signal_information import (
    validate_forecast_signal_information_result_semantics,
    validate_forecast_signal_information_semantics,
)
from voiage.exceptions import raise_input_error

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


@dataclass(frozen=True)
class ForecastSignalInformationResult:
    """Portable result for exact finite forecast-signal decision value."""

    payload: dict[str, Any]

    def to_contract_dict(self) -> dict[str, Any]:
        """Return an independent JSON-compatible result copy."""
        return cast("dict[str, Any]", json.loads(json.dumps(self.payload)))


def _choice(values: Mapping[str, float], absolute: float, relative: float) -> list[str]:
    maximum = max(values.values())
    return sorted(
        action_id
        for action_id, value in values.items()
        if math.isclose(value, maximum, abs_tol=absolute, rel_tol=relative)
    )


def _policy_value(choice: Sequence[str], values: Mapping[str, float]) -> float:
    return math.fsum(values[action_id] for action_id in choice) / len(choice)


def forecast_signal_information_value(
    specification: Mapping[str, object],
) -> ForecastSignalInformationResult:
    """Value a declared finite probabilistic forecast through its decisions.

    The function consumes a forecast artifact and a frozen payoff model. It
    does not fit, tune, recalibrate, or otherwise train a forecasting model.
    """
    try:
        payload = cast(
            "dict[str, Any]",
            json.loads(json.dumps(specification, ensure_ascii=False)),
        )
        validate_forecast_signal_information_semantics(payload)
        result = _evaluate(payload)
        validate_forecast_signal_information_result_semantics(result)
    except (ArithmeticError, KeyError, TypeError, ValueError) as error:
        raise_input_error(str(error))
    return ForecastSignalInformationResult(result)


def _evaluate(payload: Mapping[str, Any]) -> dict[str, Any]:
    outcomes = cast("list[Mapping[str, Any]]", payload["outcomes"])
    actions = cast("list[Mapping[str, Any]]", payload["actions"])
    signals = cast("list[Mapping[str, Any]]", payload["signals"])
    outcome_ids = [cast("str", item["outcome_id"]) for item in outcomes]
    feasible = cast("list[str]", payload["feasible_action_ids"])
    priors = {
        cast("str", item["outcome_id"]): float(item["probability"]) for item in outcomes
    }
    payoffs = {
        cast("str", action["action_id"]): {
            outcome_id: float(
                cast("Mapping[str, Any]", action["outcome_values"])[outcome_id]
            )
            for outcome_id in outcome_ids
        }
        for action in actions
        if action["action_id"] in feasible
    }
    tolerances = cast("Mapping[str, Any]", payload["tolerances"])
    absolute = float(tolerances["absolute_tie"])
    relative = float(tolerances["relative_tie"])
    baseline_values = {
        action_id: math.fsum(
            priors[outcome_id] * values[outcome_id] for outcome_id in outcome_ids
        )
        for action_id, values in payoffs.items()
    }
    baseline_choice = _choice(baseline_values, absolute, relative)
    baseline_value = _policy_value(baseline_choice, baseline_values)

    timing = cast("Mapping[str, Any]", payload["timing"])
    origin = float(timing["forecast_origin"])
    available = float(timing["information_available"])
    decision = float(timing["decision_time"])
    outcome_time = float(timing["outcome_time"])
    freshness = decision - origin
    operationally_usable = available <= decision and freshness <= float(
        timing["maximum_freshness"]
    )
    if available > decision:
        timing_reason = "information_available_after_decision"
    elif freshness > float(timing["maximum_freshness"]):
        timing_reason = "forecast_exceeds_maximum_freshness"
    else:
        timing_reason = "available_and_fresh_at_decision"

    partitions: list[dict[str, Any]] = []
    timely_oracle_resolved = 0.0
    timely_deployed_resolved = 0.0
    operational_resolved = 0.0
    weighted_calibration = 0.0
    reported_brier = 0.0
    reference_brier = 0.0
    coverage = 0.0
    for signal in signals:
        likelihood = cast("Mapping[str, Any]", signal["likelihood_by_outcome"])
        reported = {
            outcome_id: float(
                cast("Mapping[str, Any]", signal["reported_outcome_probabilities"])[
                    outcome_id
                ]
            )
            for outcome_id in outcome_ids
        }
        probability = math.fsum(
            priors[outcome_id] * float(likelihood[outcome_id])
            for outcome_id in outcome_ids
        )
        posterior = {
            outcome_id: priors[outcome_id] * float(likelihood[outcome_id]) / probability
            for outcome_id in outcome_ids
        }
        actual_action_values = {
            action_id: math.fsum(
                posterior[outcome_id] * values[outcome_id] for outcome_id in outcome_ids
            )
            for action_id, values in payoffs.items()
        }
        reported_action_values = {
            action_id: math.fsum(
                reported[outcome_id] * values[outcome_id] for outcome_id in outcome_ids
            )
            for action_id, values in payoffs.items()
        }
        oracle_choice = _choice(actual_action_values, absolute, relative)
        timely_deployed_choice = _choice(reported_action_values, absolute, relative)
        deployed_choice = (
            timely_deployed_choice if operationally_usable else baseline_choice
        )
        oracle_value = _policy_value(oracle_choice, actual_action_values)
        timely_deployed_value = _policy_value(
            timely_deployed_choice, actual_action_values
        )
        deployed_value = _policy_value(deployed_choice, actual_action_values)
        calibration_l1 = math.fsum(
            abs(reported[outcome_id] - posterior[outcome_id])
            for outcome_id in outcome_ids
        )
        partitions.append(
            {
                "signal_id": signal["signal_id"],
                "probability": probability,
                "posterior_outcomes": posterior,
                "reported_outcomes": reported,
                "oracle_choice_tie": oracle_choice,
                "deployed_choice_tie": deployed_choice,
                "oracle_conditional_value": oracle_value,
                "deployed_conditional_value": deployed_value,
                "value_contribution": probability * (deployed_value - baseline_value),
                "calibration_l1": calibration_l1,
            }
        )
        timely_oracle_resolved += probability * oracle_value
        timely_deployed_resolved += probability * timely_deployed_value
        operational_resolved += probability * deployed_value
        weighted_calibration += probability * calibration_l1
        coverage += probability
        for outcome_id in outcome_ids:
            joint_probability = priors[outcome_id] * float(likelihood[outcome_id])
            reported_brier += joint_probability * math.fsum(
                (reported[candidate] - float(candidate == outcome_id)) ** 2
                for candidate in outcome_ids
            )
            reference_brier += joint_probability * math.fsum(
                (posterior[candidate] - float(candidate == outcome_id)) ** 2
                for candidate in outcome_ids
            )

    oracle_voi = timely_oracle_resolved - baseline_value
    timely_deployed_voi = timely_deployed_resolved - baseline_value
    gross_deployed = operational_resolved - baseline_value
    calibration_loss = oracle_voi - timely_deployed_voi
    if oracle_voi < -absolute or calibration_loss < -absolute:
        raise ArithmeticError("signal refinement assurance failed")
    oracle_voi = max(0.0, oracle_voi)
    calibration_loss = max(0.0, calibration_loss)
    cost = float(cast("Mapping[str, Any]", payload["signal_cost"])["amount"])
    perfect_value = math.fsum(
        priors[outcome_id] * max(values[outcome_id] for values in payoffs.values())
        for outcome_id in outcome_ids
    )
    baseline_regret = perfect_value - baseline_value
    deployed_regret = perfect_value - operational_resolved
    objective = cast("Mapping[str, Any]", payload["objective"])
    return {
        "schema_version": "1.0.0",
        "analysis_id": payload["analysis_id"],
        "analysis_type": "forecast_signal_information_result",
        "method_maturity": "experimental",
        "value_unit": objective["value_unit"],
        "timing": {
            "time_unit": timing["time_unit"],
            "horizon": outcome_time - origin,
            "freshness": freshness,
            "latency": available - origin,
            "lead_time": outcome_time - decision,
            "operationally_usable": operationally_usable,
            "reason": timing_reason,
        },
        "baseline": {
            "expected_action_values": baseline_values,
            "choice_tie": baseline_choice,
            "value": baseline_value,
        },
        "signal_partitions": partitions,
        "value": {
            "counterfactual_timely_oracle": oracle_voi,
            "gross_deployed": gross_deployed,
            "calibration_loss": calibration_loss,
            "cost": cost,
            "net_deployed": gross_deployed - cost,
            "maximum_price": max(0.0, gross_deployed),
        },
        "regret": {
            "baseline_expected": baseline_regret,
            "deployed_expected": deployed_regret,
            "avoided": baseline_regret - deployed_regret,
        },
        "diagnostics": {
            "weighted_calibration_l1": weighted_calibration,
            "reported_brier": reported_brier,
            "posterior_reference_brier": reference_brier,
            "excess_brier": max(0.0, reported_brier - reference_brier),
            "signal_probability_coverage": coverage,
        },
        "assurance": {
            "estimator": "exact_finite_enumeration",
            "joint_law": "outcome_prior_times_signal_likelihood",
            "decision_policy": "reported_probabilities_evaluated_under_joint_law",
            "accuracy_is_value": False,
            "training_performed": False,
        },
        "language_dispositions": {
            "python": "executable",
            "rust": "unsupported",
            "r": "unsupported",
            "julia": "unsupported",
            "mojo": "external",
        },
        "unsupported_dispositions": [
            "forecast model training or tuning",
            "continuous signal integration",
            "multistage recourse",
            "endogenous temporal dependence estimation",
            "stable or polyglot execution",
        ],
    }
