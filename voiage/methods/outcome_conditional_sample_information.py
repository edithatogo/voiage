"""Exact finite outcome-conditional value of sample information."""

# pyright: reportAny=false, reportExplicitAny=false, reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from itertools import pairwise
import json
import math
from typing import Any, cast

from voiage.contracts.outcome_conditional_sample_information import (
    validate_outcome_conditional_sample_information_result_semantics,
    validate_outcome_conditional_sample_information_semantics,
)
from voiage.exceptions import raise_input_error

_CHRONOLOGY = ["prior", "measure", "observe", "update", "act"]


@dataclass(frozen=True)
class OutcomeConditionalSampleInformationResult:
    """Portable exact finite outcome-conditional sample-information result."""

    payload: dict[str, Any]

    def to_contract_dict(self) -> dict[str, Any]:
        """Return an independent JSON-compatible result copy."""
        return cast("dict[str, Any]", json.loads(json.dumps(self.payload)))


@dataclass(frozen=True)
class _Model:
    payload: dict[str, Any]
    actions: tuple[str, ...]
    states: tuple[str, ...]
    outcomes: tuple[str, ...]
    prior: dict[str, float]
    action_values: dict[str, dict[str, float]]
    likelihood: dict[str, dict[str, float]]
    sign: float
    tie_tolerance: float
    probability_tolerance: float
    reference_action: str
    information_cost: float
    thresholds: tuple[float, ...]
    quantile_levels: tuple[float, ...]


def outcome_conditional_sample_information_value(
    specification: dict[str, object],
) -> OutcomeConditionalSampleInformationResult:
    """Evaluate exact finite outcome-conditional sample-information value.

    Parameters
    ----------
    specification:
        Strict v1 state, action and measurement-outcome contract. Utilities
        are maximized and losses are minimized. The declared reference action
        must be an exact baseline optimizer.

    Returns
    -------
    OutcomeConditionalSampleInformationResult
        Predictive outcome distribution, conditional metrics, weighted
        population dispersion, low-value risks and reconstructable assurance.
    """
    try:
        payload = cast(
            "dict[str, Any]",
            json.loads(json.dumps(specification, ensure_ascii=False)),
        )
        validate_outcome_conditional_sample_information_semantics(payload)
        result = _evaluate_contract(_validate_and_build(payload))
        validate_outcome_conditional_sample_information_result_semantics(result)
    except (ArithmeticError, KeyError, TypeError, ValueError) as error:
        raise_input_error(str(error))
    return OutcomeConditionalSampleInformationResult(result)


def _contract_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be an object")
    return cast("dict[str, Any]", value)


def _records(value: object, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise TypeError(f"{label} must be a non-empty array")
    return [_mapping(record, f"{label} entry") for record in value]


def _clean(value: float, tolerance: float) -> float:
    return 0.0 if abs(value) <= tolerance else value


def _require_nonnegative_vsi(vsi: float) -> None:
    if vsi < 0.0:
        raise ValueError("VSI must be nonnegative under the matched model")


def _assert_assurance(
    tower_residual: float, threshold_monotonic: bool, probability_tolerance: float
) -> None:
    if abs(tower_residual) > max(probability_tolerance, 1e-12):
        raise ValueError("expectation-only EVSI tower identities failed")
    if not threshold_monotonic:
        raise ValueError("low-value risk must be monotone in delta")


def _ties(values: dict[str, float], sign: float, tolerance: float) -> list[str]:
    best_score = max(sign * value for value in values.values())
    return sorted(
        action
        for action, value in values.items()
        if math.isclose(
            sign * value,
            best_score,
            abs_tol=tolerance,
            rel_tol=0.0,
        )
    )


def _best_value(values: dict[str, float], sign: float) -> float:
    return max(values.values()) if sign > 0.0 else min(values.values())


def _validate_and_build(payload: dict[str, Any]) -> _Model:
    objective = _mapping(payload["objective"], "objective")
    measure = objective["measure"]
    direction = objective["direction"]
    if (measure, direction) not in {
        ("utility", "maximize"),
        ("loss", "minimize"),
    }:
        raise ValueError("utility must be maximized and loss must be minimized")
    if payload["chronology"] != _CHRONOLOGY:
        raise ValueError("chronology must be prior-measure-observe-update-act")
    sign = 1.0 if direction == "maximize" else -1.0

    actions_raw = payload["actions"]
    if not isinstance(actions_raw, list):
        raise TypeError("actions must be an array")
    actions = tuple(sorted(_identifier(item, "action") for item in actions_raw))
    if len(actions) < 2 or len(set(actions)) != len(actions):
        raise ValueError("actions must contain at least two unique identifiers")

    state_records = _records(payload["states"], "states")
    states = tuple(
        sorted(_identifier(record["state_id"], "state_id") for record in state_records)
    )
    if len(set(states)) != len(states):
        raise ValueError("state identifiers must be unique")
    state_by_id = {cast("str", record["state_id"]): record for record in state_records}
    prior = {
        state: _finite(state_by_id[state]["probability"], f"prior.{state}")
        for state in states
    }
    if any(probability <= 0.0 or probability > 1.0 for probability in prior.values()):
        raise ValueError("state probabilities must lie in (0, 1]")

    probability_tolerance = _finite(
        payload["probability_tolerance"], "probability_tolerance"
    )
    if not 0.0 < probability_tolerance <= 1e-6:
        raise ValueError("probability_tolerance must lie in (0, 1e-6]")
    if not math.isclose(
        math.fsum(prior.values()),
        1.0,
        abs_tol=probability_tolerance,
        rel_tol=0.0,
    ):
        raise ValueError("state probabilities must sum to one")

    action_values: dict[str, dict[str, float]] = {}
    for state in states:
        raw_values = _mapping(state_by_id[state]["action_values"], "action_values")
        if set(raw_values) != set(actions):
            raise ValueError("every state must contain exactly the declared actions")
        action_values[state] = {
            action: _finite(raw_values[action], f"action_values.{state}.{action}")
            for action in actions
        }

    outcome_records = _records(payload["outcomes"], "outcomes")
    outcomes = tuple(
        sorted(
            _identifier(record["outcome_id"], "outcome_id")
            for record in outcome_records
        )
    )
    if len(set(outcomes)) != len(outcomes):
        raise ValueError("outcome identifiers must be unique")
    outcome_by_id = {
        cast("str", record["outcome_id"]): record for record in outcome_records
    }
    likelihood: dict[str, dict[str, float]] = {}
    for outcome in outcomes:
        raw = _mapping(
            outcome_by_id[outcome]["likelihood_by_state"],
            "likelihood_by_state",
        )
        if set(raw) != set(states):
            raise ValueError("every outcome likelihood must contain exactly all states")
        likelihood[outcome] = {
            state: _finite(raw[state], f"likelihood.{outcome}.{state}")
            for state in states
        }
        if any(value < 0.0 or value > 1.0 for value in likelihood[outcome].values()):
            raise ValueError("likelihood probabilities must lie in [0, 1]")
    for state in states:
        row_sum = math.fsum(likelihood[outcome][state] for outcome in outcomes)
        if not math.isclose(row_sum, 1.0, abs_tol=probability_tolerance, rel_tol=0.0):
            raise ValueError("likelihood probabilities must sum to one by state")
    for outcome in outcomes:
        predictive = math.fsum(
            prior[state] * likelihood[outcome][state] for state in states
        )
        if predictive <= probability_tolerance:
            raise ValueError(
                "every declared outcome must have positive predictive mass"
            )

    tie_tolerance = _finite(payload["tie_tolerance"], "tie_tolerance")
    if not 0.0 <= tie_tolerance <= 1e-6:
        raise ValueError("tie_tolerance must lie in [0, 1e-6]")
    baseline_values = {
        action: math.fsum(
            prior[state] * action_values[state][action] for state in states
        )
        for action in actions
    }
    reference_action = _identifier(payload["reference_action"], "reference_action")
    if reference_action not in actions:
        raise ValueError("reference_action must be a declared action")
    true_best = _best_value(baseline_values, sign)
    if not math.isclose(
        baseline_values[reference_action],
        true_best,
        abs_tol=1e-12,
        rel_tol=0.0,
    ):
        raise ValueError("reference_action must be exactly baseline optimal")

    information_cost = _finite(payload["information_cost"], "information_cost")
    if information_cost < 0.0:
        raise ValueError("information_cost must be nonnegative")
    if payload["cost_placement"] != "subtract_after_gross_vsi":
        raise ValueError("cost_placement must preserve the gross VSI distribution")

    thresholds_raw = payload["low_value_thresholds"]
    quantiles_raw = payload["quantile_levels"]
    if not isinstance(thresholds_raw, list) or not thresholds_raw:
        raise TypeError("low_value_thresholds must be a non-empty array")
    if not isinstance(quantiles_raw, list) or not quantiles_raw:
        raise TypeError("quantile_levels must be a non-empty array")
    thresholds = tuple(
        sorted(_finite(value, "low_value_threshold") for value in thresholds_raw)
    )
    quantile_levels = tuple(
        sorted(_finite(value, "quantile_level") for value in quantiles_raw)
    )
    if any(value < 0.0 for value in thresholds) or len(set(thresholds)) != len(
        thresholds
    ):
        raise ValueError("low-value thresholds must be unique and nonnegative")
    if any(value < 0.0 or value > 1.0 for value in quantile_levels) or len(
        set(quantile_levels)
    ) != len(quantile_levels):
        raise ValueError("quantile levels must be unique probabilities")

    scope = _mapping(payload["scope"], "scope")
    observed = scope["observed_outcome_id"]
    if scope["mode"] == "prospective" and observed is not None:
        raise ValueError("prospective scope cannot declare an observed outcome")
    if scope["mode"] == "retrospective" and observed not in outcomes:
        raise ValueError("retrospective scope requires a declared observed outcome")

    return _Model(
        payload=payload,
        actions=actions,
        states=states,
        outcomes=outcomes,
        prior=prior,
        action_values=action_values,
        likelihood=likelihood,
        sign=sign,
        tie_tolerance=tie_tolerance,
        probability_tolerance=probability_tolerance,
        reference_action=reference_action,
        information_cost=information_cost,
        thresholds=thresholds,
        quantile_levels=quantile_levels,
    )


def _weighted_quantile(rows: list[dict[str, Any]], level: float) -> float:
    ordered = sorted(
        (
            (cast("float", row["vsi"]), cast("float", row["predictive_probability"]))
            for row in rows
        ),
        key=lambda item: item[0],
    )
    if level <= 0.0:
        return ordered[0][0]
    cumulative = 0.0
    for value, probability in ordered:
        cumulative += probability
        if cumulative + 1e-15 >= level:
            return value
    return ordered[-1][0]


def _lower_tail_mean(rows: list[dict[str, Any]], mass: float) -> float:
    ordered = sorted(
        (
            (cast("float", row["vsi"]), cast("float", row["predictive_probability"]))
            for row in rows
        ),
        key=lambda item: item[0],
    )
    if mass <= 0.0:
        return ordered[0][0]
    remaining = mass
    total = 0.0
    for value, probability in ordered:
        take = min(probability, remaining)
        total += take * value
        remaining -= take
        if remaining <= 1e-15:
            break
    return total / mass


def _evaluate_contract(model: _Model) -> dict[str, Any]:
    baseline_values = {
        action: math.fsum(
            model.prior[state] * model.action_values[state][action]
            for state in model.states
        )
        for action in model.actions
    }
    baseline_ties = _ties(baseline_values, model.sign, model.tie_tolerance)
    baseline_best = _best_value(baseline_values, model.sign)
    outcome_rows: list[dict[str, Any]] = []
    likelihood_residuals = [
        abs(
            math.fsum(model.likelihood[outcome][state] for outcome in model.outcomes)
            - 1.0
        )
        for state in model.states
    ]
    for outcome in model.outcomes:
        predictive = math.fsum(
            model.prior[state] * model.likelihood[outcome][state]
            for state in model.states
        )
        posterior = {
            state: model.prior[state] * model.likelihood[outcome][state] / predictive
            for state in model.states
        }
        posterior_values = {
            action: math.fsum(
                posterior[state] * model.action_values[state][action]
                for state in model.states
            )
            for action in model.actions
        }
        optimal_actions = _ties(posterior_values, model.sign, model.tie_tolerance)
        posterior_best = _best_value(posterior_values, model.sign)
        reference_value = posterior_values[model.reference_action]
        delta_ev = _clean(
            model.sign * (posterior_best - baseline_best),
            model.probability_tolerance,
        )
        vsi = _clean(
            model.sign * (posterior_best - reference_value),
            model.probability_tolerance,
        )
        _require_nonnegative_vsi(vsi)
        posterior_ties = set(optimal_actions)
        baseline_tie_set = set(baseline_ties)
        outcome_rows.append(
            {
                "outcome_id": outcome,
                "predictive_probability": predictive,
                "posterior_state_probabilities": posterior,
                "posterior_action_values": posterior_values,
                "optimal_actions": optimal_actions,
                "posterior_optimal_value": posterior_best,
                "reference_action_value": reference_value,
                "delta_ev": delta_ev,
                "vsi": vsi,
                "net_vsi": vsi - model.information_cost,
                "reference_action_excluded": model.reference_action
                not in posterior_ties,
                "mandatory_policy_switch": not bool(baseline_tie_set & posterior_ties),
                "complete_tie_set_changed": baseline_tie_set != posterior_ties,
            }
        )
    predictive_total = math.fsum(
        cast("float", row["predictive_probability"]) for row in outcome_rows
    )
    evsi = math.fsum(
        cast("float", row["predictive_probability"]) * cast("float", row["vsi"])
        for row in outcome_rows
    )
    expected_delta = math.fsum(
        cast("float", row["predictive_probability"]) * cast("float", row["delta_ev"])
        for row in outcome_rows
    )
    variance_vsi = math.fsum(
        cast("float", row["predictive_probability"])
        * (cast("float", row["vsi"]) - evsi) ** 2
        for row in outcome_rows
    )
    variance_vsi = max(0.0, _clean(variance_vsi, 1e-15))
    low_value_risks = [
        {
            "delta": delta,
            "probability": math.fsum(
                cast("float", row["predictive_probability"])
                for row in outcome_rows
                if cast("float", row["vsi"]) <= delta
            ),
        }
        for delta in model.thresholds
    ]
    probabilities = [row["probability"] for row in low_value_risks]
    threshold_monotonic = all(
        left <= right + model.probability_tolerance
        for left, right in pairwise(probabilities)
    )
    bayes_reconstructed = {
        state: math.fsum(
            cast("float", row["predictive_probability"])
            * cast("dict[str, float]", row["posterior_state_probabilities"])[state]
            for row in outcome_rows
        )
        for state in model.states
    }
    maximum_bayes_error = max(
        abs(bayes_reconstructed[state] - model.prior[state]) for state in model.states
    )

    aggregate = {
        "evsi": evsi,
        "expected_delta_ev": expected_delta,
        "information_cost": model.information_cost,
        "net_evsi": evsi - model.information_cost,
        "variance_vsi": variance_vsi,
        "sigma_vsi": math.sqrt(variance_vsi),
        "minimum_vsi": min(cast("float", row["vsi"]) for row in outcome_rows),
        "maximum_vsi": max(cast("float", row["vsi"]) for row in outcome_rows),
        "low_value_risks": low_value_risks,
        "weighted_quantiles": [
            {"level": level, "vsi": _weighted_quantile(outcome_rows, level)}
            for level in model.quantile_levels
        ],
        "lower_tail_means": [
            {"mass": level, "mean_vsi": _lower_tail_mean(outcome_rows, level)}
            for level in model.quantile_levels
        ],
        "reference_action_excluded_probability": math.fsum(
            cast("float", row["predictive_probability"])
            for row in outcome_rows
            if row["reference_action_excluded"]
        ),
        "mandatory_policy_switch_probability": math.fsum(
            cast("float", row["predictive_probability"])
            for row in outcome_rows
            if row["mandatory_policy_switch"]
        ),
        "complete_tie_set_changed_probability": math.fsum(
            cast("float", row["predictive_probability"])
            for row in outcome_rows
            if row["complete_tie_set_changed"]
        ),
    }
    scope = cast("dict[str, Any]", model.payload["scope"])
    observed = scope["observed_outcome_id"]
    retrospective = next(
        (row for row in outcome_rows if row["outcome_id"] == observed), None
    )
    tower_residual = _clean(evsi - expected_delta, model.probability_tolerance)
    result = {
        "schema_version": "v1",
        "analysis_id": model.payload["analysis_id"],
        "analysis_type": "outcome_conditional_sample_information_value_result",
        "method_maturity": "experimental",
        "value_unit": model.payload["value_unit"],
        "population": model.payload["population"],
        "horizon": model.payload["horizon"],
        "discount_basis": model.payload["discount_basis"],
        "objective": model.payload["objective"],
        "chronology": _CHRONOLOGY,
        "scope": scope,
        "baseline": {
            "action_expected_values": baseline_values,
            "optimal_actions": baseline_ties,
            "reference_action": model.reference_action,
            "reference_expected_value": baseline_values[model.reference_action],
        },
        "outcomes": outcome_rows,
        "aggregate": aggregate,
        "retrospective_outcome": retrospective,
        "assurance": {
            "estimator": "exact_finite_enumeration",
            "standard_deviation_functional": "predictive_probability_weighted_population",
            "ddof": 0,
            "tower_identity_scope": "expectations_only",
            "evsi_vsi_residual": 0.0,
            "evsi_delta_ev_residual": tower_residual,
            "predictive_probability_residual": _clean(
                predictive_total - 1.0, model.probability_tolerance
            ),
            "maximum_likelihood_row_residual": max(likelihood_residuals),
            "maximum_bayes_reconstruction_error": maximum_bayes_error,
            "minimum_vsi": aggregate["minimum_vsi"],
            "threshold_monotonic": threshold_monotonic,
            "r_vsi_zero_distinct_from_switch_metrics": True,
            "continuous_outcomes_supported": False,
            "result_reconstructed_from_input": True,
        },
        "input_assurance": {
            "input_sha256": _contract_sha256(model.payload),
            "input_contract": model.payload,
        },
        "provenance": model.payload["provenance"],
        "references": [
            "https://arxiv.org/abs/2309.09452",
            "https://doi.org/10.1016/j.ecolind.2024.111828",
        ],
        "language_dispositions": {
            "python": "experimental_executable",
            "rust": "unsupported",
            "r": "unsupported",
            "julia": "unsupported",
            "mojo": "external",
        },
    }
    _assert_assurance(tower_residual, threshold_monotonic, model.probability_tolerance)
    return result
