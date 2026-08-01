"""Exact finite information-and-implementation value decomposition.

This experimental module evaluates the four cells formed by current/perfect
information and current/perfect implementation.  Implementation is represented
as a state- and intended-action-specific distribution over realised actions;
it is therefore not assumed independent of the uncertain state or policy.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any, cast

from voiage.exceptions import raise_input_error


@dataclass(frozen=True)
class ImplementationInformationResult:
    """Portable result envelope for the experimental finite evaluator."""

    payload: dict[str, Any]

    def to_contract_dict(self) -> dict[str, Any]:
        """Return an independent JSON-compatible copy."""
        return cast("dict[str, Any]", json.loads(json.dumps(self.payload)))


def _probabilities(
    record: object, actions: list[str], context: str
) -> dict[str, float]:
    if not isinstance(record, dict) or set(record) != set(actions):
        raise ValueError(f"{context} must contain exactly the declared actions")
    probability_record = cast("dict[str, Any]", record)
    values = {action: float(probability_record[action]) for action in actions}
    if any(not math.isfinite(value) or value < 0.0 for value in values.values()):
        raise ValueError(f"{context} probabilities must be finite and non-negative")
    if not math.isclose(math.fsum(values.values()), 1.0, abs_tol=1e-12):
        raise ValueError(f"{context} probabilities must sum to one")
    return values


def _scenario(
    record: object, states: list[str], actions: list[str], context: str
) -> dict[str, dict[str, dict[str, float]]]:
    if not isinstance(record, dict) or set(record) != set(states):
        raise ValueError(f"{context} must contain exactly the declared states")
    scenario_record = cast("dict[str, Any]", record)
    result: dict[str, dict[str, dict[str, float]]] = {}
    for state in states:
        state_record = scenario_record[state]
        if not isinstance(state_record, dict) or set(state_record) != set(actions):
            raise ValueError(
                f"{context}.{state} must contain exactly the intended actions"
            )
        intended_records = cast("dict[str, Any]", state_record)
        result[state] = {
            intended: _probabilities(
                intended_records[intended], actions, f"{context}.{state}.{intended}"
            )
            for intended in actions
        }
    return result


def _ties(values: dict[str, float], tolerance: float) -> list[str]:
    maximum = max(values.values())
    return sorted(
        action
        for action, value in values.items()
        if math.isclose(value, maximum, abs_tol=tolerance, rel_tol=0.0)
    )


def _realised_value(
    state: str,
    intended: str,
    scenario: dict[str, dict[str, dict[str, float]]],
    net_benefit: dict[str, dict[str, float]],
) -> float:
    return math.fsum(
        probability * net_benefit[state][realised]
        for realised, probability in scenario[state][intended].items()
    )


def _cell(
    probabilities: dict[str, float],
    actions: list[str],
    scenario: dict[str, dict[str, dict[str, float]]],
    net_benefit: dict[str, dict[str, float]],
    perfect_information: bool,
    tolerance: float,
) -> tuple[float, dict[str, list[str]], dict[str, dict[str, float]]]:
    state_values = {
        state: {
            action: _realised_value(state, action, scenario, net_benefit)
            for action in actions
        }
        for state in probabilities
    }
    if perfect_information:
        policies = {
            state: _ties(values, tolerance) for state, values in state_values.items()
        }
        value = math.fsum(
            probabilities[state] * max(state_values[state].values())
            for state in probabilities
        )
        return value, policies, state_values
    expected = {
        action: math.fsum(
            probabilities[state] * state_values[state][action]
            for state in probabilities
        )
        for action in actions
    }
    return max(expected.values()), {"all": _ties(expected, tolerance)}, state_values


def implementation_information_value(
    specification: dict[str, object],
) -> ImplementationInformationResult:
    """Evaluate a finite joint information/implementation decomposition.

    The evaluator returns the current/perfect-information by current/perfect-
    implementation matrix, optional specific-implementation cells, and an
    implementation-adjusted sample-information cell.  All values are exact
    finite enumerations of the supplied discrete specification.
    """
    try:
        payload = cast("dict[str, Any]", json.loads(json.dumps(specification)))
        result = _evaluate(payload)
    except (ArithmeticError, KeyError, TypeError, ValueError) as error:
        raise_input_error(str(error))
    return ImplementationInformationResult(result)


def _evaluate(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("schema_version") != "v1":
        raise ValueError("schema_version must be 'v1'")
    if payload.get("analysis_type") != "implementation_information_decomposition":
        raise ValueError(
            "analysis_type is not implementation_information_decomposition"
        )
    actions = list(payload["actions"])
    if len(actions) < 2 or len(set(actions)) != len(actions):
        raise ValueError("actions must contain at least two unique identifiers")
    if any(not isinstance(action, str) or not action for action in actions):
        raise ValueError("action identifiers must be non-empty strings")

    states_input = payload["states"]
    if not isinstance(states_input, list) or not states_input:
        raise ValueError("states must be a non-empty array")
    states_input = cast("list[dict[str, Any]]", states_input)
    state_ids = [str(state["state_id"]) for state in states_input]
    if len(set(state_ids)) != len(state_ids):
        raise ValueError("state identifiers must be unique")
    probabilities = {
        state_ids[i]: float(state["probability"])
        for i, state in enumerate(states_input)
    }
    if any(not math.isfinite(value) or value < 0.0 for value in probabilities.values()):
        raise ValueError("state probabilities must be finite and non-negative")
    if not math.isclose(math.fsum(probabilities.values()), 1.0, abs_tol=1e-12):
        raise ValueError("state probabilities must sum to one")
    net_benefit: dict[str, dict[str, float]] = {}
    for state, state_id in zip(states_input, state_ids, strict=True):
        values = state["net_benefit"]
        if not isinstance(values, dict) or set(values) != set(actions):
            raise ValueError(f"net_benefit for {state_id} must contain all actions")
        net_benefit[state_id] = {action: float(values[action]) for action in actions}
        if any(not math.isfinite(value) for value in net_benefit[state_id].values()):
            raise ValueError("net benefits must be finite")

    population = float(payload["population"])
    time_factor = float(payload["discounted_time_factor"])
    if not math.isfinite(population) or population <= 0.0:
        raise ValueError("population must be positive and finite")
    if not math.isfinite(time_factor) or time_factor <= 0.0:
        raise ValueError("discounted_time_factor must be positive and finite")
    scale = population * time_factor
    tolerance = float(payload.get("tie_tolerance", 1e-12))
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tie_tolerance must be finite and non-negative")

    current = _scenario(
        payload["current_implementation"], state_ids, actions, "current_implementation"
    )
    perfect = {
        state: {
            intended: {realised: float(realised == intended) for realised in actions}
            for intended in actions
        }
        for state in state_ids
    }
    current_current = _cell(
        probabilities, actions, current, net_benefit, False, tolerance
    )
    perfect_current = _cell(
        probabilities, actions, current, net_benefit, True, tolerance
    )
    current_perfect = _cell(
        probabilities, actions, perfect, net_benefit, False, tolerance
    )
    perfect_perfect = _cell(
        probabilities, actions, perfect, net_benefit, True, tolerance
    )

    cells: dict[
        str, tuple[float, dict[str, list[str]], dict[str, dict[str, float]]]
    ] = {
        "current_information_current_implementation": current_current,
        "perfect_information_current_implementation": perfect_current,
        "current_information_perfect_implementation": current_perfect,
        "perfect_information_perfect_implementation": perfect_perfect,
    }
    specific = payload.get("specific_implementation")
    if specific is not None:
        specific_scenario = _scenario(
            specific, state_ids, actions, "specific_implementation"
        )
        cells["current_information_specific_implementation"] = _cell(
            probabilities, actions, specific_scenario, net_benefit, False, tolerance
        )
        cells["perfect_information_specific_implementation"] = _cell(
            probabilities, actions, specific_scenario, net_benefit, True, tolerance
        )

    sample_value: float | None = None
    sample_policies: dict[str, list[str]] = {}
    sampling = payload.get("sampling_model")
    if sampling is not None:
        if not isinstance(sampling, dict):
            raise ValueError("sampling_model must be an object")
        signals = sampling["signals"]
        if not isinstance(signals, list) or not signals:
            raise ValueError("sampling_model.signals must be non-empty")
        signal_ids = [str(signal["signal_id"]) for signal in signals]
        if len(set(signal_ids)) != len(signal_ids):
            raise ValueError("signal identifiers must be unique")
        likelihoods: dict[str, dict[str, float]] = {state: {} for state in state_ids}
        for signal in signals:
            signal_id = str(signal["signal_id"])
            by_state = signal["likelihood_by_state"]
            if not isinstance(by_state, dict) or set(by_state) != set(state_ids):
                raise ValueError("each signal must declare every state likelihood")
            for state in state_ids:
                likelihoods[state][signal_id] = float(by_state[state])
        for state in state_ids:
            _probabilities(likelihoods[state], signal_ids, f"signals_given_{state}")
        post = sampling["post_sample_implementation"]
        if not isinstance(post, dict) or set(post) != set(signal_ids):
            raise ValueError("post_sample_implementation must contain every signal")
        sample_value = 0.0
        for signal_id in signal_ids:
            scenario = _scenario(
                post[signal_id],
                state_ids,
                actions,
                f"post_sample_implementation.{signal_id}",
            )
            joint_values = {
                action: math.fsum(
                    probabilities[state]
                    * likelihoods[state][signal_id]
                    * _realised_value(state, action, scenario, net_benefit)
                    for state in state_ids
                )
                for action in actions
            }
            sample_policies[signal_id] = _ties(joint_values, tolerance)
            sample_value += max(joint_values.values())

    c00, c10, c01, c11 = (
        current_current[0],
        perfect_current[0],
        current_perfect[0],
        perfect_perfect[0],
    )
    interaction = c11 - c10 - c01 + c00
    costs = payload.get("costs", {})
    if not isinstance(costs, dict):
        raise TypeError("costs must be an object")
    cost_values = {
        key: float(costs.get(key, 0.0))
        for key in (
            "perfect_information",
            "perfect_implementation",
            "specific_implementation",
            "sample",
        )
    }
    if any(not math.isfinite(value) or value < 0.0 for value in cost_values.values()):
        raise ValueError("costs must be finite and non-negative")

    gross = {
        "realizable_evpi": (c10 - c00) * scale,
        "evpim": (c01 - c00) * scale,
        "evp": (c11 - c00) * scale,
        "evpi_under_perfect_implementation": (c11 - c01) * scale,
        "interaction": interaction * scale,
    }
    if specific is not None:
        gross["evsim"] = (
            cells["current_information_specific_implementation"][0] - c00
        ) * scale
    if sample_value is not None:
        gross["ia_evsi"] = (sample_value - c00) * scale
    net = {
        "realizable_evpi": gross["realizable_evpi"]
        - cost_values["perfect_information"],
        "evpim": gross["evpim"] - cost_values["perfect_implementation"],
        "evp": gross["evp"]
        - cost_values["perfect_information"]
        - cost_values["perfect_implementation"],
    }
    if "evsim" in gross:
        net["evsim"] = gross["evsim"] - cost_values["specific_implementation"]
    if "ia_evsi" in gross:
        net["ia_evsi"] = gross["ia_evsi"] - cost_values["sample"]

    rendered_cells = {
        name: {
            "per_person_time_value": cell[0],
            "aggregate_value": cell[0] * scale,
            "policy_ties": cell[1],
        }
        for name, cell in cells.items()
    }
    if sample_value is not None:
        rendered_cells["sample_information_post_sample_implementation"] = {
            "per_person_time_value": sample_value,
            "aggregate_value": sample_value * scale,
            "policy_ties": sample_policies,
        }

    state_dependent = any(
        current[state] != current[state_ids[0]] for state in state_ids[1:]
    )
    return {
        "schema_version": "v1",
        "analysis_id": payload["analysis_id"],
        "analysis_type": "implementation_information_decomposition",
        "method_maturity": "experimental",
        "value_unit": payload["value_unit"],
        "population": population,
        "discounted_time_factor": time_factor,
        "chronology": payload["chronology"],
        "matrix": rendered_cells,
        "gross_components": gross,
        "net_components": net,
        "identity_residuals": {
            "evp_equals_realizable_evpi_plus_evpim_plus_interaction": (
                gross["evp"]
                - gross["realizable_evpi"]
                - gross["evpim"]
                - gross["interaction"]
            ),
            "perfect_implementation_evpi_equals_evp_minus_evpim": (
                gross["evpi_under_perfect_implementation"]
                - gross["evp"]
                + gross["evpim"]
            ),
        },
        "decision_switches": {
            "current_to_perfect_information": current_current[1] != perfect_current[1],
            "current_to_perfect_implementation": current_current[1]
            != current_perfect[1],
            "sample_information": sample_policies if sample_value is not None else None,
        },
        "terminology": {
            "EVEIm": "review_candidate_for_expected_value_of_eliminating_implementation_imperfection",
            "EVSEIm": "review_candidate_for_expected_value_of_a_specific_implementation_intervention",
            "status": "presentation_alias_candidates_not_new_estimands",
        },
        "assurance": {
            "estimator": "exact_finite_enumeration",
            "tie_policy": "complete_ties_sorted_by_action_id",
            "tie_tolerance": tolerance,
            "implementation_information_independence_assumed": False,
            "state_dependent_current_implementation_observed": state_dependent,
            "signal_dependent_post_sample_implementation_supported": sampling
            is not None,
            "language_dispositions": {
                "Python": "experimental_runtime",
                "Rust": "not_implemented",
                "R": "not_implemented",
                "Julia": "not_implemented",
                "Mojo": "not_implemented",
            },
        },
    }


__all__ = ["ImplementationInformationResult", "implementation_information_value"]
