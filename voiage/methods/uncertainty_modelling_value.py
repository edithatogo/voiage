"""Exact finite EVIU, VSS and stochastic-program diagnostics."""

# pyright: reportAny=false, reportExplicitAny=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import TYPE_CHECKING, Any, cast

from voiage.contracts.uncertainty_modelling_value import (
    validate_uncertainty_modelling_value_result,
    validate_uncertainty_modelling_value_semantics,
)
from voiage.exceptions import raise_input_error

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


@dataclass(frozen=True)
class UncertaintyModellingValueResult:
    """Portable result envelope for exact finite uncertainty-modelling value."""

    payload: dict[str, Any]

    def to_contract_dict(self) -> dict[str, Any]:
        """Return an independent JSON-compatible result copy."""
        return cast("dict[str, Any]", json.loads(json.dumps(self.payload)))


def value_of_uncertainty_modelling(
    specification: Mapping[str, object],
) -> UncertaintyModellingValueResult:
    """Evaluate EV, EEV, recourse, wait-and-see, VSS/EVIU and EVPI exactly."""
    try:
        payload = cast("dict[str, Any]", json.loads(json.dumps(specification)))
        validate_uncertainty_modelling_value_semantics(payload)
        result = _evaluate(payload)
        validate_uncertainty_modelling_value_result(result)
    except (ArithmeticError, KeyError, TypeError, ValueError) as error:
        raise_input_error(str(error))
    return UncertaintyModellingValueResult(result)


def _ties(
    values: Mapping[str, float],
    direction: str,
    absolute: float,
    relative: float,
) -> list[str]:
    best = min(values.values()) if direction == "minimize" else max(values.values())
    return sorted(
        key
        for key, value in values.items()
        if math.isclose(value, best, abs_tol=absolute, rel_tol=relative)
    )


def _policy_outcomes(policy: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(item["state_id"]): item
        for item in cast("Sequence[Mapping[str, Any]]", policy["state_outcomes"])
    }


def _evaluate(data: dict[str, Any]) -> dict[str, Any]:
    objective = cast("dict[str, Any]", data["objective"])
    direction = str(objective["direction"])
    tie_policy = cast("dict[str, Any]", data["tie_policy"])
    absolute = float(tie_policy["absolute_tolerance"])
    relative = float(tie_policy["relative_tolerance"])
    states = cast("list[dict[str, Any]]", data["states"])
    probabilities = {
        str(state["state_id"]): float(state["probability"]) for state in states
    }
    policies = cast("list[dict[str, Any]]", data["policies"])
    policy_by_id = {str(policy["policy_id"]): policy for policy in policies}
    candidates = cast("list[dict[str, Any]]", data["deterministic_candidates"])

    point_values = {
        str(candidate["candidate_id"]): float(candidate["point_objective_value"])
        for candidate in candidates
    }
    ev_tie = _ties(point_values, direction, absolute, relative)
    selected_candidate_id = ev_tie[0]
    selected_candidate = next(
        candidate
        for candidate in candidates
        if candidate["candidate_id"] == selected_candidate_id
    )
    induced_policy_id = str(selected_candidate["induced_policy_id"])
    induced_policy = policy_by_id[induced_policy_id]
    induced_outcomes = _policy_outcomes(induced_policy)
    infeasible_states = sorted(
        state_id
        for state_id, outcome in induced_outcomes.items()
        if not bool(outcome["feasible"])
    )
    eev_value = (
        None
        if infeasible_states
        else math.fsum(
            probabilities[state_id] * float(outcome["objective_value"])
            for state_id, outcome in induced_outcomes.items()
        )
    )

    policy_audit: list[dict[str, Any]] = []
    feasible_policy_values: dict[str, float] = {}
    for policy_id in sorted(policy_by_id):
        policy = policy_by_id[policy_id]
        outcomes = _policy_outcomes(policy)
        infeasible = sorted(
            state_id
            for state_id, outcome in outcomes.items()
            if not bool(outcome["feasible"])
        )
        expected_value = (
            None
            if infeasible
            else math.fsum(
                probabilities[state_id] * float(outcome["objective_value"])
                for state_id, outcome in outcomes.items()
            )
        )
        if expected_value is not None:
            feasible_policy_values[policy_id] = expected_value
        policy_audit.append(
            {
                "policy_id": policy_id,
                "first_stage_decision": policy["first_stage_decision"],
                "history_decisions": sorted(
                    policy["history_decisions"], key=lambda item: item["history_id"]
                ),
                "expected_value": expected_value,
                "feasible_all_states": not infeasible,
                "infeasible_states": infeasible,
            }
        )
    if not feasible_policy_values:
        raise ValueError("recourse problem has no policy feasible in every state")
    rp_tie = _ties(feasible_policy_values, direction, absolute, relative)
    rp_value = feasible_policy_values[rp_tie[0]]

    ws_states: list[dict[str, Any]] = []
    for state in sorted(states, key=lambda item: item["state_id"]):
        state_id = str(state["state_id"])
        feasible = {
            policy_id: float(_policy_outcomes(policy)[state_id]["objective_value"])
            for policy_id, policy in policy_by_id.items()
            if bool(_policy_outcomes(policy)[state_id]["feasible"])
        }
        if not feasible:  # pragma: no cover - RP feasibility proves this invariant
            raise ValueError(f"wait-and-see problem infeasible in state {state_id}")
        state_tie = _ties(feasible, direction, absolute, relative)
        ws_states.append(
            {
                "state_id": state_id,
                "probability": probabilities[state_id],
                "policy_tie": state_tie,
                "selected_policy_id": state_tie[0],
                "objective_value": feasible[state_tie[0]],
            }
        )
    ws_value = math.fsum(
        item["probability"] * item["objective_value"] for item in ws_states
    )
    if direction == "minimize":
        evpi = rp_value - ws_value
        vss = None if eev_value is None else eev_value - rp_value
    else:
        evpi = ws_value - rp_value
        vss = None if eev_value is None else rp_value - eev_value
    tolerance = max(
        absolute, float(data["solver_assurance"]["objective_bound_tolerance"])
    )
    if evpi < -tolerance or (  # pragma: no cover - exact optimum identity
        vss is not None and vss < -tolerance
    ):
        raise ValueError("direction-aware stochastic-program identities are violated")
    evpi = 0.0 if math.isclose(evpi, 0.0, abs_tol=tolerance) else evpi
    if vss is not None and math.isclose(vss, 0.0, abs_tol=tolerance):
        vss = 0.0

    return {
        "schema_version": "1.0.0",
        "analysis_id": data["analysis_id"],
        "analysis_type": "uncertainty_modelling_value_result",
        "method_maturity": "experimental",
        "objective": objective,
        "point_estimate": data["point_estimate"],
        "expected_value_problem": {
            "candidate_values": dict(sorted(point_values.items())),
            "candidate_tie": ev_tie,
            "selected_candidate_id": selected_candidate_id,
            "selected_first_stage_decision": selected_candidate["first_stage_decision"],
            "point_objective_value": point_values[selected_candidate_id],
            "induced_policy_id": induced_policy_id,
        },
        "expected_result_of_ev_solution": {
            "status": "infeasible_recourse" if infeasible_states else "feasible",
            "value": eev_value,
            "infeasible_states": infeasible_states,
        },
        "recourse_problem": {
            "value": rp_value,
            "policy_tie": rp_tie,
            "selected_policy_id": rp_tie[0],
        },
        "wait_and_see": {"value": ws_value, "state_solutions": ws_states},
        "decomposition": {
            "vss": vss,
            "eviu": vss,
            "evpi": evpi,
            "eviu_comparator": "declared_point_estimate_ev_solution",
            "eviu_equals_vss_under_v1_contract": True,
            "identity_status": "not_estimable_infeasible_eev"
            if vss is None
            else "verified",
        },
        "policy_audit": policy_audit,
        "assurance": {
            **data["solver_assurance"],
            "states_evaluated": len(states),
            "policies_evaluated": len(policies),
            "deterministic_candidates_evaluated": len(candidates),
            "nonanticipativity_representation": "one_decision_per_shared_history",
            "recourse_feasibility_checked": True,
            "information_acquisition_modelled": False,
            "information_acquisition_separate_from_uncertainty_modelling": True,
            "deterministic_serialization": True,
            "objective_bound": rp_value,
            "optimality_gap": 0.0,
            "feasible_policies": len(feasible_policy_values),
            "infeasible_policies": len(policies) - len(feasible_policy_values),
        },
        "language_dispositions": {
            "python": "experimental_exact_finite_execution",
            "rust": "not_implemented",
            "r": "not_implemented",
            "julia": "not_implemented",
            "mojo": "external_upstream_boundary",
        },
        "unsupported_dispositions": {
            "dvss": "deferred_pending_separate_multistage_reference_contract",
            "vms": "deferred_pending_separate_multistage_reference_contract",
            "approximate_or_external_solvers": "not_supported_in_v1",
            "risk_criteria_beyond_expected_value": "not_supported_in_v1",
        },
        "provenance": data["provenance"],
    }
