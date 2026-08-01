"""Exact finite evaluator for risk-sensitive constrained perfect information."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import json
import math
from typing import Any, cast

from voiage.contracts.risk_sensitive_voi import (
    validate_risk_sensitive_voi_result,
    validate_risk_sensitive_voi_semantics,
)
from voiage.exceptions import raise_input_error


@dataclass(frozen=True)
class RiskSensitiveVoiResult:
    """Portable result for the experimental exact finite v1 contract."""

    payload: dict[str, Any]

    def to_contract_dict(self) -> dict[str, Any]:
        """Return an independent JSON-compatible result copy."""
        return cast("dict[str, Any]", json.loads(json.dumps(self.payload)))


def _constraint_diagnostics(
    mapping: dict[str, str],
    constraints: list[dict[str, Any]],
    policies: dict[str, dict[str, Any]],
    probabilities: dict[str, float],
) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    for constraint in constraints:
        constraint_id = cast("str", constraint["constraint_id"])
        slacks: dict[str, float] = {}
        for state_id, policy_id in mapping.items():
            usage = float(
                policies[policy_id]["constraint_usage"][constraint_id][state_id]
            )
            limit = float(constraint["limit"])
            slacks[state_id] = (
                limit - usage
                if constraint["sense"] == "less_than_or_equal"
                else usage - limit
            )
        satisfied = [state_id for state_id, slack in slacks.items() if slack >= 0]
        satisfaction_probability = math.fsum(
            probabilities[state_id] for state_id in satisfied
        )
        required = (
            1.0
            if constraint["enforcement"] == "deterministic"
            else float(constraint["minimum_satisfaction_probability"])
        )
        diagnostics.append(
            {
                "constraint_id": constraint_id,
                "kind": constraint["kind"],
                "unit": constraint["unit"],
                "enforcement": constraint["enforcement"],
                "required_satisfaction_probability": required,
                "satisfaction_probability": satisfaction_probability,
                "feasible": satisfaction_probability >= required - 1e-12,
                "worst_slack": min(slacks.values()),
                "slack_by_state": slacks,
                "violating_state_ids": sorted(set(mapping) - set(satisfied)),
            }
        )
    return diagnostics


def _objective_score(
    mapping: dict[str, str],
    objective: dict[str, Any],
    policies: dict[str, dict[str, Any]],
    probabilities: dict[str, float],
) -> tuple[float, float, float | None]:
    values = {
        state_id: float(policies[policy_id]["objective_by_state"][state_id])
        for state_id, policy_id in mapping.items()
    }
    expected = math.fsum(
        probabilities[state_id] * value for state_id, value in values.items()
    )
    kind = objective["kind"]
    regret: float | None = None
    if kind in {"expected_value", "expected_utility"}:
        return expected, expected, regret
    if kind == "minimax_regret":
        reference = cast("dict[str, float]", objective["regret_reference_by_state"])
        regret = max(
            float(reference[state_id]) - value for state_id, value in values.items()
        )
        return -regret, expected, regret
    tail_mass = 1.0 - float(objective["confidence_level"])
    remaining = tail_mass
    weighted = 0.0
    for state_id, value in sorted(values.items(), key=lambda item: (item[1], item[0])):
        mass = min(remaining, probabilities[state_id])
        weighted += mass * value
        remaining -= mass
        if remaining <= 1e-15:
            break
    return weighted / tail_mass, expected, regret


def _solve(
    payload: dict[str, Any], constraints: list[dict[str, Any]]
) -> dict[str, Any]:
    states = [cast("str", item["state_id"]) for item in payload["states"]]
    probabilities = {
        cast("str", item["state_id"]): float(item["probability"])
        for item in payload["states"]
    }
    policies = {
        cast("str", item["policy_id"]): cast("dict[str, Any]", item)
        for item in payload["policies"]
    }
    policy_ids = sorted(policies)
    objective = cast("dict[str, Any]", payload["objective"])
    tolerances = cast("dict[str, float]", payload["tolerances"])
    absolute = float(tolerances["absolute_tie"])
    relative = float(tolerances["relative_tie"])

    baseline_candidates: list[
        tuple[float, str, dict[str, str], float, float | None, list[dict[str, Any]]]
    ] = []
    for policy_id in policy_ids:
        mapping = dict.fromkeys(states, policy_id)
        diagnostics = _constraint_diagnostics(
            mapping, constraints, policies, probabilities
        )
        if all(item["feasible"] for item in diagnostics):
            score, expected, regret = _objective_score(
                mapping, objective, policies, probabilities
            )
            baseline_candidates.append(
                (score, policy_id, mapping, expected, regret, diagnostics)
            )
    if not baseline_candidates:
        raise ValueError("no feasible baseline policy under the declared constraints")
    baseline_best = max(item[0] for item in baseline_candidates)
    baseline_ties = [
        item
        for item in baseline_candidates
        if math.isclose(item[0], baseline_best, abs_tol=absolute, rel_tol=relative)
    ]
    baseline = min(baseline_ties, key=lambda item: item[1])

    mapping_candidates: list[
        tuple[
            float,
            tuple[str, ...],
            dict[str, str],
            float,
            float | None,
            list[dict[str, Any]],
        ]
    ] = []
    for choices in product(policy_ids, repeat=len(states)):
        mapping = dict(zip(states, choices, strict=True))
        diagnostics = _constraint_diagnostics(
            mapping, constraints, policies, probabilities
        )
        if all(item["feasible"] for item in diagnostics):
            score, expected, regret = _objective_score(
                mapping, objective, policies, probabilities
            )
            mapping_candidates.append(
                (score, choices, mapping, expected, regret, diagnostics)
            )
    if not mapping_candidates:
        raise ValueError("no feasible post-information policy mapping")
    mapping_best = max(item[0] for item in mapping_candidates)
    mapping_ties = [
        item
        for item in mapping_candidates
        if math.isclose(item[0], mapping_best, abs_tol=absolute, rel_tol=relative)
    ]
    informed = min(mapping_ties, key=lambda item: item[1])
    return {
        "baseline": baseline,
        "baseline_ties": baseline_ties,
        "informed": informed,
        "mapping_ties": mapping_ties,
        "feasible_baseline_count": len(baseline_candidates),
        "feasible_mapping_count": len(mapping_candidates),
        "mapping_count": len(policy_ids) ** len(states),
    }


def risk_sensitive_constrained_voi(
    specification: dict[str, object],
) -> RiskSensitiveVoiResult:
    """Evaluate exact perfect-information VOI under a fixed risk/constraint model."""
    try:
        payload = cast(
            "dict[str, Any]", json.loads(json.dumps(specification, ensure_ascii=False))
        )
        validate_risk_sensitive_voi_semantics(payload)
        constraints = cast("list[dict[str, Any]]", payload["constraints"])
        solved = _solve(payload, constraints)
        baseline = solved["baseline"]
        informed = solved["informed"]
        gross = float(informed[0]) - float(baseline[0])
        cost = float(payload["information_action"]["cost"]["amount"])
        baseline_policy = cast("str", baseline[1])
        informed_mapping = cast("dict[str, str]", informed[2])
        shadow_evidence: list[dict[str, Any]] = []
        for omitted in constraints:
            reduced = [
                item
                for item in constraints
                if item["constraint_id"] != omitted["constraint_id"]
            ]
            relaxed = _solve(payload, reduced)
            relaxed_gross = float(relaxed["informed"][0]) - float(
                relaxed["baseline"][0]
            )
            shadow_evidence.append(
                {
                    "constraint_id": omitted["constraint_id"],
                    "shadow_value_status": "not_a_local_shadow_price",
                    "method": "exact_constraint_removal",
                    "gross_voi_without_constraint": relaxed_gross,
                    "constraint_removal_effect_on_gross_voi": relaxed_gross - gross,
                    "unit": payload["objective"]["unit"],
                }
            )
        result = {
            "schema_version": "1.0.0",
            "analysis_id": payload["analysis_id"],
            "analysis_type": "risk_sensitive_constrained_perfect_information_result",
            "method_maturity": "experimental",
            "planned_version": "v1.3.0",
            "objective": payload["objective"],
            "baseline": {
                "selected_policy_id": baseline_policy,
                "tied_policy_ids": sorted(item[1] for item in solved["baseline_ties"]),
                "objective_score": baseline[0],
                "expected_value": baseline[3],
                "worst_case_regret": baseline[4],
            },
            "perfect_information": {
                "selected_policy_by_state": informed_mapping,
                "tied_policy_mappings": [item[2] for item in solved["mapping_ties"]],
                "objective_score": informed[0],
                "expected_value": informed[3],
                "worst_case_regret": informed[4],
            },
            "value": {
                "gross": gross,
                "information_cost": cost,
                "net": gross - cost,
                "unit": payload["objective"]["unit"],
            },
            "switches": [
                {
                    "state_id": state_id,
                    "from_policy_id": baseline_policy,
                    "to_policy_id": policy_id,
                }
                for state_id, policy_id in informed_mapping.items()
                if policy_id != baseline_policy
            ],
            "risk_diagnostics": {
                "objective_kind": payload["objective"]["kind"],
                "baseline_objective_score": baseline[0],
                "perfect_information_objective_score": informed[0],
                "baseline_expected_value": baseline[3],
                "perfect_information_expected_value": informed[3],
                "baseline_worst_case_regret": baseline[4],
                "perfect_information_worst_case_regret": informed[4],
                "lower_tail_mass": 1.0 - float(payload["objective"]["confidence_level"])
                if payload["objective"]["kind"] == "lower_tail_cvar"
                else None,
            },
            "constraint_diagnostics": {
                "baseline": baseline[5],
                "perfect_information": informed[5],
            },
            "shadow_value_evidence": shadow_evidence,
            "enumeration": {
                "exact": True,
                "estimator": "exact_finite_enumeration",
                "tie_policy": "lexicographic_minimum_with_complete_ties",
                "policy_count": len(payload["policies"]),
                "state_count": len(payload["states"]),
                "mapping_count_evaluated": solved["mapping_count"],
                "feasible_baseline_count": solved["feasible_baseline_count"],
                "feasible_mapping_count": solved["feasible_mapping_count"],
            },
            "provenance": payload["provenance"],
        }
        validate_risk_sensitive_voi_result(result)
    except (ArithmeticError, KeyError, TypeError, ValueError) as error:
        raise_input_error(str(error))
    return RiskSensitiveVoiResult(result)


__all__ = ["RiskSensitiveVoiResult", "risk_sensitive_constrained_voi"]
