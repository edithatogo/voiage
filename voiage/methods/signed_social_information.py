"""Exact finite signed, social and selective-sharing information value.

The experimental evaluator consumes a complete finite joint-world law and a
bounded policy catalog.  It intentionally does not solve persuasion,
mechanism-design, rational-inattention or general game problems.
"""

# pyright: reportAny=false, reportExplicitAny=false, reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import TYPE_CHECKING, Any, cast

from voiage._runtime import compute_signed_social_information
from voiage.contracts.signed_social_information import (
    validate_signed_social_information_result,
    validate_signed_social_information_semantics,
)
from voiage.exceptions import InputError, raise_input_error

if TYPE_CHECKING:
    from collections.abc import Mapping


_ROLES = ("decision_maker", "recipient", "controller", "stakeholder")


@dataclass(frozen=True)
class SignedSocialInformationResult:
    """Portable result envelope for exact finite signed/social information."""

    payload: dict[str, Any]

    def to_contract_dict(self) -> dict[str, Any]:
        """Return an independent JSON-compatible result copy."""
        return cast("dict[str, Any]", json.loads(json.dumps(self.payload)))


def signed_social_information_value(
    specification: Mapping[str, object],
) -> SignedSocialInformationResult:
    """Evaluate bounded signed private, role and social information values.

    The installed Rust extension evaluates the finite payoff tables. A centralized
    design selects the exact best policy for its declared agent or welfare objective.
    Fixed, declared-response, and verified finite-equilibrium designs use their
    declared catalog entry. Negative values are retained without clipping.
    """
    try:
        payload = cast("dict[str, Any]", json.loads(json.dumps(specification)))
        validate_signed_social_information_semantics(payload)
        result = _evaluate(payload)
        validate_signed_social_information_result(result)
    except InputError:
        raise
    except (ArithmeticError, KeyError, TypeError, ValueError) as error:
        raise_input_error(str(error))
    return SignedSocialInformationResult(result)


def _policy_payoffs(
    policy: Mapping[str, Any],
    worlds: list[dict[str, Any]],
    agent_ids: list[str],
) -> list[list[float]]:
    decisions = {
        str(item["observation"]): str(item["action_id"])
        for item in cast("list[dict[str, str]]", policy["decisions"])
    }
    unobserved = decisions.get("unobserved")
    payoffs: list[list[float]] = []
    for world in worlds:
        action = unobserved if unobserved is not None else decisions[world["signal"]]
        utilities = world["action_utilities"][action]
        payoffs.append([float(utilities[agent_id]) for agent_id in agent_ids])
    return payoffs


def _ties(values: Mapping[str, float], absolute: float, relative: float) -> list[str]:
    best = max(values.values())
    return sorted(
        key
        for key, value in values.items()
        if math.isclose(value, best, abs_tol=absolute, rel_tol=relative)
    )


def _exact_best(values: Mapping[str, float]) -> str:
    best = max(values.values())
    return min(key for key, value in values.items() if value == best)


def _infeasibility_reasons(
    design: Mapping[str, Any], receipts: Mapping[str, Mapping[str, Any]]
) -> list[str]:
    return sorted(
        f"consent_denied:{receipts[receipt_id]['subject_agent_id']}"
        for receipt_id in cast("list[str]", design["rights_receipt_ids"])
        if receipts[receipt_id]["consent_status"] == "denied"
    )


def _blackwell_check(
    design: Mapping[str, Any],
    comparator: Mapping[str, Any],
    evaluated: Mapping[str, Any],
    comparator_evaluated: Mapping[str, Any],
    tolerance: float,
) -> dict[str, Any]:
    assurance = design["blackwell_assurance"]
    reasons: list[str] = []
    if design["selection_mode"] != "centralized":
        reasons.append("selection_mode_not_centralized")
    if comparator["selection_mode"] != "centralized":
        reasons.append("comparator_not_centralized")
    if design["selector"] != comparator["selector"]:
        reasons.append("selector_not_aligned")
    if not bool(evaluated["feasible"]):
        reasons.append("design_infeasible")
    if not bool(comparator_evaluated["feasible"]):
        reasons.append("comparator_infeasible")
    if design["transfers"] or design["costs"]:
        reasons.append("design_has_transfers_or_costs")
    if comparator["transfers"] or comparator["costs"]:
        reasons.append("comparator_has_transfers_or_costs")
    if assurance is None:
        reasons.append("refinement_assurance_not_declared")
    applicable = not reasons
    checked_value: float | None = None
    passed: bool | None = None
    if applicable:
        selector = str(design["selector"])
        if selector == "social_welfare":
            checked_value = float(evaluated["social_pre_transfer"]) - float(
                comparator_evaluated["social_pre_transfer"]
            )
        else:
            agent_id = selector.removeprefix("agent:")
            checked_value = float(
                evaluated["ledgers"]["pre_transfer"][agent_id]
            ) - float(comparator_evaluated["ledgers"]["pre_transfer"][agent_id])
        passed = checked_value >= -tolerance
        if not passed:  # pragma: no cover - exact embedded maximization theorem
            raise ValueError("applicable Blackwell nonnegativity check failed")
    return {
        "applicable": applicable,
        "checked_value": checked_value,
        "passed": passed,
        "reasons_not_applicable": sorted(reasons),
    }


def _numerical_designs(
    designs: Mapping[str, dict[str, Any]],
    agent_indices: Mapping[str, int],
    policy_indices: Mapping[str, int],
) -> list[dict[str, object]]:
    design_indices = {design_id: i for i, design_id in enumerate(designs)}
    numerical: list[dict[str, object]] = []
    for design_id, design in designs.items():
        selector = str(design["selector"])
        selected = design["selected_policy_id"]
        comparator = design["comparator_design_id"]
        numerical.append(
            {
                "policies": [
                    policy_indices[key] for key in sorted(design["policy_ids"])
                ],
                "selected_policy": None
                if selected is None
                else policy_indices[selected],
                "selector_agent": (
                    None
                    if selector == "social_welfare"
                    else agent_indices[selector.removeprefix("agent:")]
                ),
                "comparator": design_indices[
                    design_id if comparator is None else comparator
                ],
                "transfers": [
                    (
                        agent_indices[item["payer_agent_id"]],
                        agent_indices[item["recipient_agent_id"]],
                        float(item["amount"]),
                    )
                    for item in design["transfers"]
                ],
                "costs": [
                    (agent_indices[item["agent_id"]], float(item["amount"]))
                    for item in design["costs"]
                ],
            }
        )
    return numerical


def _evaluate(data: dict[str, Any]) -> dict[str, Any]:
    agents = cast("list[dict[str, Any]]", data["agents"])
    agent_ids = sorted(str(agent["agent_id"]) for agent in agents)
    worlds = cast("list[dict[str, Any]]", data["worlds"])
    policies = cast("list[dict[str, Any]]", data["policies"])
    policy_by_id = {str(policy["policy_id"]): policy for policy in policies}
    policy_ids = sorted(policy_by_id)
    agent_indices = {agent_id: i for i, agent_id in enumerate(agent_ids)}
    policy_indices = {policy_id: i for i, policy_id in enumerate(policy_ids)}
    welfare = cast("dict[str, Any]", data["welfare"])
    receipts = {
        str(receipt["receipt_id"]): receipt
        for receipt in cast("list[dict[str, Any]]", data["receipts"])
    }
    designs = cast("list[dict[str, Any]]", data["designs"])
    design_by_id = {
        str(design["design_id"]): design
        for design in sorted(designs, key=lambda item: str(item["design_id"]))
    }
    arithmetic = compute_signed_social_information(
        {
            "world_probabilities": [float(world["probability"]) for world in worlds],
            "policy_payoffs": [
                _policy_payoffs(policy_by_id[key], worlds, agent_ids)
                for key in policy_ids
            ],
            "welfare_weights": [float(welfare["weights"][key]) for key in agent_ids],
            "role_agents": [
                [
                    agent_indices[agent["agent_id"]]
                    for agent in agents
                    if role in agent["roles"]
                ]
                for role in _ROLES
            ],
            "post_transfer_welfare": welfare["ledger_stage"] == "post_transfer",
            "designs": _numerical_designs(design_by_id, agent_indices, policy_indices),
        }
    )
    tie_policy = cast("dict[str, float]", data["tie_policy"])
    absolute = float(tie_policy["absolute_tolerance"])
    relative = float(tie_policy["relative_tolerance"])

    absolute_evaluations: dict[str, dict[str, Any]] = {}
    for design_id, values in zip(design_by_id, arithmetic, strict=True):
        design = design_by_id[design_id]
        numerical = cast("dict[str, Any]", values)
        catalog = sorted(cast("list[str]", design["policy_ids"]))
        selector_values = dict(zip(catalog, numerical["selector_values"], strict=True))
        selected_policy_id = policy_ids[numerical["selected_policy"]]
        if design["selection_mode"] == "centralized":
            policy_tie = _ties(selector_values, absolute, relative)
        else:
            policy_tie = [selected_policy_id]
        reasons = _infeasibility_reasons(design, receipts)
        absolute_evaluations[design_id] = {
            "design_id": design_id,
            "comparator_design_id": design["comparator_design_id"],
            "recipients": sorted(design["recipients"]),
            "selection_mode": design["selection_mode"],
            "selector": design["selector"],
            "feasible": not reasons,
            "infeasibility_reasons": reasons,
            "policies_evaluated": catalog,
            "policy_selector_values": selector_values,
            "policy_tie": policy_tie,
            "selected_policy_id": selected_policy_id,
            "ledgers": {
                stage: dict(zip(agent_ids, entries, strict=True))
                for stage, entries in numerical["ledger"].items()
            },
            "social_pre_transfer": numerical["social_pre_transfer"],
            "social_post_transfer": numerical["social_post_transfer"],
            "rights_receipts": sorted(design["rights_receipt_ids"]),
            "equilibrium_receipt": design["equilibrium_receipt"],
            "signed_values": {
                "by_agent": dict(
                    zip(agent_ids, numerical["signed_by_agent"], strict=True)
                ),
                "by_role": dict(zip(_ROLES, numerical["signed_by_role"], strict=True)),
                "social": numerical["signed_welfare"],
                "comparator_design_id": design["comparator_design_id"],
                "clipped_at_zero": False,
            },
        }

    evaluated_designs: list[dict[str, Any]] = []
    for design_id in sorted(absolute_evaluations):
        evaluated = absolute_evaluations[design_id]
        design = design_by_id[design_id]
        comparator_id = design["comparator_design_id"]
        comparator = (
            evaluated if comparator_id is None else absolute_evaluations[comparator_id]
        )
        evaluated_designs.append(
            {
                **evaluated,
                "policy_switch": evaluated["selected_policy_id"]
                != comparator["selected_policy_id"],
                "blackwell_nonnegativity": _blackwell_check(
                    design,
                    design if comparator_id is None else design_by_id[comparator_id],
                    evaluated,
                    comparator,
                    absolute,
                ),
            }
        )

    feasible_social = {
        str(item["design_id"]): float(
            item[
                "social_pre_transfer"
                if welfare["ledger_stage"] == "pre_transfer"
                else "social_post_transfer"
            ]
        )
        for item in evaluated_designs
        if bool(item["feasible"])
    }
    optimum_tie = _ties(feasible_social, absolute, relative)
    selected_design_id = _exact_best(feasible_social)
    selected_design = next(
        item for item in evaluated_designs if item["design_id"] == selected_design_id
    )
    selected_agent_values = cast(
        "dict[str, float]", selected_design["signed_values"]["by_agent"]
    )
    winners = sorted(
        agent_id
        for agent_id, value in selected_agent_values.items()
        if value > absolute
    )
    losers = sorted(
        agent_id
        for agent_id, value in selected_agent_values.items()
        if value < -absolute
    )
    recipient_ids = {
        str(agent["agent_id"])
        for agent in agents
        if "recipient" in cast("list[str]", agent["roles"])
    }
    harmful_private = sorted(
        str(item["design_id"])
        for item in evaluated_designs
        if item["comparator_design_id"] is not None
        and any(
            float(item["signed_values"]["by_agent"][agent_id]) < -absolute
            for agent_id in recipient_ids
        )
    )
    avoidance = sorted(
        (
            {"agent_id": agent_id, "design_id": str(item["design_id"])}
            for item in evaluated_designs
            if item["comparator_design_id"] is not None
            for agent_id, value in item["signed_values"]["by_agent"].items()
            if float(value) < -absolute
        ),
        key=lambda record: (record["agent_id"], record["design_id"]),
    )
    baseline_id = str(data["baseline_design_id"])
    baseline = next(
        item for item in evaluated_designs if item["design_id"] == baseline_id
    )
    return {
        "schema_version": "1.0.0",
        "analysis_id": data["analysis_id"],
        "analysis_type": "signed_social_information_value_result",
        "method_maturity": "experimental",
        "value_unit": data["value_unit"],
        "agent_roles": {
            str(agent["agent_id"]): sorted(cast("list[str]", agent["roles"]))
            for agent in sorted(agents, key=lambda item: str(item["agent_id"]))
        },
        "welfare_contract": welfare,
        "topology": data["topology"],
        "baseline": baseline,
        "designs": evaluated_designs,
        "optimum": {
            "feasible_design_values": dict(sorted(feasible_social.items())),
            "design_tie": optimum_tie,
            "selected_design_id": selected_design_id,
            "social_value": feasible_social[selected_design_id],
            "tie_policy": data["tie_policy"],
        },
        "diagnostics": {
            "winners": winners,
            "losers": losers,
            "harmful_private_designs": harmful_private,
            "information_avoidance": avoidance,
            "policy_switches": sorted(
                str(item["design_id"])
                for item in evaluated_designs
                if bool(item["policy_switch"])
            ),
            "winner_loser_design_id": selected_design_id,
            "externality_by_design": {
                str(item["design_id"]): float(item["signed_values"]["social"])
                - math.fsum(
                    float(item["signed_values"]["by_agent"][agent_id])
                    for agent_id in cast("list[str]", item["recipients"])
                )
                for item in evaluated_designs
            },
        },
        "assurance": {
            "worlds_evaluated": len(worlds),
            "world_ids_evaluated": sorted(str(world["world_id"]) for world in worlds),
            "policies_evaluated": len(policies),
            "policy_ids_evaluated": sorted(policy_by_id),
            "designs_evaluated": len(designs),
            "design_ids_evaluated": sorted(design_by_id),
            "complete_joint_world_law": True,
            "nonanticipativity": "one action per observable signal or one unobserved action",
            "finite_catalog_only": True,
            "general_game_solver_used": False,
            "negative_values_clipped": False,
            "rights_consent_purpose_receipts_checked": True,
            "deterministic_serialization": True,
        },
        "language_dispositions": {
            "python": "experimental_exact_finite_execution",
            "rust": "not_implemented",
            "r": "not_implemented",
            "julia": "not_implemented",
            "mojo": "external_upstream_boundary",
        },
        "unsupported_dispositions": {
            "bayesian_persuasion": "adjacent_not_supported_in_v1",
            "mechanism_design": "adjacent_not_supported_in_v1",
            "rational_inattention": "adjacent_not_supported_in_v1",
            "general_game_solving": "adjacent_not_supported_in_v1",
            "continuous_or_incomplete_world_laws": "not_supported_in_v1",
        },
        "provenance": data["provenance"],
    }
