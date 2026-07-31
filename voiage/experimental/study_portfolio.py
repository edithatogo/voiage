"""Experimental exact allocation of governed single-study COSS optima."""

from __future__ import annotations

from itertools import combinations
from math import isclose
from typing import TYPE_CHECKING

from pydantic import ValidationError

from voiage.contracts.study_portfolio import (
    CossPortfolioCandidateV1,
    PortfolioCapacityConstraintV1,
    StudyPortfolioEvaluationV1,
    StudyPortfolioResultV1,
)
from voiage.exceptions import InputError

if TYPE_CHECKING:
    from collections.abc import Sequence

_DEFAULT_ATOL = 1e-10
_DEFAULT_RTOL = 1e-8
_MAX_EXACT_CANDIDATES = 24


def _normalize_candidates(
    values: Sequence[object],
) -> tuple[CossPortfolioCandidateV1, ...]:
    """Return governed candidates or fail with the public domain error."""
    if any(not isinstance(item, CossPortfolioCandidateV1) for item in values):
        raise InputError("candidates must contain CossPortfolioCandidateV1 objects")
    return tuple(item for item in values if isinstance(item, CossPortfolioCandidateV1))


def _normalize_constraints(
    values: Sequence[object],
) -> tuple[PortfolioCapacityConstraintV1, ...]:
    """Return governed constraints or fail with the public domain error."""
    if any(not isinstance(item, PortfolioCapacityConstraintV1) for item in values):
        raise InputError(
            "constraints must contain PortfolioCapacityConstraintV1 objects"
        )
    return tuple(
        item for item in values if isinstance(item, PortfolioCapacityConstraintV1)
    )


def _optimum(
    candidate: CossPortfolioCandidateV1,
) -> tuple[str, int, float, float, float]:
    design_id = candidate.coss.optimal_design_id
    if design_id is None:  # guarded by the candidate contract
        raise InputError("portfolio candidate COSS must have a feasible optimum")
    point = next(
        item for item in candidate.coss.evaluated_designs if item.design_id == design_id
    )
    return (
        point.design_id,
        point.sample_size,
        point.evsi,
        point.research_cost,
        point.enbs,
    )


def _is_admissible(
    selected: tuple[CossPortfolioCandidateV1, ...],
    constraints: tuple[PortfolioCapacityConstraintV1, ...],
    absolute_tolerance: float,
) -> bool:
    selected_ids = {item.study_id for item in selected}
    if any(not item.guardrails_passed for item in selected):
        return False
    if any(not set(item.required_study_ids) <= selected_ids for item in selected):
        return False
    occupied_groups: set[str] = set()
    for item in selected:
        for group_id in item.exclusion_group_ids:
            if group_id in occupied_groups:
                return False
            occupied_groups.add(group_id)
    return all(
        sum(item.resource_use.get(constraint.constraint_id, 0.0) for item in selected)
        <= constraint.capacity + absolute_tolerance
        for constraint in constraints
    )


def allocate_coss_portfolio(
    *,
    candidates: Sequence[CossPortfolioCandidateV1],
    constraints: Sequence[PortfolioCapacityConstraintV1] = (),
    absolute_tolerance: float = _DEFAULT_ATOL,
    relative_tolerance: float = _DEFAULT_RTOL,
) -> StudyPortfolioResultV1:
    """Maximize additive signed ENBS over COSS optima under hard constraints.

    The empty portfolio is always admissible. Consequently, a study with
    negative incremental ENBS is never funded merely because spare capacity is
    available. Ties are resolved by lower research cost and then the
    lexicographically ordered study-ID tuple.
    """
    raw_candidates: tuple[object, ...] = tuple(candidates)
    raw_constraints: tuple[object, ...] = tuple(constraints)
    candidate_tuple = _normalize_candidates(raw_candidates)
    constraint_tuple = _normalize_constraints(raw_constraints)
    if not candidate_tuple:
        raise InputError("at least one COSS portfolio candidate is required")
    if len(candidate_tuple) > _MAX_EXACT_CANDIDATES:
        raise InputError(
            f"exact portfolio allocation supports at most {_MAX_EXACT_CANDIDATES} candidates"
        )
    if absolute_tolerance < 0.0 or relative_tolerance < 0.0:
        raise InputError("portfolio tolerances must be non-negative")
    study_ids = [item.study_id for item in candidate_tuple]
    if len(set(study_ids)) != len(study_ids):
        raise InputError("portfolio study_id values must be unique")
    constraint_ids = [item.constraint_id for item in constraint_tuple]
    if len(set(constraint_ids)) != len(constraint_ids):
        raise InputError("portfolio constraint_id values must be unique")
    declared_constraints = set(constraint_ids)
    if any(set(item.resource_use) - declared_constraints for item in candidate_tuple):
        raise InputError("candidate resource_use references an undeclared constraint")
    known_studies = set(study_ids)
    if any(set(item.required_study_ids) - known_studies for item in candidate_tuple):
        raise InputError("candidate dependency references an unknown study_id")
    context_key = candidate_tuple[0].coss.context.commensurability_key()
    if any(
        item.coss.context.commensurability_key() != context_key
        for item in candidate_tuple[1:]
    ):
        raise InputError("portfolio COSS candidates must be commensurate")

    optimum = {item.study_id: _optimum(item) for item in candidate_tuple}
    admissible: list[
        tuple[tuple[CossPortfolioCandidateV1, ...], float, float, tuple[str, ...]]
    ] = [((), 0.0, 0.0, ())]
    for count in range(1, len(candidate_tuple) + 1):
        for subset in combinations(candidate_tuple, count):
            if not _is_admissible(subset, constraint_tuple, absolute_tolerance):
                continue
            enbs = sum(
                optimum[item.study_id][4]
                - item.incremental_cost.opportunity_cost
                - item.incremental_cost.implementation_delay_cost
                for item in subset
            )
            cost = sum(
                optimum[item.study_id][3]
                + item.incremental_cost.opportunity_cost
                + item.incremental_cost.implementation_delay_cost
                for item in subset
            )
            ordered_ids = tuple(sorted(item.study_id for item in subset))
            admissible.append((subset, enbs, cost, ordered_ids))

    global_max_enbs = max(item[1] for item in admissible)
    tie_tolerance = absolute_tolerance + relative_tolerance * max(
        abs(global_max_enbs), 1.0
    )
    tie_set = tuple(
        item for item in admissible if global_max_enbs - item[1] <= tie_tolerance
    )
    best, _best_enbs, _best_cost, _best_ids = min(
        tie_set, key=lambda item: (item[2], item[3])
    )

    selected_ids = {item.study_id for item in best}
    evaluations = tuple(
        StudyPortfolioEvaluationV1(
            study_id=item.study_id,
            design_id=optimum[item.study_id][0],
            sample_size=optimum[item.study_id][1],
            selected=item.study_id in selected_ids,
            gross_evsi=optimum[item.study_id][2],
            net_evsi=(
                optimum[item.study_id][2]
                - item.incremental_cost.opportunity_cost
                - item.incremental_cost.implementation_delay_cost
            ),
            research_cost=optimum[item.study_id][3],
            opportunity_cost=item.incremental_cost.opportunity_cost,
            implementation_delay_cost=item.incremental_cost.implementation_delay_cost,
            gross_enbs=optimum[item.study_id][4],
            net_enbs=(
                optimum[item.study_id][4]
                - item.incremental_cost.opportunity_cost
                - item.incremental_cost.implementation_delay_cost
            ),
            enbs=(
                optimum[item.study_id][4]
                - item.incremental_cost.opportunity_cost
                - item.incremental_cost.implementation_delay_cost
            ),
            efficiency_ratio=None if item.efficiency is None else item.efficiency.ratio,
            resource_use=item.resource_use,
            primary_metric_id=item.primary_metric_id,
            secondary_metric_ids=item.secondary_metric_ids,
            guardrail_ids=item.guardrail_ids,
            failed_guardrail_ids=item.failed_guardrail_ids,
            heterogeneous_effect_model_id=item.heterogeneous_effect_model_id,
            delayed_effect_model_id=item.delayed_effect_model_id,
            interference_model_id=item.interference_model_id,
            sequential_monitoring_plan_id=item.sequential_monitoring_plan_id,
            multiplicity_adjustment_id=item.multiplicity_adjustment_id,
            stopping_rule_ids=item.stopping_rule_ids,
            model_assurances=item.model_assurances,
            study_duration=item.study_duration,
            duration_unit=item.duration_unit,
            incremental_cost=item.incremental_cost,
            expected_policy_change_id=item.expected_policy_change_id,
        )
        for item in candidate_tuple
    )
    selected = tuple(item for item in evaluations if item.selected)
    used_capacity = {
        constraint.constraint_id: sum(
            item.resource_use.get(constraint.constraint_id, 0.0) for item in selected
        )
        for constraint in constraint_tuple
    }
    binding = tuple(
        constraint.constraint_id
        for constraint in constraint_tuple
        if isclose(
            used_capacity[constraint.constraint_id],
            constraint.capacity,
            rel_tol=relative_tolerance,
            abs_tol=absolute_tolerance,
        )
    )
    diagnostics = [
        "additive_enbs_assumption",
        "candidate_model_assurances_verified",
        "incremental_cost_exclusion_declared",
        "global_maximum_tie_set",
    ]
    if not selected:
        diagnostics.append("empty_portfolio_selected")
    if any(not item.guardrails_passed for item in candidate_tuple):
        diagnostics.append("guardrail_failed_candidates_excluded")
    if any(item.efficiency is None for item in candidate_tuple):
        diagnostics.append("efficiency_not_supplied_for_all_candidates")
    try:
        return StudyPortfolioResultV1(
            evaluations=evaluations,
            constraints=constraint_tuple,
            selected_study_ids=tuple(item.study_id for item in selected),
            total_gross_evsi=sum(item.gross_evsi for item in selected),
            total_net_evsi=sum(item.net_evsi for item in selected),
            total_research_cost=sum(item.research_cost for item in selected),
            total_opportunity_cost=sum(item.opportunity_cost for item in selected),
            total_implementation_delay_cost=sum(
                item.implementation_delay_cost for item in selected
            ),
            total_gross_enbs=sum(item.gross_enbs for item in selected),
            total_net_enbs=sum(item.net_enbs for item in selected),
            total_enbs=sum(item.net_enbs for item in selected),
            used_capacity=used_capacity,
            binding_constraint_ids=binding,
            selected_policy_change_ids=tuple(
                item.expected_policy_change_id for item in selected
            ),
            selected_stopping_rule_ids=tuple(
                dict.fromkeys(
                    rule for item in selected for rule in item.stopping_rule_ids
                )
            ),
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
            diagnostics=tuple(diagnostics),
        )
    except ValidationError as error:
        raise InputError(
            "portfolio result failed scientific contract validation"
        ) from error


__all__ = ["allocate_coss_portfolio"]
