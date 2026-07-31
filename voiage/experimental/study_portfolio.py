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


def _optimum(candidate: CossPortfolioCandidateV1) -> tuple[str, int, float, float, float]:
    design_id = candidate.coss.optimal_design_id
    if design_id is None:  # guarded by the candidate contract
        raise InputError("portfolio candidate COSS must have a feasible optimum")
    point = next(
        item for item in candidate.coss.evaluated_designs if item.design_id == design_id
    )
    return point.design_id, point.sample_size, point.evsi, point.research_cost, point.enbs


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
    candidate_tuple = tuple(candidates)
    constraint_tuple = tuple(constraints)
    if not candidate_tuple:
        raise InputError("at least one COSS portfolio candidate is required")
    if len(candidate_tuple) > _MAX_EXACT_CANDIDATES:
        raise InputError(
            f"exact portfolio allocation supports at most {_MAX_EXACT_CANDIDATES} candidates"
        )
    if any(not isinstance(item, CossPortfolioCandidateV1) for item in candidate_tuple):
        raise InputError("candidates must contain CossPortfolioCandidateV1 records")
    if any(
        not isinstance(item, PortfolioCapacityConstraintV1)
        for item in constraint_tuple
    ):
        raise InputError(
            "constraints must contain PortfolioCapacityConstraintV1 records"
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
    best: tuple[CossPortfolioCandidateV1, ...] = ()
    best_enbs = 0.0
    best_cost = 0.0
    for count in range(1, len(candidate_tuple) + 1):
        for subset in combinations(candidate_tuple, count):
            if not _is_admissible(subset, constraint_tuple, absolute_tolerance):
                continue
            enbs = sum(optimum[item.study_id][4] for item in subset)
            cost = sum(optimum[item.study_id][3] for item in subset)
            tolerance = absolute_tolerance + relative_tolerance * max(abs(best_enbs), 1.0)
            ordered_ids = tuple(sorted(item.study_id for item in subset))
            best_ids = tuple(sorted(item.study_id for item in best))
            if enbs > best_enbs + tolerance or (
                isclose(enbs, best_enbs, rel_tol=relative_tolerance, abs_tol=absolute_tolerance)
                and (cost, ordered_ids) < (best_cost, best_ids)
            ):
                best = subset
                best_enbs = enbs
                best_cost = cost

    selected_ids = {item.study_id for item in best}
    evaluations = tuple(
        StudyPortfolioEvaluationV1(
            study_id=item.study_id,
            design_id=optimum[item.study_id][0],
            sample_size=optimum[item.study_id][1],
            selected=item.study_id in selected_ids,
            gross_evsi=optimum[item.study_id][2],
            research_cost=optimum[item.study_id][3],
            enbs=optimum[item.study_id][4],
            efficiency_ratio=None if item.efficiency is None else item.efficiency.ratio,
            resource_use=item.resource_use,
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
    diagnostics = ["additive_enbs_assumption"]
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
            total_research_cost=sum(item.research_cost for item in selected),
            total_enbs=sum(item.enbs for item in selected),
            used_capacity=used_capacity,
            binding_constraint_ids=binding,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
            diagnostics=tuple(diagnostics),
        )
    except ValidationError as error:
        raise InputError("portfolio result failed scientific contract validation") from error


__all__ = ["allocate_coss_portfolio"]
