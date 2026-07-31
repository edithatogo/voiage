"""Versioned contracts for experimental COSS portfolio allocation."""

from __future__ import annotations

from collections.abc import Mapping  # noqa: TC003 - Pydantic resolves annotations
from math import isclose
from typing import Annotated, Literal, Self

from pydantic import Field, StringConstraints, field_serializer, model_validator

from .analysis import ContractModel, thaw_json
from .study_design import (  # noqa: TC001 - Pydantic resolves nested models
    CossResultV1,
    InformationEfficiencyResultV1,
)

Identifier = Annotated[str, StringConstraints(min_length=1, strip_whitespace=True)]


class PortfolioCapacityConstraintV1(ContractModel):
    """One shared, additive resource limit such as traffic or duration."""

    constraint_id: Identifier
    capacity: float = Field(ge=0.0)
    unit: Identifier


class CossPortfolioCandidateV1(ContractModel):
    """A candidate study represented by its governed single-study optimum."""

    study_id: Identifier
    coss: CossResultV1
    efficiency: InformationEfficiencyResultV1 | None = None
    resource_use: Mapping[Identifier, float] = Field(default_factory=dict)
    required_study_ids: tuple[Identifier, ...] = ()
    exclusion_group_ids: tuple[Identifier, ...] = ()
    guardrails_passed: bool = True

    @field_serializer("resource_use")
    def serialize_resource_use(self, value: object) -> object:
        """Restore the JSON mapping shape for canonical serialization."""
        return thaw_json(value)

    @model_validator(mode="after")
    def validate_candidate(self) -> Self:
        """Require an allocable optimum and coherent optional efficiency."""
        if self.coss.optimal_design_id is None:
            raise ValueError("portfolio candidate COSS must have a feasible optimum")
        if any(value < 0.0 for value in self.resource_use.values()):
            raise ValueError("portfolio resource use must be non-negative")
        if self.study_id in self.required_study_ids:
            raise ValueError("a portfolio candidate cannot require itself")
        efficiency = self.efficiency
        if efficiency is not None:
            if (
                efficiency.context.commensurability_key()
                != self.coss.context.commensurability_key()
            ):
                raise ValueError("portfolio efficiency must match the COSS context")
            optimum = next(
                point
                for point in self.coss.evaluated_designs
                if point.design_id == self.coss.optimal_design_id
            )
            if not isclose(
                efficiency.evsi, optimum.evsi, rel_tol=1e-15, abs_tol=0.0
            ):
                raise ValueError("portfolio efficiency EVSI must match the COSS optimum")
        return self


class StudyPortfolioEvaluationV1(ContractModel):
    """Allocation record for one evaluated study candidate."""

    study_id: Identifier
    design_id: Identifier
    sample_size: int = Field(ge=0)
    selected: bool
    gross_evsi: float = Field(ge=0.0)
    research_cost: float = Field(ge=0.0)
    enbs: float
    efficiency_ratio: float | None = None
    resource_use: Mapping[Identifier, float] = Field(default_factory=dict)

    @field_serializer("resource_use")
    def serialize_resource_use(self, value: object) -> object:
        """Restore the JSON mapping shape for canonical serialization."""
        return thaw_json(value)

    @model_validator(mode="after")
    def validate_enbs(self) -> Self:
        """Preserve the signed-ENBS identity at portfolio boundaries."""
        if not isclose(
            self.enbs,
            self.gross_evsi - self.research_cost,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise ValueError("portfolio ENBS must equal gross EVSI minus research cost")
        return self


class StudyPortfolioResultV1(ContractModel):
    """Exact allocation over governed single-study COSS optima."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    method: Literal["coss_portfolio_allocation"] = "coss_portfolio_allocation"
    estimator: Literal["python_exact_subset_enumeration"] = (
        "python_exact_subset_enumeration"
    )
    evaluations: tuple[StudyPortfolioEvaluationV1, ...]
    constraints: tuple[PortfolioCapacityConstraintV1, ...] = ()
    selected_study_ids: tuple[Identifier, ...]
    total_gross_evsi: float = Field(ge=0.0)
    total_research_cost: float = Field(ge=0.0)
    total_enbs: float
    used_capacity: Mapping[Identifier, float] = Field(default_factory=dict)
    binding_constraint_ids: tuple[Identifier, ...] = ()
    absolute_tolerance: float = Field(ge=0.0)
    relative_tolerance: float = Field(ge=0.0)
    diagnostics: tuple[Identifier, ...] = ()

    @field_serializer("used_capacity")
    def serialize_used_capacity(self, value: object) -> object:
        """Restore the JSON mapping shape for canonical serialization."""
        return thaw_json(value)

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        """Reject allocation envelopes with inconsistent totals or capacities."""
        if not self.evaluations:
            raise ValueError("portfolio evaluations must not be empty")
        ids = tuple(item.study_id for item in self.evaluations)
        if len(set(ids)) != len(ids):
            raise ValueError("portfolio study_id values must be unique")
        selected = tuple(item for item in self.evaluations if item.selected)
        if self.selected_study_ids != tuple(item.study_id for item in selected):
            raise ValueError("selected_study_ids disagree with portfolio evaluations")
        expected_totals = (
            sum(item.gross_evsi for item in selected),
            sum(item.research_cost for item in selected),
            sum(item.enbs for item in selected),
        )
        actual_totals = (
            self.total_gross_evsi,
            self.total_research_cost,
            self.total_enbs,
        )
        if any(
            not isclose(actual, expected, rel_tol=1e-15, abs_tol=1e-12)
            for actual, expected in zip(actual_totals, expected_totals, strict=True)
        ):
            raise ValueError("portfolio totals disagree with selected evaluations")
        constraints = {item.constraint_id: item for item in self.constraints}
        if len(constraints) != len(self.constraints):
            raise ValueError("portfolio constraint_id values must be unique")
        if set(self.used_capacity) != set(constraints):
            raise ValueError("used_capacity must cover every declared constraint")
        expected_used = {
            constraint_id: sum(
                item.resource_use.get(constraint_id, 0.0) for item in selected
            )
            for constraint_id in constraints
        }
        for constraint_id, used in self.used_capacity.items():
            if not isclose(
                used, expected_used[constraint_id], rel_tol=1e-15, abs_tol=1e-12
            ):
                raise ValueError("used_capacity disagrees with selected evaluations")
            if used > constraints[constraint_id].capacity + self.absolute_tolerance:
                raise ValueError("selected portfolio exceeds a capacity constraint")
        return self


__all__ = [
    "CossPortfolioCandidateV1",
    "PortfolioCapacityConstraintV1",
    "StudyPortfolioEvaluationV1",
    "StudyPortfolioResultV1",
]
