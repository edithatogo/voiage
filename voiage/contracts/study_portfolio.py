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
    primary_metric_id: Identifier
    secondary_metric_ids: tuple[Identifier, ...]
    guardrail_ids: tuple[Identifier, ...]
    failed_guardrail_ids: tuple[Identifier, ...]
    heterogeneous_effect_model_id: Identifier
    delayed_effect_model_id: Identifier
    interference_model_id: Identifier
    sequential_monitoring_plan_id: Identifier
    multiplicity_adjustment_id: Identifier
    stopping_rule_ids: tuple[Identifier, ...]
    study_duration: float = Field(gt=0.0)
    duration_unit: Identifier
    opportunity_cost: float = Field(ge=0.0)
    implementation_delay_cost: float = Field(ge=0.0)
    expected_policy_change_id: Identifier

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
        if len(set(self.secondary_metric_ids)) != len(self.secondary_metric_ids):
            raise ValueError("secondary_metric_ids must be unique")
        if self.primary_metric_id in self.secondary_metric_ids:
            raise ValueError("primary metric cannot also be a secondary metric")
        if len(set(self.guardrail_ids)) != len(self.guardrail_ids):
            raise ValueError("guardrail_ids must be unique")
        if not set(self.failed_guardrail_ids) <= set(self.guardrail_ids):
            raise ValueError("failed_guardrail_ids must be declared guardrails")
        if self.guardrails_passed == bool(self.failed_guardrail_ids):
            raise ValueError("guardrails_passed must agree with failed_guardrail_ids")
        if not self.stopping_rule_ids or len(set(self.stopping_rule_ids)) != len(
            self.stopping_rule_ids
        ):
            raise ValueError("stopping_rule_ids must be non-empty and unique")
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
            if not isclose(efficiency.evsi, optimum.evsi, rel_tol=1e-15, abs_tol=0.0):
                raise ValueError(
                    "portfolio efficiency EVSI must match the COSS optimum"
                )
        return self


class StudyPortfolioEvaluationV1(ContractModel):
    """Allocation record for one evaluated study candidate."""

    study_id: Identifier
    design_id: Identifier
    sample_size: int = Field(ge=0)
    selected: bool
    gross_evsi: float = Field(ge=0.0)
    net_evsi: float
    research_cost: float = Field(ge=0.0)
    opportunity_cost: float = Field(ge=0.0)
    implementation_delay_cost: float = Field(ge=0.0)
    gross_enbs: float
    net_enbs: float
    enbs: float
    efficiency_ratio: float | None = None
    resource_use: Mapping[Identifier, float] = Field(default_factory=dict)
    primary_metric_id: Identifier
    secondary_metric_ids: tuple[Identifier, ...]
    guardrail_ids: tuple[Identifier, ...]
    failed_guardrail_ids: tuple[Identifier, ...]
    heterogeneous_effect_model_id: Identifier
    delayed_effect_model_id: Identifier
    interference_model_id: Identifier
    sequential_monitoring_plan_id: Identifier
    multiplicity_adjustment_id: Identifier
    stopping_rule_ids: tuple[Identifier, ...]
    study_duration: float = Field(gt=0.0)
    duration_unit: Identifier
    expected_policy_change_id: Identifier

    @field_serializer("resource_use")
    def serialize_resource_use(self, value: object) -> object:
        """Restore the JSON mapping shape for canonical serialization."""
        return thaw_json(value)

    @model_validator(mode="after")
    def validate_enbs(self) -> Self:
        """Preserve the signed-ENBS identity at portfolio boundaries."""
        identities = (
            (
                self.net_evsi,
                self.gross_evsi
                - self.opportunity_cost
                - self.implementation_delay_cost,
            ),
            (self.gross_enbs, self.gross_evsi - self.research_cost),
            (self.net_enbs, self.net_evsi - self.research_cost),
            (self.enbs, self.net_enbs),
        )
        if any(
            not isclose(actual, expected, rel_tol=0.0, abs_tol=0.0)
            for actual, expected in identities
        ):
            raise ValueError("portfolio value and signed-ENBS identities disagree")
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
    total_net_evsi: float
    total_research_cost: float = Field(ge=0.0)
    total_opportunity_cost: float = Field(ge=0.0)
    total_implementation_delay_cost: float = Field(ge=0.0)
    total_gross_enbs: float
    total_net_enbs: float
    total_enbs: float
    used_capacity: Mapping[Identifier, float] = Field(default_factory=dict)
    binding_constraint_ids: tuple[Identifier, ...] = ()
    selected_policy_change_ids: tuple[Identifier, ...]
    selected_stopping_rule_ids: tuple[Identifier, ...]
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
            sum(item.net_evsi for item in selected),
            sum(item.research_cost for item in selected),
            sum(item.opportunity_cost for item in selected),
            sum(item.implementation_delay_cost for item in selected),
            sum(item.gross_enbs for item in selected),
            sum(item.net_enbs for item in selected),
            sum(item.net_enbs for item in selected),
        )
        actual_totals = (
            self.total_gross_evsi,
            self.total_net_evsi,
            self.total_research_cost,
            self.total_opportunity_cost,
            self.total_implementation_delay_cost,
            self.total_gross_enbs,
            self.total_net_enbs,
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
        if any(set(item.resource_use) - set(constraints) for item in self.evaluations):
            raise ValueError(
                "evaluation resource_use references an undeclared constraint"
            )
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
        expected_binding = tuple(
            constraint.constraint_id
            for constraint in self.constraints
            if isclose(
                self.used_capacity[constraint.constraint_id],
                constraint.capacity,
                rel_tol=self.relative_tolerance,
                abs_tol=self.absolute_tolerance,
            )
        )
        if self.binding_constraint_ids != expected_binding:
            raise ValueError("binding_constraint_ids disagree with used capacity")
        expected_policy_changes = tuple(
            item.expected_policy_change_id for item in selected
        )
        if self.selected_policy_change_ids != expected_policy_changes:
            raise ValueError("selected_policy_change_ids disagree with evaluations")
        expected_stopping_rules = tuple(
            dict.fromkeys(rule for item in selected for rule in item.stopping_rule_ids)
        )
        if self.selected_stopping_rule_ids != expected_stopping_rules:
            raise ValueError("selected_stopping_rule_ids disagree with evaluations")
        return self


__all__ = [
    "CossPortfolioCandidateV1",
    "PortfolioCapacityConstraintV1",
    "StudyPortfolioEvaluationV1",
    "StudyPortfolioResultV1",
]
