"""Versioned contracts for experimental study-design efficiency methods."""

from __future__ import annotations

from collections.abc import Mapping  # noqa: TC003 - Pydantic resolves annotations
from typing import Annotated, Literal, Self

from pydantic import (
    Field,
    JsonValue,
    StringConstraints,
    field_serializer,
    model_validator,
)

from .analysis import ContractModel, thaw_json

Identifier = Annotated[str, StringConstraints(min_length=1, strip_whitespace=True)]
TiePolicy = Literal["smallest_sample_size", "largest_sample_size", "first_declared"]
BoundaryState = Literal["none", "lower", "upper", "both", "interior"]
SelectionUncertaintyMethod = Literal[
    "unavailable", "analytic", "monte_carlo", "bootstrap", "externally_supplied"
]
EfficiencyStatus = Literal[
    "within_bounds",
    "below_zero_within_tolerance",
    "above_one_within_tolerance",
    "undefined_zero_evpi",
]


class StudyDesignContextV1(ContractModel):
    """Commensurability and provenance shared by study values and costs."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    decision_problem_id: Identifier
    value_unit: Identifier
    population_scale: float = Field(gt=0.0)
    time_horizon: Identifier
    discounting_id: Identifier
    study_model_id: Identifier
    cost_model_id: Identifier
    random_seed: int | None = Field(default=None, ge=0)

    def commensurability_key(self) -> tuple[str, str, float, str, str]:
        """Return the fields that must match for an information-value ratio."""
        return (
            self.decision_problem_id,
            self.value_unit,
            self.population_scale,
            self.time_horizon,
            self.discounting_id,
        )


class FeasibleDesignRangeV1(ContractModel):
    """Inclusive sample-size range optionally declaring a regular step."""

    lower_sample_size: int = Field(ge=0)
    upper_sample_size: int = Field(ge=0)
    step: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_order(self) -> Self:
        """Reject inverted ranges."""
        if self.lower_sample_size > self.upper_sample_size:
            raise ValueError("lower_sample_size must not exceed upper_sample_size")
        return self


class StudyDesignPointInputV1(ContractModel):
    """One evaluated design and its commensurate EVSI and research cost."""

    design_id: Identifier
    sample_size: int = Field(ge=0)
    evsi: float = Field(ge=0.0)
    research_cost: float = Field(ge=0.0)
    feasible: bool = True
    feasibility_codes: tuple[Identifier, ...] = ()
    evsi_standard_error: float | None = Field(default=None, ge=0.0)
    cost_standard_error: float | None = Field(default=None, ge=0.0)
    enbs_standard_error: float | None = Field(default=None, ge=0.0)
    enbs_confidence_interval: tuple[float, float] | None = None
    estimator_provenance: Mapping[str, JsonValue] = Field(default_factory=dict)

    @field_serializer("estimator_provenance")
    def serialize_estimator_provenance(self, value: object) -> object:
        """Restore the JSON mapping shape for canonical serialization."""
        return thaw_json(value)

    @model_validator(mode="after")
    def validate_interval(self) -> Self:
        """Reject inverted ENBS intervals."""
        interval = self.enbs_confidence_interval
        if interval is not None and interval[0] > interval[1]:
            raise ValueError("ENBS confidence interval lower bound exceeds upper bound")
        return self


class CossCurvePointV1(ContractModel):
    """One returned point on the complete signed-ENBS design curve."""

    design_id: Identifier
    sample_size: int = Field(ge=0)
    evsi: float
    research_cost: float = Field(ge=0.0)
    enbs: float
    feasible: bool
    feasibility_codes: tuple[Identifier, ...] = ()
    enbs_standard_error: float | None = Field(default=None, ge=0.0)
    enbs_confidence_interval: tuple[float, float] | None = None


class SelectionUncertaintyV1(ContractModel):
    """Descriptive uncertainty in the point-estimate COSS selection."""

    method: SelectionUncertaintyMethod = "unavailable"
    replicate_count: int | None = Field(default=None, ge=1)
    probability_by_design: Mapping[Identifier, float] | None = None
    confidence_set_design_ids: tuple[Identifier, ...] = ()

    @field_serializer("probability_by_design")
    def serialize_probabilities(self, value: object) -> object:
        """Restore the JSON mapping shape for canonical serialization."""
        return thaw_json(value)

    @model_validator(mode="after")
    def validate_probabilities(self) -> Self:
        """Require finite closed-unit-interval selection probabilities."""
        if self.probability_by_design is not None:
            for probability in self.probability_by_design.values():
                if not 0.0 <= probability <= 1.0:
                    raise ValueError("selection probabilities must be in [0, 1]")
        if self.method == "unavailable" and (
            self.replicate_count is not None
            or self.probability_by_design is not None
            or self.confidence_set_design_ids
        ):
            raise ValueError("unavailable uncertainty cannot carry estimates")
        return self


class CossPlotDataV1(ContractModel):
    """Backend-independent aligned vectors for accessible COSS rendering."""

    design_ids: tuple[Identifier, ...]
    sample_sizes: tuple[int, ...]
    evsi: tuple[float, ...]
    research_cost: tuple[float, ...]
    enbs: tuple[float, ...]
    feasible: tuple[bool, ...]
    enbs_lower: tuple[float | None, ...]
    enbs_upper: tuple[float | None, ...]
    optimal_design_id: Identifier | None = None
    tied_optimal_design_ids: tuple[Identifier, ...] = ()
    boundary_state: BoundaryState

    @model_validator(mode="after")
    def validate_alignment(self) -> Self:
        """Require every plotting vector to align one-to-one by design."""
        expected = len(self.design_ids)
        vectors = (
            self.sample_sizes,
            self.evsi,
            self.research_cost,
            self.enbs,
            self.feasible,
            self.enbs_lower,
            self.enbs_upper,
        )
        if any(len(values) != expected for values in vectors):
            raise ValueError("COSS plotting vectors must have equal lengths")
        return self


class CossResultV1(ContractModel):
    """Versioned governed result for a finite Curve of Optimal Sample Size."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    method: Literal["coss"] = "coss"
    estimator: Identifier
    context: StudyDesignContextV1
    evaluated_designs: tuple[CossCurvePointV1, ...]
    feasible_sample_sizes: tuple[int, ...]
    declared_feasible_range: FeasibleDesignRangeV1 | None = None
    tie_policy: TiePolicy
    absolute_tolerance: float = Field(ge=0.0)
    relative_tolerance: float = Field(ge=0.0)
    tied_optimal_design_ids: tuple[Identifier, ...] = ()
    optimal_design_id: Identifier | None = None
    optimal_sample_size: int | None = Field(default=None, ge=0)
    maximum_enbs: float | None = None
    boundary_state: BoundaryState
    selection_uncertainty: SelectionUncertaintyV1
    plot_data: CossPlotDataV1
    diagnostics: tuple[Identifier, ...] = ()


class InformationValueInputV1(ContractModel):
    """One finite information value bound to its interpretation context."""

    value: float
    context: StudyDesignContextV1


class InformationEfficiencyResultV1(ContractModel):
    """Dimensionless EVSI/EVPI diagnostic with explicit undefined state."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    method: Literal["evsi_evpi_efficiency"] = "evsi_evpi_efficiency"
    estimator: Identifier
    context: StudyDesignContextV1
    evsi: float
    evpi: float
    ratio: float | None
    percentage: float | None
    status: EfficiencyStatus
    absolute_tolerance: float = Field(ge=0.0)
    relative_tolerance: float = Field(ge=0.0)
    bound_tolerance: float = Field(ge=0.0)
    diagnostics: tuple[Identifier, ...] = ()


__all__ = [
    "BoundaryState",
    "CossCurvePointV1",
    "CossPlotDataV1",
    "CossResultV1",
    "EfficiencyStatus",
    "FeasibleDesignRangeV1",
    "InformationEfficiencyResultV1",
    "InformationValueInputV1",
    "SelectionUncertaintyV1",
    "StudyDesignContextV1",
    "StudyDesignPointInputV1",
    "TiePolicy",
]
