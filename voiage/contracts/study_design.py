"""Versioned contracts for experimental study-design efficiency methods."""

from __future__ import annotations

from collections.abc import Mapping  # noqa: TC003 - Pydantic resolves annotations
from math import isclose
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
EnumerationScope = Literal["complete_feasible_set", "evaluated_set_only"]
CommissioningStatus = Literal[
    "recommend_commission", "do_not_commission", "indifferent", "no_feasible_design"
]
BoundarySensitivity = Literal[
    "complete_enumeration",
    "requires_evaluated_set_expansion",
    "no_boundary_signal",
    "no_feasible_design",
]
SelectionUncertaintyMethod = Literal[
    "unavailable",
    "analytic",
    "monte_carlo",
    "bootstrap",
    "joint_monte_carlo",
    "joint_bootstrap",
    "externally_supplied",
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
    evsi: float = Field(ge=0.0)
    research_cost: float = Field(ge=0.0)
    enbs: float
    feasible: bool
    feasibility_codes: tuple[Identifier, ...] = ()
    enbs_standard_error: float | None = Field(default=None, ge=0.0)
    enbs_confidence_interval: tuple[float, float] | None = None
    estimator_provenance: Mapping[str, JsonValue] = Field(default_factory=dict)

    @field_serializer("estimator_provenance")
    def serialize_estimator_provenance(self, value: object) -> object:
        """Restore the JSON mapping shape for canonical serialization."""
        return thaw_json(value)

    @model_validator(mode="after")
    def validate_scientific_values(self) -> Self:
        """Require signed subtraction and an ordered uncertainty interval."""
        if not isclose(
            self.enbs, self.evsi - self.research_cost, rel_tol=0.0, abs_tol=0.0
        ):
            raise ValueError("enbs must equal evsi minus research_cost")
        interval = self.enbs_confidence_interval
        if interval is not None and interval[0] > interval[1]:
            raise ValueError("ENBS confidence interval lower bound exceeds upper bound")
        return self


class SelectionUncertaintyV1(ContractModel):
    """Descriptive uncertainty in the point-estimate COSS selection."""

    method: SelectionUncertaintyMethod = "unavailable"
    replicate_count: int | None = Field(default=None, ge=1)
    probability_by_design: Mapping[Identifier, float] | None = None
    confidence_set_design_ids: tuple[Identifier, ...] = ()
    replicate_design_ids: tuple[Identifier, ...] = ()
    selection_count_by_design: Mapping[Identifier, int] | None = None
    joint_replicate_digest: Identifier | None = None
    replay_artifact: Identifier | None = None
    near_tie_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    expected_selection_regret: float | None = Field(default=None, ge=0.0)
    winner_optimism: float | None = None
    mean_selected_design_enbs: float | None = None
    calibration_status: Literal["not_assessed", "joint_replicate_empirical"] = (
        "not_assessed"
    )

    @field_serializer("probability_by_design")
    def serialize_probabilities(self, value: object) -> object:
        """Restore the JSON mapping shape for canonical serialization."""
        return thaw_json(value)

    @field_serializer("selection_count_by_design")
    def serialize_selection_counts(self, value: object) -> object:
        """Restore the JSON mapping shape for canonical serialization."""
        return thaw_json(value)

    @model_validator(mode="after")
    def validate_probabilities(self) -> Self:
        """Require finite closed-unit-interval selection probabilities."""
        if self.probability_by_design is not None:
            for probability in self.probability_by_design.values():
                if not 0.0 <= probability <= 1.0:
                    raise ValueError("selection probabilities must be in [0, 1]")
        simulation_methods = {
            "bootstrap",
            "monte_carlo",
            "joint_bootstrap",
            "joint_monte_carlo",
        }
        if self.method in simulation_methods and self.replicate_count is None:
            raise ValueError("simulation-based uncertainty requires replicate_count")
        joint_fields = (
            self.replicate_design_ids,
            self.selection_count_by_design,
            self.joint_replicate_digest,
            self.replay_artifact,
            self.near_tie_probability,
            self.expected_selection_regret,
            self.winner_optimism,
            self.mean_selected_design_enbs,
        )
        if self.method in {"joint_bootstrap", "joint_monte_carlo"}:
            if any(value in (None, ()) for value in joint_fields):
                raise ValueError("joint replicate uncertainty requires replay metadata")
            if self.calibration_status != "joint_replicate_empirical":
                raise ValueError(
                    "joint replicate uncertainty requires calibration status"
                )
            if (
                self.selection_count_by_design is None
            ):  # pragma: no cover - joint_fields
                raise ValueError(
                    "joint replicate uncertainty requires selection counts"
                )
            if any(value < 0 for value in self.selection_count_by_design.values()):
                raise ValueError("selection counts must be non-negative")
            if sum(self.selection_count_by_design.values()) != self.replicate_count:
                raise ValueError("selection counts must sum to replicate_count")
        elif any(value not in (None, ()) for value in joint_fields):
            raise ValueError("joint replay metadata requires a joint replicate method")
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
    enumeration_scope: EnumerationScope
    feasible_sample_sizes: tuple[int, ...]
    declared_feasible_range: FeasibleDesignRangeV1 | None = None
    tie_policy: TiePolicy
    absolute_tolerance: float = Field(ge=0.0)
    relative_tolerance: float = Field(ge=0.0)
    tied_optimal_design_ids: tuple[Identifier, ...] = ()
    optimal_design_id: Identifier | None = None
    best_evaluated_design_id: Identifier | None = None
    optimal_sample_size: int | None = Field(default=None, ge=0)
    maximum_enbs: float | None = None
    no_study_enbs: float
    commissioning_status: CommissioningStatus
    recommended_design_id: Identifier | None = None
    economic_viability: bool
    regret_if_no_study: float = Field(ge=0.0)
    boundary_state: BoundaryState
    boundary_sensitivity: BoundarySensitivity
    selection_uncertainty: SelectionUncertaintyV1
    plot_data: CossPlotDataV1
    diagnostics: tuple[Identifier, ...] = ()
    estimator_provenance: Mapping[str, JsonValue] = Field(default_factory=dict)

    @field_serializer("estimator_provenance")
    def serialize_estimator_provenance(self, value: object) -> object:
        """Restore the JSON mapping shape for canonical serialization."""
        return thaw_json(value)

    @model_validator(mode="after")
    def validate_result_relations(self) -> Self:
        """Reject internally inconsistent or corrupted result envelopes."""
        if not self.evaluated_designs:
            raise ValueError("evaluated_designs must not be empty")
        ids = tuple(point.design_id for point in self.evaluated_designs)
        if len(set(ids)) != len(ids):
            raise ValueError("evaluated design_id values must be unique")
        feasible = tuple(point for point in self.evaluated_designs if point.feasible)
        expected_sizes = tuple(sorted({point.sample_size for point in feasible}))
        if self.feasible_sample_sizes != expected_sizes:
            raise ValueError("feasible_sample_sizes disagree with evaluated designs")
        by_id = {point.design_id: point for point in self.evaluated_designs}
        feasible_ids = {point.design_id for point in feasible}
        if any(
            item not in by_id or not by_id[item].feasible
            for item in self.tied_optimal_design_ids
        ):
            raise ValueError("tied optimum must reference feasible evaluated designs")
        uncertainty = self.selection_uncertainty
        if any(
            item not in feasible_ids for item in uncertainty.confidence_set_design_ids
        ):
            raise ValueError("selection confidence set must reference feasible designs")
        if uncertainty.probability_by_design is not None:
            probability_ids = set(uncertainty.probability_by_design)
            if any(item not in by_id for item in uncertainty.probability_by_design):
                raise ValueError("selection probabilities reference unknown designs")
            if any(
                probability > 0.0 and item not in feasible_ids
                for item, probability in uncertainty.probability_by_design.items()
            ):
                raise ValueError(
                    "positive selection probability requires a feasible design"
                )
            probability_tolerance = self.absolute_tolerance + self.relative_tolerance
            probability_total = sum(uncertainty.probability_by_design.values())
            if feasible_ids <= probability_ids:
                if not isclose(
                    probability_total,
                    1.0,
                    rel_tol=0.0,
                    abs_tol=probability_tolerance,
                ):
                    raise ValueError(
                        "complete selection probability map must sum to one"
                    )
            elif probability_total > 1.0 + probability_tolerance:
                raise ValueError("selection probability mass must not exceed one")
        if not feasible:
            if (
                any(
                    value is not None
                    for value in (
                        self.optimal_design_id,
                        self.optimal_sample_size,
                        self.maximum_enbs,
                    )
                )
                or self.tied_optimal_design_ids
                or self.boundary_state != "none"
            ):
                raise ValueError(
                    "a result without feasible designs cannot have an optimum"
                )
            if self.best_evaluated_design_id is not None:
                raise ValueError("best_evaluated_design_id requires a feasible design")
        else:
            maximum = max(point.enbs for point in feasible)
            if self.maximum_enbs is None or not isclose(
                self.maximum_enbs, maximum, rel_tol=0.0, abs_tol=0.0
            ):
                raise ValueError("maximum_enbs disagrees with evaluated designs")
            tolerance = self.absolute_tolerance + self.relative_tolerance * max(
                abs(maximum), 1.0
            )
            expected_ties = tuple(
                point.design_id
                for point in feasible
                if maximum - point.enbs <= tolerance
            )
            if self.tied_optimal_design_ids != expected_ties:
                raise ValueError("tied optimum set disagrees with tolerance policy")
            if self.optimal_design_id not in expected_ties:
                raise ValueError(
                    "optimal_design_id must belong to the tied optimum set"
                )
            optimum = by_id[self.optimal_design_id]
            if self.optimal_sample_size != optimum.sample_size:
                raise ValueError("optimal_sample_size disagrees with optimal_design_id")
            if self.tie_policy == "first_declared":
                expected_id = expected_ties[0]
            elif self.tie_policy == "smallest_sample_size":
                expected_id = min(
                    expected_ties,
                    key=lambda item: (by_id[item].sample_size, item),
                )
            else:
                largest_size = max(by_id[item].sample_size for item in expected_ties)
                expected_id = min(
                    item
                    for item in expected_ties
                    if by_id[item].sample_size == largest_size
                )
            if self.optimal_design_id != expected_id:
                raise ValueError("optimal_design_id disagrees with tie policy")
            if self.best_evaluated_design_id != self.optimal_design_id:
                raise ValueError(
                    "best_evaluated_design_id must preserve the curve argmax"
                )
            low, high = expected_sizes[0], expected_sizes[-1]
            expected_boundary = (
                "both"
                if low == high
                else "lower"
                if optimum.sample_size == low
                else "upper"
                if optimum.sample_size == high
                else "interior"
            )
            if self.boundary_state != expected_boundary:
                raise ValueError("boundary_state disagrees with feasible designs")
        comparison_tolerance = self.absolute_tolerance + self.relative_tolerance * max(
            abs(self.no_study_enbs), abs(self.maximum_enbs or 0.0), 1.0
        )
        if self.maximum_enbs is None:
            expected_commissioning: CommissioningStatus = "no_feasible_design"
            expected_recommendation = None
            expected_regret = 0.0
            expected_viability = False
            expected_sensitivity: BoundarySensitivity = "no_feasible_design"
        else:
            difference = self.maximum_enbs - self.no_study_enbs
            if difference > comparison_tolerance:
                expected_commissioning = "recommend_commission"
                expected_recommendation = self.optimal_design_id
                expected_viability = True
            elif difference < -comparison_tolerance:
                expected_commissioning = "do_not_commission"
                expected_recommendation = None
                expected_viability = False
            else:
                expected_commissioning = "indifferent"
                expected_recommendation = None
                expected_viability = False
            expected_regret = max(difference, 0.0)
            expected_sensitivity = (
                "complete_enumeration"
                if self.enumeration_scope == "complete_feasible_set"
                else "requires_evaluated_set_expansion"
                if self.boundary_state in {"lower", "upper", "both"}
                else "no_boundary_signal"
            )
        if (
            self.commissioning_status != expected_commissioning
            or self.recommended_design_id != expected_recommendation
            or self.economic_viability is not expected_viability
            or not isclose(
                self.regret_if_no_study,
                expected_regret,
                rel_tol=0.0,
                abs_tol=0.0,
            )
        ):
            raise ValueError(
                "commissioning recommendation disagrees with no-study comparison"
            )
        if self.boundary_sensitivity != expected_sensitivity:
            raise ValueError("boundary_sensitivity disagrees with enumeration scope")
        expected_plot = (
            ids,
            tuple(point.sample_size for point in self.evaluated_designs),
            tuple(point.evsi for point in self.evaluated_designs),
            tuple(point.research_cost for point in self.evaluated_designs),
            tuple(point.enbs for point in self.evaluated_designs),
            tuple(point.feasible for point in self.evaluated_designs),
            tuple(
                None
                if point.enbs_confidence_interval is None
                else point.enbs_confidence_interval[0]
                for point in self.evaluated_designs
            ),
            tuple(
                None
                if point.enbs_confidence_interval is None
                else point.enbs_confidence_interval[1]
                for point in self.evaluated_designs
            ),
        )
        actual_plot = (
            self.plot_data.design_ids,
            self.plot_data.sample_sizes,
            self.plot_data.evsi,
            self.plot_data.research_cost,
            self.plot_data.enbs,
            self.plot_data.feasible,
            self.plot_data.enbs_lower,
            self.plot_data.enbs_upper,
        )
        if actual_plot != expected_plot or (
            self.plot_data.optimal_design_id,
            self.plot_data.tied_optimal_design_ids,
            self.plot_data.boundary_state,
        ) != (
            self.optimal_design_id,
            self.tied_optimal_design_ids,
            self.boundary_state,
        ):
            raise ValueError("plot_data disagrees with the COSS result")
        if not self.estimator_provenance:
            raise ValueError("estimator_provenance must not be empty")
        return self


class InformationValueInputV1(ContractModel):
    """One finite information value bound to its interpretation context."""

    value: float
    context: StudyDesignContextV1


class CossRequestV1(ContractModel):
    """Portable request envelope for an enumerated COSS calculation."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    context: StudyDesignContextV1
    designs: tuple[StudyDesignPointInputV1, ...] = Field(min_length=1)
    declared_feasible_range: FeasibleDesignRangeV1 | None = None
    tie_policy: TiePolicy = "smallest_sample_size"
    absolute_tolerance: float = Field(default=1e-10, ge=0.0)
    relative_tolerance: float = Field(default=1e-8, ge=0.0)
    selection_uncertainty: SelectionUncertaintyV1 | None = None
    enumeration_scope: EnumerationScope = "evaluated_set_only"
    no_study_enbs: float = 0.0
    joint_enbs_replicates: tuple[tuple[float, ...], ...] | None = None
    joint_replicate_method: Literal["joint_bootstrap", "joint_monte_carlo"] = (
        "joint_bootstrap"
    )
    replay_artifact: Identifier | None = None

    @model_validator(mode="after")
    def validate_design_identity(self) -> Self:
        """Reject duplicate design identifiers at the portable boundary."""
        design_ids = tuple(item.design_id for item in self.designs)
        if len(set(design_ids)) != len(design_ids):
            raise ValueError("design_id values must be unique")
        if (
            self.selection_uncertainty is not None
            and self.joint_enbs_replicates is not None
        ):
            raise ValueError(
                "selection_uncertainty and joint_enbs_replicates are mutually exclusive"
            )
        if self.joint_enbs_replicates is not None:
            if self.replay_artifact is None:
                raise ValueError("joint ENBS replicates require replay_artifact")
            if not self.joint_enbs_replicates:
                raise ValueError("joint ENBS replicates must not be empty")
            if any(len(row) != len(self.designs) for row in self.joint_enbs_replicates):
                raise ValueError("joint ENBS replicate rows must align with designs")
        elif self.replay_artifact is not None:
            raise ValueError("replay_artifact requires joint ENBS replicates")
        return self


class InformationEfficiencyRequestV1(ContractModel):
    """Portable request envelope for an EVSI/EVPI efficiency diagnostic."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    evsi: InformationValueInputV1
    evpi: InformationValueInputV1
    absolute_tolerance: float = Field(default=1e-10, ge=0.0)
    relative_tolerance: float = Field(default=1e-8, ge=0.0)
    paired_evsi_replicates: tuple[float, ...] | None = None
    paired_evpi_replicates: tuple[float, ...] | None = None
    replay_artifact: Identifier | None = None

    @model_validator(mode="after")
    def validate_commensurability(self) -> Self:
        """Require a common economic interpretation before division."""
        if (
            self.evsi.context.commensurability_key()
            != self.evpi.context.commensurability_key()
        ):
            raise ValueError("EVSI and EVPI must be commensurate")
        paired = (self.paired_evsi_replicates, self.paired_evpi_replicates)
        if (paired[0] is None) != (paired[1] is None):
            raise ValueError("EVSI and EVPI uncertainty replicates must be paired")
        if paired[0] is not None:
            if paired[1] is None:  # pragma: no cover - paired-presence check above
                raise ValueError("EVSI and EVPI uncertainty replicates must be paired")
            if len(paired[0]) != len(paired[1]) or len(paired[0]) < 2:
                raise ValueError(
                    "at least two paired efficiency replicates are required"
                )
            if self.replay_artifact is None:
                raise ValueError("paired efficiency replicates require replay_artifact")
        elif self.replay_artifact is not None:
            raise ValueError("replay_artifact requires paired efficiency replicates")
        return self


class InformationEfficiencyUncertaintyV1(ContractModel):
    """Replayable uncertainty summary for a paired efficiency estimator."""

    method: Literal["paired_empirical"] = "paired_empirical"
    replicate_count: int = Field(ge=2)
    mean_ratio: float
    standard_error: float = Field(ge=0.0)
    confidence_interval: tuple[float, float]
    estimated_bias: float
    point_ratio_in_interval: bool
    paired_replicate_digest: Identifier
    replay_artifact: Identifier
    calibration_status: Literal["paired_replicate_empirical"] = (
        "paired_replicate_empirical"
    )

    @model_validator(mode="after")
    def validate_interval(self) -> Self:
        """Require an ordered interval consistent with its containment flag."""
        lower, upper = self.confidence_interval
        if lower > upper:
            raise ValueError("efficiency uncertainty interval is inverted")
        return self


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
    uncertainty: InformationEfficiencyUncertaintyV1 | None = None
    diagnostics: tuple[Identifier, ...] = ()

    @model_validator(mode="after")
    def validate_result_relations(self) -> Self:
        """Reject inconsistent ratios, percentages and status labels."""
        expected_tolerance = self.absolute_tolerance + self.relative_tolerance * max(
            abs(self.evpi), 1.0
        )
        if not isclose(
            self.bound_tolerance, expected_tolerance, rel_tol=0.0, abs_tol=0.0
        ):
            raise ValueError("bound_tolerance disagrees with EVPI and tolerances")
        if self.status == "undefined_zero_evpi":
            if (
                self.ratio is not None
                or self.percentage is not None
                or abs(self.evpi) > self.bound_tolerance
            ):
                raise ValueError("undefined_zero_evpi fields are inconsistent")
            if abs(self.evsi) > self.bound_tolerance:
                raise ValueError("zero EVPI requires EVSI to be numerically zero")
            if self.uncertainty is not None:
                raise ValueError("undefined efficiency cannot carry uncertainty")
            return self
        if (
            self.evpi <= self.bound_tolerance
            or self.ratio is None
            or self.percentage is None
        ):
            raise ValueError(
                "defined efficiency requires positive non-zero EVPI and ratio fields"
            )
        if self.evsi < -self.bound_tolerance:
            raise ValueError("EVSI is materially below zero")
        if self.evsi > self.evpi + self.bound_tolerance:
            raise ValueError("EVSI materially exceeds EVPI")
        expected_ratio = self.evsi / self.evpi
        if not isclose(
            self.ratio, expected_ratio, rel_tol=1e-15, abs_tol=0.0
        ) or not isclose(
            self.percentage, 100.0 * expected_ratio, rel_tol=1e-15, abs_tol=0.0
        ):
            raise ValueError("ratio or percentage disagrees with EVSI and EVPI")
        expected_status = (
            "below_zero_within_tolerance"
            if self.evsi < 0.0
            else "above_one_within_tolerance"
            if self.evsi > self.evpi
            else "within_bounds"
        )
        if self.status != expected_status:
            raise ValueError("status disagrees with EVSI and EVPI")
        if self.uncertainty is not None:
            lower, upper = self.uncertainty.confidence_interval
            expected_contains = lower <= self.ratio <= upper
            if self.uncertainty.point_ratio_in_interval is not expected_contains:
                raise ValueError(
                    "efficiency uncertainty containment flag disagrees with ratio"
                )
            if not isclose(
                self.uncertainty.estimated_bias,
                self.uncertainty.mean_ratio - self.ratio,
                rel_tol=1e-15,
                abs_tol=1e-15,
            ):
                raise ValueError("efficiency uncertainty bias disagrees with ratio")
        return self


__all__ = [
    "BoundaryState",
    "CossCurvePointV1",
    "CossPlotDataV1",
    "CossResultV1",
    "EfficiencyStatus",
    "FeasibleDesignRangeV1",
    "InformationEfficiencyResultV1",
    "InformationEfficiencyUncertaintyV1",
    "InformationValueInputV1",
    "SelectionUncertaintyV1",
    "StudyDesignContextV1",
    "StudyDesignPointInputV1",
    "TiePolicy",
]
