"""Experimental governed COSS and EVSI/EVPI study-efficiency façade."""

from __future__ import annotations

from itertools import pairwise
from math import isfinite
from typing import TYPE_CHECKING, cast

from pydantic import ValidationError

if TYPE_CHECKING:
    from collections.abc import Sequence

from voiage import _runtime
from voiage.contracts.study_design import (
    BoundaryState,
    CossCurvePointV1,
    CossPlotDataV1,
    CossResultV1,
    EfficiencyStatus,
    FeasibleDesignRangeV1,
    InformationEfficiencyResultV1,
    InformationValueInputV1,
    SelectionUncertaintyV1,
    StudyDesignContextV1,
    StudyDesignPointInputV1,
    TiePolicy,
)
from voiage.exceptions import InputError

_DEFAULT_ATOL = 1e-10
_DEFAULT_RTOL = 1e-8


def _require_contract_version(native: dict[str, object]) -> None:
    if native.get("contract_version") != "1.0.0":
        raise KeyError("contract_version")


def _native_indices(value: object, *, field: str, size: int) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise TypeError(f"{field} must be a list")
    items = cast("list[object]", value)
    if any(type(index) is not int or index < 0 or index >= size for index in items):
        raise ValueError(f"{field} contains an invalid index")
    indices = tuple(cast("int", index) for index in items)
    if len(set(indices)) != len(indices):
        raise ValueError(f"{field} contains duplicate indices")
    return indices


def _expected_boundary(
    optimal_index: int,
    feasible_indices: tuple[int, ...],
    designs: tuple[StudyDesignPointInputV1, ...],
) -> BoundaryState:
    sizes = {designs[index].sample_size for index in feasible_indices}
    if len(sizes) == 1:
        return "both"
    selected = designs[optimal_index].sample_size
    if selected == min(sizes):
        return "lower"
    if selected == max(sizes):
        return "upper"
    return "interior"


def _parse_native_coss(
    native: dict[str, object],
    designs: tuple[StudyDesignPointInputV1, ...],
    tie_policy: TiePolicy,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> tuple[
    tuple[float, ...], tuple[int, ...], int | None, float | None, BoundaryState, str
]:
    """Validate the complete untyped native envelope before binding metadata."""
    _require_contract_version(native)
    size = len(designs)
    raw_enbs = native["enbs"]
    if not isinstance(raw_enbs, list):
        raise TypeError("enbs must be a list")
    raw_enbs = cast("list[object]", raw_enbs)
    if len(raw_enbs) != size:
        raise ValueError("enbs vector length mismatch")
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in raw_enbs
    ):
        raise TypeError("enbs values must be numeric")
    enbs = tuple(float(value) for value in cast("list[float]", raw_enbs))
    if any(not isfinite(value) for value in enbs):
        raise ValueError("enbs values must be finite")
    expected_enbs = tuple(item.evsi - item.research_cost for item in designs)
    if enbs != expected_enbs:
        raise ValueError("enbs values disagree with signed subtraction")
    feasible_indices = _native_indices(
        native["feasible_indices"], field="feasible_indices", size=size
    )
    expected_feasible = tuple(
        index for index, item in enumerate(designs) if item.feasible
    )
    if feasible_indices != expected_feasible:
        raise ValueError("feasible_indices disagree with design metadata")
    tied_indices = _native_indices(
        native["tied_indices"], field="tied_indices", size=size
    )
    if any(index not in feasible_indices for index in tied_indices):
        raise ValueError("tied_indices contain an infeasible design")
    maximum_raw = native["maximum_enbs"]
    optimum_raw = native["optimal_index"]
    boundary_raw = native["boundary_state"]
    if not isinstance(boundary_raw, str):
        raise TypeError("boundary_state must be a string")
    if not feasible_indices:
        if (
            maximum_raw is not None
            or optimum_raw is not None
            or tied_indices
            or boundary_raw != "none"
        ):
            raise ValueError("native no-feasible result is inconsistent")
        maximum = None
        optimum = None
        boundary: BoundaryState = "none"
    else:
        if isinstance(maximum_raw, bool) or not isinstance(maximum_raw, (int, float)):
            raise TypeError("maximum_enbs must be numeric")
        maximum = float(maximum_raw)
        expected_maximum = max(enbs[index] for index in feasible_indices)
        if not isfinite(maximum) or maximum != expected_maximum:
            raise ValueError("maximum_enbs disagrees with the curve")
        tolerance = absolute_tolerance + relative_tolerance * max(abs(maximum), 1.0)
        expected_tied = tuple(
            index for index in feasible_indices if maximum - enbs[index] <= tolerance
        )
        if tied_indices != expected_tied:
            raise ValueError("tied_indices disagree with the tolerance policy")
        if type(optimum_raw) is not int:
            raise TypeError("optimal_index must be an integer")
        optimum = optimum_raw
        if optimum not in tied_indices:
            raise ValueError("optimal_index is not tied and feasible")
        if tie_policy == "first_declared":
            expected_optimum = tied_indices[0]
        elif tie_policy == "smallest_sample_size":
            expected_optimum = min(
                tied_indices, key=lambda index: (designs[index].sample_size, index)
            )
        else:
            expected_optimum = min(
                tied_indices, key=lambda index: (-designs[index].sample_size, index)
            )
        if optimum != expected_optimum:
            raise ValueError("optimal_index disagrees with tie_policy")
        expected_boundary = _expected_boundary(optimum, feasible_indices, designs)
        if boundary_raw != expected_boundary:
            raise ValueError("boundary_state disagrees with feasible designs")
        boundary = expected_boundary
    estimator_raw = native["estimator"]
    if not isinstance(estimator_raw, str) or not estimator_raw.strip():
        raise TypeError("estimator must be a non-empty string")
    return enbs, tied_indices, optimum, maximum, boundary, estimator_raw


def _normalize_range(
    value: FeasibleDesignRangeV1 | tuple[int, int] | None,
) -> FeasibleDesignRangeV1 | None:
    if value is None or isinstance(value, FeasibleDesignRangeV1):
        return value
    try:
        lower, upper = value
        return FeasibleDesignRangeV1(
            lower_sample_size=lower,
            upper_sample_size=upper,
        )
    except (TypeError, ValueError) as error:
        raise InputError(
            "declared_feasible_range must contain lower and upper sizes"
        ) from error


def _diagnostics(
    designs: tuple[StudyDesignPointInputV1, ...],
    declared_range: FeasibleDesignRangeV1 | None,
    selection_uncertainty: SelectionUncertaintyV1,
) -> tuple[str, ...]:
    diagnostics: list[str] = []
    ordered = sorted(designs, key=lambda item: (item.sample_size, item.design_id))
    if any(right.evsi < left.evsi for left, right in pairwise(ordered)):
        diagnostics.append("non_monotone_evsi")
    feasible_sizes = {item.sample_size for item in designs if item.feasible}
    if declared_range is not None:
        endpoints_match = (
            bool(feasible_sizes)
            and min(feasible_sizes) == declared_range.lower_sample_size
            and max(feasible_sizes) == declared_range.upper_sample_size
        )
        if not endpoints_match or any(
            size < declared_range.lower_sample_size
            or size > declared_range.upper_sample_size
            for size in feasible_sizes
        ):
            diagnostics.append("feasible_range_set_disagreement")
        if any(
            not item.feasible
            and declared_range.lower_sample_size
            <= item.sample_size
            <= declared_range.upper_sample_size
            for item in designs
        ):
            diagnostics.append("infeasible_design_within_declared_range")
        if declared_range.step is not None:
            expected = set(
                range(
                    declared_range.lower_sample_size,
                    declared_range.upper_sample_size + 1,
                    declared_range.step,
                )
            )
            if expected - feasible_sizes:
                diagnostics.append("feasible_set_has_gaps")
    if selection_uncertainty.method == "unavailable":
        diagnostics.append("selection_uncertainty_unavailable")
    if not feasible_sizes:
        diagnostics.append("no_feasible_design")
    if any(
        item.enbs_standard_error is None
        and item.enbs_confidence_interval is None
        and (
            item.evsi_standard_error is not None or item.cost_standard_error is not None
        )
        for item in designs
    ):
        diagnostics.append("enbs_uncertainty_unavailable")
    return tuple(diagnostics)


def _validate_selection_uncertainty(
    uncertainty: SelectionUncertaintyV1,
    design_ids: set[str],
    feasible_design_ids: set[str],
    absolute_tolerance: float,
) -> None:
    if any(item not in design_ids for item in uncertainty.confidence_set_design_ids):
        raise InputError("selection uncertainty references an unknown design_id")
    if any(
        item not in feasible_design_ids
        for item in uncertainty.confidence_set_design_ids
    ):
        raise InputError("selection confidence set references an infeasible design_id")
    probabilities = uncertainty.probability_by_design
    if probabilities is None:
        return
    if any(item not in design_ids for item in probabilities):
        raise InputError("selection probabilities reference an unknown design_id")
    if any(
        probability > 0.0 and item not in feasible_design_ids
        for item, probability in probabilities.items()
    ):
        raise InputError(
            "positive selection probability references an infeasible design_id"
        )
    if set(probabilities) == feasible_design_ids and abs(
        sum(probabilities.values()) - 1.0
    ) > max(absolute_tolerance, 1e-12):
        raise InputError("complete selection probabilities must sum to one")


def calculate_coss(
    *,
    context: StudyDesignContextV1,
    designs: Sequence[StudyDesignPointInputV1],
    declared_feasible_range: FeasibleDesignRangeV1 | tuple[int, int] | None = None,
    tie_policy: TiePolicy = "smallest_sample_size",
    absolute_tolerance: float = _DEFAULT_ATOL,
    relative_tolerance: float = _DEFAULT_RTOL,
    selection_uncertainty: SelectionUncertaintyV1 | None = None,
) -> CossResultV1:
    """Evaluate a finite COSS curve using the Rust signed-ENBS kernel."""
    design_tuple = tuple(designs)
    if not design_tuple:
        raise InputError("at least one evaluated design is required")
    if any(
        not isinstance(item, StudyDesignPointInputV1)  # pyright: ignore[reportUnnecessaryIsInstance]
        for item in design_tuple
    ):
        raise InputError("designs must contain StudyDesignPointInputV1 records")
    design_ids = [item.design_id for item in design_tuple]
    if len(set(design_ids)) != len(design_ids):
        raise InputError("design_id values must be unique")
    uncertainty = selection_uncertainty or SelectionUncertaintyV1()
    feasible_ids = {item.design_id for item in design_tuple if item.feasible}
    _validate_selection_uncertainty(
        uncertainty, set(design_ids), feasible_ids, absolute_tolerance
    )
    feasible_range = _normalize_range(declared_feasible_range)

    native = _runtime.compute_coss(
        sample_sizes=[item.sample_size for item in design_tuple],
        evsi_values=[item.evsi for item in design_tuple],
        research_costs=[item.research_cost for item in design_tuple],
        feasible=[item.feasible for item in design_tuple],
        tie_policy=tie_policy,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )
    try:
        (
            enbs_values,
            tied_indices,
            optimal_index,
            maximum_enbs,
            boundary_state,
            estimator,
        ) = _parse_native_coss(
            native,
            design_tuple,
            tie_policy,
            absolute_tolerance,
            relative_tolerance,
        )
    except (IndexError, KeyError, TypeError, ValueError) as error:
        raise InputError(
            "native COSS result violated contract version 1.0.0"
        ) from error

    # The native kernel owns numerical ties. Python binds the declared design-ID
    # secondary key when multiple tied designs share the chosen sample size.
    if optimal_index is not None and tie_policy in {
        "smallest_sample_size",
        "largest_sample_size",
    }:
        chosen_size = design_tuple[optimal_index].sample_size
        same_size = [
            index
            for index in tied_indices
            if design_tuple[index].sample_size == chosen_size
        ]
        optimal_index = min(same_size, key=lambda index: design_tuple[index].design_id)

    curve = tuple(
        CossCurvePointV1(
            design_id=item.design_id,
            sample_size=item.sample_size,
            evsi=item.evsi,
            research_cost=item.research_cost,
            enbs=enbs_values[index],
            feasible=item.feasible,
            feasibility_codes=item.feasibility_codes,
            enbs_standard_error=item.enbs_standard_error,
            enbs_confidence_interval=item.enbs_confidence_interval,
            estimator_provenance=item.estimator_provenance,
        )
        for index, item in enumerate(design_tuple)
    )
    tied_ids = tuple(design_tuple[index].design_id for index in tied_indices)
    optimal_id = (
        None if optimal_index is None else design_tuple[optimal_index].design_id
    )
    optimal_size = (
        None if optimal_index is None else design_tuple[optimal_index].sample_size
    )
    intervals = tuple(item.enbs_confidence_interval for item in design_tuple)
    plot_data = CossPlotDataV1(
        design_ids=tuple(design_ids),
        sample_sizes=tuple(item.sample_size for item in design_tuple),
        evsi=tuple(item.evsi for item in design_tuple),
        research_cost=tuple(item.research_cost for item in design_tuple),
        enbs=enbs_values,
        feasible=tuple(item.feasible for item in design_tuple),
        enbs_lower=tuple(
            None if interval is None else interval[0] for interval in intervals
        ),
        enbs_upper=tuple(
            None if interval is None else interval[1] for interval in intervals
        ),
        optimal_design_id=optimal_id,
        tied_optimal_design_ids=tied_ids,
        boundary_state=boundary_state,
    )
    try:
        return CossResultV1(
            estimator=estimator,
            context=context,
            evaluated_designs=curve,
            feasible_sample_sizes=tuple(
                sorted({item.sample_size for item in design_tuple if item.feasible})
            ),
            declared_feasible_range=feasible_range,
            tie_policy=tie_policy,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
            tied_optimal_design_ids=tied_ids,
            optimal_design_id=optimal_id,
            optimal_sample_size=optimal_size,
            maximum_enbs=maximum_enbs,
            boundary_state=boundary_state,
            selection_uncertainty=uncertainty,
            plot_data=plot_data,
            diagnostics=_diagnostics(design_tuple, feasible_range, uncertainty),
            estimator_provenance={
                "runtime": "rust",
                "kernel": estimator,
                "contract_version": "1.0.0",
            },
        )
    except ValidationError as error:
        raise InputError("COSS result failed scientific contract validation") from error


def evsi_evpi_efficiency(
    *,
    evsi: InformationValueInputV1,
    evpi: InformationValueInputV1,
    absolute_tolerance: float = _DEFAULT_ATOL,
    relative_tolerance: float = _DEFAULT_RTOL,
) -> InformationEfficiencyResultV1:
    """Return the unclamped dimensionless EVSI/EVPI efficiency diagnostic."""
    if evsi.context.commensurability_key() != evpi.context.commensurability_key():
        raise InputError("EVSI and EVPI must be commensurate")
    native = _runtime.compute_evsi_evpi_efficiency(
        evsi.value,
        evpi.value,
        absolute_tolerance,
        relative_tolerance,
    )
    try:
        _require_contract_version(native)
        ratio_raw = native["ratio"]
        ratio = None if ratio_raw is None else float(cast("float", ratio_raw))
        status = cast("EfficiencyStatus", str(native["status"]))
        bound_tolerance = float(cast("float", native["bound_tolerance"]))
    except (KeyError, TypeError, ValueError) as error:
        raise InputError(
            "native efficiency result violated contract version 1.0.0"
        ) from error
    diagnostics = () if status == "within_bounds" else (status,)
    if status in {"below_zero_within_tolerance", "above_one_within_tolerance"}:
        diagnostics = (*diagnostics, "ratio_not_clamped")
    try:
        return InformationEfficiencyResultV1(
            estimator="derived_evsi_evpi_ratio",
            context=evsi.context,
            evsi=evsi.value,
            evpi=evpi.value,
            ratio=ratio,
            percentage=None if ratio is None else 100.0 * ratio,
            status=status,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
            bound_tolerance=bound_tolerance,
            diagnostics=diagnostics,
        )
    except ValidationError as error:
        raise InputError(
            "efficiency result failed scientific contract validation"
        ) from error


__all__ = ["calculate_coss", "evsi_evpi_efficiency"]
