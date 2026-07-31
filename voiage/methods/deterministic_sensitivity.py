"""Experimental deterministic sensitivity and scenario analysis."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from itertools import pairwise
from typing import Any, Literal, cast

import numpy as np

from voiage.deterministic_sensitivity_contract import (
    validate_deterministic_sensitivity_specification,
)
from voiage.exceptions import raise_input_error

Direction = Literal["maximize", "minimize"]


@dataclass(frozen=True)
class DsaPoint:
    """One fully evaluated deterministic coordinate vector."""

    record_id: str
    coordinates: dict[str, float]
    alternative_outputs: dict[str, float]
    direction_normalized_increments: dict[str, float]
    optimal_alternatives: list[str]
    selected_alternative: str
    optimal_metric: float


@dataclass(frozen=True)
class DsaParameterSummary:
    """Tornado range and extrema for one varied coordinate."""

    parameter_name: str
    parameter_unit: str
    ranking_metric: str
    minimum_metric: float
    maximum_metric: float
    evaluated_range: float
    minimum_coordinates: list[float]
    maximum_coordinates: list[float]
    endpoint_range: float
    interior_extremum_observed: bool
    rank: int


@dataclass(frozen=True)
class DsaSwitchInterval:
    """Observed exact tie/plateau or adjacent decision-switch bracket."""

    parameter_name: str
    parameter_unit: str
    status: Literal["exact", "plateau", "bracket"]
    lower_coordinate: float
    upper_coordinate: float
    lower_optimal_alternatives: list[str]
    upper_optimal_alternatives: list[str]
    interpolation: str = "not-performed"


@dataclass(frozen=True)
class DeterministicSensitivityResult:
    """Complete experimental DSA result envelope."""

    baseline_point: DsaPoint
    one_way_points: dict[str, list[DsaPoint]]
    two_way_points: dict[str, list[DsaPoint]]
    scenario_points: dict[str, DsaPoint]
    parameter_summaries: list[DsaParameterSummary]
    switch_intervals: list[DsaSwitchInterval]
    parameter_units: dict[str, str]
    direction: Direction
    output_unit: str
    analysis_id: str = "callback-analysis"
    two_way_semantics: dict[str, str] = field(default_factory=dict)
    scenario_assumptions: dict[str, str] = field(default_factory=dict)
    absolute_tolerance: float = 1e-12
    relative_tolerance: float = 1e-12
    evaluated_record_count: int = 0
    diagnostics: dict[str, object] = field(default_factory=dict)
    reporting: dict[str, object] = field(default_factory=dict)

    def to_contract_dict(self) -> dict[str, object]:
        """Return the strict deterministic-sensitivity-result-v1 payload."""

        def point_payload(point: DsaPoint) -> dict[str, object]:
            return {
                "record_id": point.record_id,
                "coordinates": [
                    {"parameter_name": name, "value": value}
                    for name, value in point.coordinates.items()
                ],
                "alternative_outputs": [
                    {"alternative_name": name, "value": value}
                    for name, value in point.alternative_outputs.items()
                ],
                "direction_normalized_increments": [
                    {"alternative_name": name, "value": value}
                    for name, value in point.direction_normalized_increments.items()
                ],
                "optimal_alternatives": point.optimal_alternatives,
                "selected_alternative": point.selected_alternative,
                "optimal_metric": point.optimal_metric,
            }

        return {
            "schema_version": "deterministic-sensitivity-result-v1",
            "analysis_type": "deterministic_sensitivity_analysis",
            "method_maturity": "experimental",
            "analysis_id": self.analysis_id,
            "direction": self.direction,
            "output_unit": self.output_unit,
            "baseline_point": point_payload(self.baseline_point),
            "one_way": [
                {
                    "parameter_name": name,
                    "parameter_unit": self.parameter_units[name],
                    "points": [point_payload(point) for point in points],
                }
                for name, points in self.one_way_points.items()
            ],
            "two_way": [
                {
                    "surface_id": surface_id,
                    "parameter_names": surface_id.split("|"),
                    "feasibility_semantics": self.two_way_semantics[surface_id],
                    "points": [point_payload(point) for point in points],
                }
                for surface_id, points in self.two_way_points.items()
            ],
            "scenarios": [
                {
                    "scenario_id": scenario_id,
                    "structural_assumption": self.scenario_assumptions[scenario_id],
                    "point": point_payload(point),
                }
                for scenario_id, point in self.scenario_points.items()
            ],
            "parameter_summaries": [
                {
                    "parameter_name": item.parameter_name,
                    "parameter_unit": item.parameter_unit,
                    "ranking_metric": item.ranking_metric,
                    "minimum_metric": item.minimum_metric,
                    "maximum_metric": item.maximum_metric,
                    "evaluated_range": item.evaluated_range,
                    "minimum_coordinates": item.minimum_coordinates,
                    "maximum_coordinates": item.maximum_coordinates,
                    "endpoint_range": item.endpoint_range,
                    "interior_extremum_observed": item.interior_extremum_observed,
                    "rank": item.rank,
                }
                for item in self.parameter_summaries
            ],
            "switch_intervals": [
                {
                    "parameter_name": item.parameter_name,
                    "parameter_unit": item.parameter_unit,
                    "status": item.status,
                    "lower_coordinate": item.lower_coordinate,
                    "upper_coordinate": item.upper_coordinate,
                    "lower_optimal_alternatives": item.lower_optimal_alternatives,
                    "upper_optimal_alternatives": item.upper_optimal_alternatives,
                    "interpolation": item.interpolation,
                }
                for item in self.switch_intervals
            ],
            "tie_tolerance": {
                "absolute": self.absolute_tolerance,
                "relative": self.relative_tolerance,
                "representative_policy": "canonical-lexicographic",
            },
            "diagnostics": self.diagnostics,
            "reporting": self.reporting,
        }


_Evaluator = Callable[[str, str, Mapping[str, float]], tuple[str, Mapping[str, float]]]


def _validated_inputs(
    baseline_parameters: Mapping[str, float],
    parameter_grids: Mapping[str, Sequence[float]],
    parameter_units: Mapping[str, str],
    alternative_names: Sequence[str],
    output_unit: str,
    direction: str,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> tuple[
    dict[str, float], dict[str, list[float]], dict[str, str], list[str], Direction
]:
    baseline = {str(name): float(value) for name, value in baseline_parameters.items()}
    if not baseline or not all(np.isfinite(value) for value in baseline.values()):
        raise_input_error("DSA requires a non-empty finite baseline.")
    if set(parameter_grids) != set(baseline):
        raise_input_error("parameter_grids keys must exactly match the baseline.")
    if set(parameter_units) != set(baseline):
        raise_input_error("parameter_units keys must exactly match the baseline.")
    grids: dict[str, list[float]] = {}
    for name in baseline:
        values = [float(value) for value in parameter_grids[name]]
        if not values:
            raise_input_error("Each DSA parameter requires a non-empty grid.")
        if not all(np.isfinite(value) for value in values):
            raise_input_error("DSA grids must contain only finite values.")
        if len(set(values)) != len(values):
            raise_input_error("DSA parameter grids must not contain duplicate values.")
        if any(right <= left for left, right in pairwise(values)):
            raise_input_error("DSA parameter grids must be strictly increasing.")
        grids[name] = values
    units = {str(name): str(unit).strip() for name, unit in parameter_units.items()}
    if not all(units.values()):
        raise_input_error("Every DSA parameter requires a non-empty unit.")
    alternatives = [str(name) for name in alternative_names]
    if not alternatives or len(set(alternatives)) != len(alternatives):
        raise_input_error("alternative_names must be non-empty and unique.")
    if not isinstance(output_unit, str) or not output_unit.strip():
        raise_input_error("output_unit must be a non-empty comparable unit.")
    if direction not in {"maximize", "minimize"}:
        raise_input_error("direction must be 'maximize' or 'minimize'.")
    tolerances = np.asarray([absolute_tolerance, relative_tolerance], dtype=float)
    if not np.all(np.isfinite(tolerances)) or np.any(tolerances < 0):
        raise_input_error("Tie tolerances must be finite and non-negative.")
    return baseline, grids, units, alternatives, direction  # type: ignore[return-value]


def _make_point(
    record_id: str,
    coordinates: Mapping[str, float],
    outputs: Mapping[str, float],
    alternatives: Sequence[str],
    baseline_outputs: Mapping[str, float],
    direction: Direction,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> DsaPoint:
    if set(outputs) != set(alternatives):
        raise_input_error("Callback outputs must exactly match alternative_names.")
    values = {name: float(outputs[name]) for name in alternatives}
    if not all(np.isfinite(value) for value in values.values()):
        raise_input_error("Callback must return only finite outputs.")
    optimum = max(values.values()) if direction == "maximize" else min(values.values())
    tied = sorted(
        name
        for name, value in values.items()
        if abs(value - optimum)
        <= absolute_tolerance + relative_tolerance * max(1.0, abs(optimum))
    )
    sign = 1.0 if direction == "maximize" else -1.0
    return DsaPoint(
        record_id=record_id,
        coordinates={name: float(value) for name, value in coordinates.items()},
        alternative_outputs=values,
        direction_normalized_increments={
            name: sign * (values[name] - baseline_outputs[name])
            for name in alternatives
        },
        optimal_alternatives=tied,
        selected_alternative=tied[0],
        optimal_metric=float(optimum),
    )


def _safe_evaluate(
    evaluator: _Evaluator,
    kind: str,
    reference: str,
    coordinates: Mapping[str, float],
) -> tuple[str, Mapping[str, float]]:
    try:
        return evaluator(kind, reference, dict(coordinates))
    except ValueError:
        raise
    except Exception as error:
        return raise_input_error(f"DSA callback failed: {error}")


def _build_result(
    evaluator: _Evaluator,
    *,
    baseline: dict[str, float],
    grids: dict[str, list[float]],
    units: dict[str, str],
    alternatives: list[str],
    output_unit: str,
    direction: Direction,
    scenarios: Mapping[str, Mapping[str, float]],
    scenario_assumptions: Mapping[str, str],
    two_way_pairs: Sequence[tuple[str, str]],
    feasible_two_way_points: Mapping[str, Sequence[tuple[float, float]]],
    two_way_semantics: Mapping[str, str],
    absolute_tolerance: float,
    relative_tolerance: float,
    analysis_id: str,
) -> DeterministicSensitivityResult:
    baseline_record, raw_baseline_outputs = _safe_evaluate(
        evaluator, "baseline", "baseline", baseline
    )
    if set(raw_baseline_outputs) != set(alternatives):
        raise_input_error("Callback outputs must exactly match alternative_names.")
    baseline_outputs = {
        name: float(raw_baseline_outputs[name]) for name in alternatives
    }
    baseline_point = _make_point(
        baseline_record,
        baseline,
        raw_baseline_outputs,
        alternatives,
        baseline_outputs,
        direction,
        absolute_tolerance,
        relative_tolerance,
    )

    one_way: dict[str, list[DsaPoint]] = {}
    for name, grid in grids.items():
        points: list[DsaPoint] = []
        for index, value in enumerate(grid):
            coordinates = {**baseline, name: value}
            record_id, outputs = _safe_evaluate(evaluator, "one-way", name, coordinates)
            if not record_id:
                record_id = f"{name}-{index}"
            points.append(
                _make_point(
                    record_id,
                    coordinates,
                    outputs,
                    alternatives,
                    baseline_outputs,
                    direction,
                    absolute_tolerance,
                    relative_tolerance,
                )
            )
        one_way[name] = points

    two_way: dict[str, list[DsaPoint]] = {}
    for first, second in two_way_pairs:
        if first == second or first not in baseline or second not in baseline:
            raise_input_error(
                "two_way_pairs must name two distinct baseline parameters."
            )
        surface_id = f"{first}|{second}"
        pairs = feasible_two_way_points.get(surface_id)
        if pairs is None:
            pairs = [(a, b) for a in grids[first] for b in grids[second]]
        points = []
        for index, pair in enumerate(pairs):
            a, b = float(pair[0]), float(pair[1])
            if a not in grids[first] or b not in grids[second]:
                raise_input_error(
                    "Feasible two-way points must belong to declared grids."
                )
            coordinates = {**baseline, first: a, second: b}
            record_id, outputs = _safe_evaluate(
                evaluator, "two-way", surface_id, coordinates
            )
            if not record_id:
                record_id = f"{surface_id}-{index}"
            points.append(
                _make_point(
                    record_id,
                    coordinates,
                    outputs,
                    alternatives,
                    baseline_outputs,
                    direction,
                    absolute_tolerance,
                    relative_tolerance,
                )
            )
        two_way[surface_id] = points

    scenario_points: dict[str, DsaPoint] = {}
    for scenario_id, overrides in scenarios.items():
        unknown = set(overrides) - set(baseline)
        if unknown:
            raise_input_error(f"Unknown scenario coordinates: {sorted(unknown)}.")
        coordinates = {**baseline, **{name: float(v) for name, v in overrides.items()}}
        if not all(np.isfinite(value) for value in coordinates.values()):
            raise_input_error("Scenario coordinates must be finite.")
        record_id, outputs = _safe_evaluate(
            evaluator, "scenario", scenario_id, coordinates
        )
        scenario_points[scenario_id] = _make_point(
            record_id or f"scenario-{scenario_id}",
            coordinates,
            outputs,
            alternatives,
            baseline_outputs,
            direction,
            absolute_tolerance,
            relative_tolerance,
        )

    summaries: list[DsaParameterSummary] = []
    switches: list[DsaSwitchInterval] = []
    for name, points in one_way.items():
        metrics = np.asarray([point.optimal_metric for point in points], dtype=float)
        minimum = float(np.min(metrics))
        maximum = float(np.max(metrics))
        min_coords = [
            grids[name][i] for i, value in enumerate(metrics) if value == minimum
        ]
        max_coords = [
            grids[name][i] for i, value in enumerate(metrics) if value == maximum
        ]
        endpoint_range = abs(float(metrics[-1]) - float(metrics[0]))
        interior = any(
            coordinate not in {grids[name][0], grids[name][-1]}
            for coordinate in [*min_coords, *max_coords]
        )
        summaries.append(
            DsaParameterSummary(
                name,
                units[name],
                "evaluated-grid-range",
                minimum,
                maximum,
                maximum - minimum,
                min_coords,
                max_coords,
                endpoint_range,
                interior,
                0,
            )
        )
        tied_indices = [
            i for i, point in enumerate(points) if len(point.optimal_alternatives) > 1
        ]
        consumed: set[int] = set()
        for index in tied_indices:
            if index in consumed:
                continue
            end = index
            while (
                end + 1 in tied_indices
                and points[end + 1].optimal_alternatives
                == points[index].optimal_alternatives
            ):
                end += 1
                consumed.add(end)
            status: Literal["exact", "plateau", "bracket"] = (
                "plateau" if end > index else "exact"
            )
            switches.append(
                DsaSwitchInterval(
                    name,
                    units[name],
                    status,
                    grids[name][index],
                    grids[name][end],
                    points[index].optimal_alternatives,
                    points[end].optimal_alternatives,
                )
            )
        for index in range(len(points) - 1):
            left, right = points[index], points[index + 1]
            if (
                len(left.optimal_alternatives) > 1
                or len(right.optimal_alternatives) > 1
            ):
                continue
            if left.optimal_alternatives != right.optimal_alternatives:
                switches.append(
                    DsaSwitchInterval(
                        name,
                        units[name],
                        "bracket",
                        grids[name][index],
                        grids[name][index + 1],
                        left.optimal_alternatives,
                        right.optimal_alternatives,
                    )
                )

    summaries.sort(key=lambda item: (-item.evaluated_range, item.parameter_name))
    ranked: list[DsaParameterSummary] = []
    prior_range: float | None = None
    current_rank = 0
    for index, item in enumerate(summaries, 1):
        if prior_range is None or item.evaluated_range != prior_range:
            current_rank = index
        ranked.append(replace(item, rank=current_rank))
        prior_range = item.evaluated_range

    evaluated_count = (
        1
        + sum(map(len, one_way.values()))
        + sum(map(len, two_way.values()))
        + len(scenario_points)
    )
    return DeterministicSensitivityResult(
        baseline_point=baseline_point,
        one_way_points=one_way,
        two_way_points=two_way,
        scenario_points=scenario_points,
        parameter_summaries=ranked,
        switch_intervals=switches,
        parameter_units=units,
        direction=direction,
        output_unit=output_unit.strip(),
        analysis_id=analysis_id,
        two_way_semantics=dict(two_way_semantics),
        scenario_assumptions=dict(scenario_assumptions),
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
        evaluated_record_count=evaluated_count,
        diagnostics={
            "assurance": "exact-enumeration",
            "evaluated_record_count": evaluated_count,
            "complete_coordinate_vectors": True,
            "complete_tie_sets": True,
            "unsupported_operations": [
                "interpolation",
                "extrapolation",
                "probabilistic-sensitivity",
                "evppi",
                "global-sensitivity",
            ],
        },
        reporting={
            "probabilistic_analysis": False,
            "information_value": False,
            "global_sensitivity": False,
            "tornado_semantics": "deterministic-evaluated-grid-range",
            "switch_semantics": "observed-exact-plateau-or-adjacent-bracket",
        },
    )


def deterministic_sensitivity(
    model: Callable[[Mapping[str, float]], Mapping[str, float]],
    *,
    baseline_parameters: Mapping[str, float],
    parameter_grids: Mapping[str, Sequence[float]],
    parameter_units: Mapping[str, str],
    alternative_names: Sequence[str],
    output_unit: str,
    direction: str,
    scenarios: Mapping[str, Mapping[str, float]] | None = None,
    two_way_pairs: Sequence[tuple[str, str]] = (),
    feasible_two_way_points: Mapping[str, Sequence[tuple[float, float]]] | None = None,
    absolute_tolerance: float = 1e-12,
    relative_tolerance: float = 1e-12,
    analysis_id: str = "callback-analysis",
) -> DeterministicSensitivityResult:
    """Evaluate governed one-way, two-way and named deterministic scenarios."""
    baseline, grids, units, alternatives, checked_direction = _validated_inputs(
        baseline_parameters,
        parameter_grids,
        parameter_units,
        alternative_names,
        output_unit,
        direction,
        absolute_tolerance,
        relative_tolerance,
    )

    def evaluator(
        kind: str, reference: str, coordinates: Mapping[str, float]
    ) -> tuple[str, Mapping[str, float]]:
        del kind, reference
        try:
            return "", model(dict(coordinates))
        except Exception as error:
            return raise_input_error(f"DSA callback failed: {error}")

    scenario_map = scenarios or {}
    surface_ids = [f"{first}|{second}" for first, second in two_way_pairs]
    if len(surface_ids) != len(set(surface_ids)):
        raise_input_error("two_way_pairs must not contain duplicate surfaces.")
    feasible_map = feasible_two_way_points or {}
    unknown_surfaces = set(feasible_map) - set(surface_ids)
    if unknown_surfaces:
        raise_input_error(
            f"Unknown feasible_two_way_points surfaces: {sorted(unknown_surfaces)}."
        )
    return _build_result(
        evaluator,
        baseline=baseline,
        grids=grids,
        units=units,
        alternatives=alternatives,
        output_unit=output_unit,
        direction=checked_direction,
        scenarios=scenario_map,
        scenario_assumptions=dict.fromkeys(
            scenario_map, "declared deterministic scenario"
        ),
        two_way_pairs=two_way_pairs,
        feasible_two_way_points=feasible_map,
        two_way_semantics={
            f"{first}|{second}": (
                "explicit-mask"
                if f"{first}|{second}" in feasible_map
                else "full-cartesian-independent"
            )
            for first, second in two_way_pairs
        },
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
        analysis_id=analysis_id,
    )


def deterministic_sensitivity_from_specification(
    specification: Mapping[str, object],
) -> DeterministicSensitivityResult:
    """Evaluate the normalized deterministic-sensitivity-input-v1 contract."""
    validate_deterministic_sensitivity_specification(specification)
    baseline_entries = cast("list[dict[str, Any]]", specification["baseline"])
    grid_entries = cast("list[dict[str, Any]]", specification["parameter_grids"])
    if not isinstance(baseline_entries, list) or not isinstance(grid_entries, list):
        raise_input_error("DSA baseline and parameter_grids must be arrays.")
    baseline = {item["parameter_name"]: item["value"] for item in baseline_entries}
    grids = {item["parameter_name"]: item["values"] for item in grid_entries}
    units = {item["parameter_name"]: item["unit"] for item in grid_entries}
    tolerance = cast("dict[str, Any]", specification["tie_tolerance"])
    records = cast("list[dict[str, Any]]", specification["model_evaluation_records"])
    if not isinstance(tolerance, Mapping) or not isinstance(records, list):
        raise_input_error("DSA tolerance and evaluation records are malformed.")
    lookup: dict[tuple[str, str, tuple[tuple[str, float], ...]], dict[str, Any]] = {}
    consumed: set[tuple[str, str, tuple[tuple[str, float], ...]]] = set()
    for record in records:
        coordinates = tuple(
            (item["parameter_name"], float(item["value"]))
            for item in record["coordinates"]
        )
        key = (record["analysis_kind"], record["analysis_ref"], coordinates)
        if key in lookup:
            raise_input_error("Normalized DSA records must have unique identities.")
        lookup[key] = record

    def evaluator(
        kind: str, reference: str, coordinates: Mapping[str, float]
    ) -> tuple[str, Mapping[str, float]]:
        key = (
            kind,
            reference,
            tuple((name, float(value)) for name, value in coordinates.items()),
        )
        if key not in lookup:
            raise_input_error(
                f"Missing normalized DSA evaluation record for {kind}:{reference}."
            )
        record = lookup[key]
        consumed.add(key)
        outputs = {
            item["alternative_name"]: item["value"]
            for item in record["alternative_outputs"]
        }
        return str(record["record_id"]), outputs

    two_way_designs = cast("list[dict[str, Any]]", specification["two_way_designs"])
    scenario_entries = cast("list[dict[str, Any]]", specification["scenarios"])
    two_way_pairs = [
        (item["first_parameter"], item["second_parameter"]) for item in two_way_designs
    ]
    feasible = {
        item["surface_id"]: [
            (point["first"], point["second"]) for point in item["feasible_points"]
        ]
        for item in two_way_designs
    }
    scenarios = {
        item["scenario_id"]: {
            coordinate["parameter_name"]: coordinate["value"]
            for coordinate in item["coordinates"]
        }
        for item in scenario_entries
    }
    checked = _validated_inputs(
        baseline,
        grids,
        units,
        cast("Sequence[str]", specification["alternative_names"]),
        str(specification["output_unit"]),
        str(specification["direction"]),
        float(tolerance["absolute"]),
        float(tolerance["relative"]),
    )
    result = _build_result(
        evaluator,
        baseline=checked[0],
        grids=checked[1],
        units=checked[2],
        alternatives=checked[3],
        output_unit=str(specification["output_unit"]),
        direction=checked[4],
        scenarios=scenarios,
        scenario_assumptions={
            item["scenario_id"]: item["structural_assumption"]
            for item in scenario_entries
        },
        two_way_pairs=two_way_pairs,
        feasible_two_way_points=feasible,
        two_way_semantics={
            item["surface_id"]: item["feasibility_semantics"]
            for item in two_way_designs
        },
        absolute_tolerance=float(tolerance["absolute"]),
        relative_tolerance=float(tolerance["relative"]),
        analysis_id=str(specification["analysis_id"]),
    )
    if consumed != set(lookup):
        unused = sorted(
            f"{kind}:{reference}" for kind, reference, _ in set(lookup) - consumed
        )
        raise_input_error(f"Unused normalized DSA evaluation records: {unused}.")
    return result
