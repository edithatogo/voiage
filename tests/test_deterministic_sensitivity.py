"""Analytical and pathology contracts for governed DSA issue #556."""

from __future__ import annotations

from copy import deepcopy
import importlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

from hypothesis import given
from hypothesis import strategies as st
import numpy as np
import pytest

from voiage.methods.deterministic_sensitivity import (
    deterministic_sensitivity,
    deterministic_sensitivity_from_specification,
)

if TYPE_CHECKING:
    from collections.abc import Mapping


dsa_module = importlib.import_module("voiage.methods.deterministic_sensitivity")


def _linear_model(parameters: Mapping[str, float]) -> Mapping[str, float]:
    x = parameters["x"]
    y = parameters["y"]
    return {"a": 10.0 + x - y, "b": 8.0 + 2.0 * x + y}


def _base_kwargs() -> dict[str, object]:
    return {
        "baseline_parameters": {"x": 0.0, "y": 0.0},
        "parameter_grids": {"x": [-2.0, 0.0, 2.0], "y": [-1.0, 0.0, 1.0]},
        "parameter_units": {"x": "point", "y": "point"},
        "alternative_names": ["a", "b"],
        "output_unit": "net-benefit-point",
        "direction": "maximize",
    }


def test_one_way_dsa_freezes_other_coordinates_and_matches_linear_reference() -> None:
    calls: list[dict[str, float]] = []

    def recording_model(parameters: Mapping[str, float]) -> Mapping[str, float]:
        calls.append(dict(parameters))
        return _linear_model(parameters)

    result = deterministic_sensitivity(recording_model, **_base_kwargs())

    assert result.baseline_point.alternative_outputs == {"a": 10.0, "b": 8.0}
    assert result.baseline_point.optimal_alternatives == ["a"]
    assert [point.coordinates["x"] for point in result.one_way_points["x"]] == [
        -2.0,
        0.0,
        2.0,
    ]
    assert all(point.coordinates["y"] == 0.0 for point in result.one_way_points["x"])
    assert all(point.coordinates["x"] == 0.0 for point in result.one_way_points["y"])
    assert all(set(call) == {"x", "y"} for call in calls)


def test_tornado_range_uses_grid_extrema_not_only_endpoints() -> None:
    def u_shape(parameters: Mapping[str, float]) -> Mapping[str, float]:
        x = parameters["x"]
        return {"only": (x - 1.0) ** 2}

    result = deterministic_sensitivity(
        u_shape,
        baseline_parameters={"x": 0.0},
        parameter_grids={"x": [0.0, 1.0, 2.0, 3.0]},
        parameter_units={"x": "point"},
        alternative_names=["only"],
        output_unit="point",
        direction="maximize",
    )
    summary = result.parameter_summaries[0]
    assert summary.parameter_name == "x"
    assert summary.evaluated_range == pytest.approx(4.0)
    assert summary.minimum_metric == pytest.approx(0.0)
    assert summary.maximum_metric == pytest.approx(4.0)
    assert summary.ranking_metric == "evaluated-grid-range"


@given(st.lists(st.integers(-20, 20), min_size=2, max_size=8, unique=True).map(sorted))
def test_tornado_range_matches_direct_brute_force(grid: list[int]) -> None:
    result = deterministic_sensitivity(
        lambda parameters: {"only": float(parameters["x"] ** 2 - parameters["x"])},
        baseline_parameters={"x": 0.0},
        parameter_grids={"x": grid},
        parameter_units={"x": "point"},
        alternative_names=["only"],
        output_unit="point",
        direction="maximize",
    )
    evaluated = [float(value**2 - value) for value in grid]

    assert result.parameter_summaries[0].minimum_metric == min(evaluated)
    assert result.parameter_summaries[0].maximum_metric == max(evaluated)
    assert result.parameter_summaries[0].evaluated_range == max(evaluated) - min(
        evaluated
    )


def test_switch_contract_returns_exact_ties_and_brackets_without_root_invention() -> (
    None
):
    def crossing(parameters: Mapping[str, float]) -> Mapping[str, float]:
        x = parameters["x"]
        return {"left": -x, "right": x}

    exact = deterministic_sensitivity(
        crossing,
        baseline_parameters={"x": -1.0},
        parameter_grids={"x": [-1.0, 0.0, 1.0]},
        parameter_units={"x": "point"},
        alternative_names=["right", "left"],
        output_unit="point",
        direction="maximize",
    )
    assert exact.one_way_points["x"][1].optimal_alternatives == ["left", "right"]
    assert exact.one_way_points["x"][1].selected_alternative == "left"
    assert [
        (item.status, item.lower_coordinate, item.upper_coordinate)
        for item in exact.switch_intervals
    ] == [("exact", 0.0, 0.0)]

    bracketed = deterministic_sensitivity(
        crossing,
        baseline_parameters={"x": -1.0},
        parameter_grids={"x": [-1.0, 1.0]},
        parameter_units={"x": "point"},
        alternative_names=["right", "left"],
        output_unit="point",
        direction="maximize",
    )
    assert [
        (item.status, item.lower_coordinate, item.upper_coordinate)
        for item in bracketed.switch_intervals
    ] == [("bracket", -1.0, 1.0)]


def test_switch_pathologies_cover_multiple_none_discontinuity_and_plateau() -> None:
    multiple = deterministic_sensitivity(
        lambda p: {"outer": p["x"] ** 2, "middle": 1.0},
        baseline_parameters={"x": 0.0},
        parameter_grids={"x": [-2.0, 0.0, 2.0]},
        parameter_units={"x": "point"},
        alternative_names=["outer", "middle"],
        output_unit="point",
        direction="maximize",
    )
    assert [item.status for item in multiple.switch_intervals] == [
        "bracket",
        "bracket",
    ]

    no_switch = deterministic_sensitivity(
        lambda p: {"always": p["x"], "never": p["x"] - 1.0},
        baseline_parameters={"x": 0.0},
        parameter_grids={"x": [-1.0, 0.0, 1.0]},
        parameter_units={"x": "point"},
        alternative_names=["always", "never"],
        output_unit="point",
        direction="maximize",
    )
    assert no_switch.switch_intervals == []

    plateau = deterministic_sensitivity(
        lambda p: {"a": p["x"], "b": p["x"]},
        baseline_parameters={"x": 0.0},
        parameter_grids={"x": [-1.0, 0.0, 1.0]},
        parameter_units={"x": "point"},
        alternative_names=["a", "b"],
        output_unit="point",
        direction="maximize",
    )
    assert [
        (item.status, item.lower_coordinate, item.upper_coordinate)
        for item in plateau.switch_intervals
    ] == [("plateau", -1.0, 1.0)]

    discontinuous = deterministic_sensitivity(
        lambda p: {"left": float(p["x"] < 0), "right": float(p["x"] >= 0)},
        baseline_parameters={"x": -1.0},
        parameter_grids={"x": [-1.0, 1.0]},
        parameter_units={"x": "point"},
        alternative_names=["left", "right"],
        output_unit="point",
        direction="maximize",
    )
    assert discontinuous.switch_intervals[0].status == "bracket"
    assert discontinuous.switch_intervals[0].interpolation == "not-performed"


def test_tolerance_boundary_repeatability_and_input_immutability() -> None:
    baseline = {"x": 0.0}
    grids = {"x": [0.0, 1.0]}
    kwargs = {
        "baseline_parameters": baseline,
        "parameter_grids": grids,
        "parameter_units": {"x": "point"},
        "alternative_names": ["a", "b"],
        "output_unit": "point",
        "direction": "maximize",
        "absolute_tolerance": 0.1,
        "relative_tolerance": 0.0,
    }
    first = deterministic_sensitivity(lambda _p: {"a": 1.0, "b": 0.9}, **kwargs)
    second = deterministic_sensitivity(lambda _p: {"a": 1.0, "b": 0.9}, **kwargs)
    outside = deterministic_sensitivity(lambda _p: {"a": 1.0, "b": 0.899}, **kwargs)

    assert first.to_contract_dict() == second.to_contract_dict()
    assert first.baseline_point.optimal_alternatives == ["a", "b"]
    assert outside.baseline_point.optimal_alternatives == ["a"]
    assert baseline == {"x": 0.0}
    assert grids == {"x": [0.0, 1.0]}


def test_direction_reversal_and_permutation_preserve_complete_tie_semantics() -> None:
    kwargs = _base_kwargs()
    maximized = deterministic_sensitivity(_linear_model, **kwargs)
    permuted = deterministic_sensitivity(
        _linear_model,
        **{**kwargs, "alternative_names": ["b", "a"]},
    )
    minimized = deterministic_sensitivity(
        lambda p: {name: -value for name, value in _linear_model(p).items()},
        **{**kwargs, "direction": "minimize"},
    )
    assert maximized.baseline_point.optimal_alternatives == ["a"]
    assert permuted.baseline_point.optimal_alternatives == ["a"]
    assert minimized.baseline_point.optimal_alternatives == ["a"]


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"baseline_parameters": {"x": np.nan, "y": 0.0}}, "finite baseline"),
        ({"parameter_grids": {"x": [0.0]}}, "exactly match the baseline"),
        ({"parameter_grids": {"x": [], "y": [0.0]}}, "non-empty grid"),
        ({"parameter_grids": {"x": [np.inf], "y": [0.0]}}, "finite values"),
        ({"parameter_grids": {"x": [0.0, 0.0], "y": [0.0]}}, "duplicate"),
        ({"parameter_grids": {"x": [1.0, 0.0], "y": [0.0]}}, "increasing"),
        ({"parameter_units": {"x": "point"}}, "exactly match"),
        ({"parameter_units": {"x": "", "y": "point"}}, "non-empty unit"),
        ({"alternative_names": ["a", "a"]}, "unique"),
        ({"output_unit": ""}, "output_unit"),
        ({"direction": "sideways"}, "direction"),
        ({"absolute_tolerance": -1.0}, "finite and non-negative"),
    ],
)
def test_dsa_rejects_malformed_contracts(
    override: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        deterministic_sensitivity(_linear_model, **{**_base_kwargs(), **override})


def test_dsa_rejects_nonfinite_unknown_or_malformed_callback_outputs() -> None:
    kwargs = _base_kwargs()
    with pytest.raises(ValueError, match="finite outputs"):
        deterministic_sensitivity(lambda _p: {"a": np.inf, "b": 1.0}, **kwargs)
    with pytest.raises(ValueError, match="exactly match alternative_names"):
        deterministic_sensitivity(lambda _p: {"a": 1.0}, **kwargs)
    with pytest.raises(ValueError, match="exactly match alternative_names"):
        deterministic_sensitivity(
            lambda p: _linear_model(p) if p == {"x": 0.0, "y": 0.0} else {"a": 1.0},
            **kwargs,
        )
    with pytest.raises(ValueError, match="callback failed"):
        deterministic_sensitivity(
            lambda _p: (_ for _ in ()).throw(RuntimeError("boom")), **kwargs
        )

    with pytest.raises(ValueError, match="callback failed"):
        dsa_module._safe_evaluate(
            lambda *_args: (_ for _ in ()).throw(RuntimeError("boom")),
            "baseline",
            "baseline",
            {"x": 0.0},
        )


def test_scenarios_and_two_way_cells_are_explicit_and_reproducible() -> None:
    result = deterministic_sensitivity(
        _linear_model,
        **_base_kwargs(),
        scenarios={"adverse": {"x": -2.0, "y": 1.0}},
        two_way_pairs=[("x", "y")],
        feasible_two_way_points={"x|y": [(-2.0, -1.0), (0.0, 0.0)]},
    )
    assert result.scenario_points["adverse"].coordinates == {"x": -2.0, "y": 1.0}
    assert [point.coordinates for point in result.two_way_points["x|y"]] == [
        {"x": -2.0, "y": -1.0},
        {"x": 0.0, "y": 0.0},
    ]
    assert result.reporting["probabilistic_analysis"] is False
    assert result.reporting["information_value"] is False


def test_callback_two_way_surface_declarations_fail_closed() -> None:
    kwargs = _base_kwargs()
    with pytest.raises(ValueError, match="duplicate surfaces"):
        deterministic_sensitivity(
            _linear_model,
            **kwargs,
            two_way_pairs=[("x", "y"), ("x", "y")],
        )
    with pytest.raises(ValueError, match="Unknown feasible"):
        deterministic_sensitivity(
            _linear_model,
            **kwargs,
            two_way_pairs=[("x", "y")],
            feasible_two_way_points={"y|x": [(0.0, 0.0)]},
        )
    with pytest.raises(ValueError, match="distinct baseline parameters"):
        deterministic_sensitivity(
            _linear_model,
            **kwargs,
            two_way_pairs=[("x", "x")],
        )
    with pytest.raises(ValueError, match="belong to declared grids"):
        deterministic_sensitivity(
            _linear_model,
            **kwargs,
            two_way_pairs=[("x", "y")],
            feasible_two_way_points={"x|y": [(999.0, 0.0)]},
        )


def test_callback_default_cartesian_scenarios_and_tied_ranks_cover_both_paths() -> None:
    cartesian = deterministic_sensitivity(
        _linear_model,
        **_base_kwargs(),
        two_way_pairs=[("x", "y")],
    )
    assert len(cartesian.two_way_points["x|y"]) == 9

    tied = deterministic_sensitivity(
        lambda p: {"only": p["x"] + p["y"]},
        baseline_parameters={"x": 0.0, "y": 0.0},
        parameter_grids={"x": [0.0, 1.0], "y": [0.0, 1.0]},
        parameter_units={"x": "point", "y": "point"},
        alternative_names=["only"],
        output_unit="point",
        direction="maximize",
    )
    assert [summary.rank for summary in tied.parameter_summaries] == [1, 1]

    for scenarios, message in [
        ({"unknown": {"z": 1.0}}, "Unknown scenario coordinates"),
        ({"nonfinite": {"x": np.nan}}, "Scenario coordinates must be finite"),
    ]:
        with pytest.raises(ValueError, match=message):
            deterministic_sensitivity(
                _linear_model,
                **_base_kwargs(),
                scenarios=scenarios,
            )


def test_normalized_runtime_rejects_post_schema_malformed_and_record_gaps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = Path(
        "specs/frontier/deterministic-sensitivity-analysis/v1/fixtures/normative/input.json"
    )
    specification = json.loads(fixture.read_text(encoding="utf-8"))
    monkeypatch.setattr(
        dsa_module,
        "validate_deterministic_sensitivity_specification",
        lambda _value: None,
    )

    malformed_arrays = deepcopy(specification)
    malformed_arrays["baseline"] = {}
    with pytest.raises(ValueError, match="must be arrays"):
        deterministic_sensitivity_from_specification(malformed_arrays)

    malformed_records = deepcopy(specification)
    malformed_records["model_evaluation_records"] = {}
    with pytest.raises(ValueError, match="records are malformed"):
        deterministic_sensitivity_from_specification(malformed_records)

    duplicate = deepcopy(specification)
    duplicate["model_evaluation_records"].append(
        deepcopy(duplicate["model_evaluation_records"][0])
    )
    with pytest.raises(ValueError, match="unique identities"):
        deterministic_sensitivity_from_specification(duplicate)

    missing = deepcopy(specification)
    missing["model_evaluation_records"] = missing["model_evaluation_records"][:-1]
    with pytest.raises(ValueError, match="Missing normalized DSA evaluation record"):
        deterministic_sensitivity_from_specification(missing)

    unused = deepcopy(specification)
    extra = deepcopy(unused["model_evaluation_records"][0])
    extra["record_id"] = "unused"
    extra["analysis_ref"] = "unused"
    unused["model_evaluation_records"].append(extra)
    with pytest.raises(ValueError, match="Unused normalized DSA evaluation records"):
        deterministic_sensitivity_from_specification(unused)


def test_dsa_experimental_exports_are_discoverable() -> None:
    import voiage
    from voiage import methods

    assert voiage.deterministic_sensitivity is deterministic_sensitivity
    assert methods.deterministic_sensitivity is deterministic_sensitivity
