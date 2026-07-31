"""Analytical and pathology contracts for governed DSA issue #556."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from voiage.methods.deterministic_sensitivity import deterministic_sensitivity

if TYPE_CHECKING:
    from collections.abc import Mapping


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
        ({"parameter_grids": {"x": [], "y": [0.0]}}, "non-empty grid"),
        ({"parameter_grids": {"x": [0.0, 0.0], "y": [0.0]}}, "duplicate"),
        ({"parameter_units": {"x": "point"}}, "exactly match"),
        ({"alternative_names": ["a", "a"]}, "unique"),
        ({"output_unit": ""}, "output_unit"),
        ({"direction": "sideways"}, "direction"),
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
    with pytest.raises(ValueError, match="callback failed"):
        deterministic_sensitivity(
            lambda _p: (_ for _ in ()).throw(RuntimeError("boom")), **kwargs
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


def test_dsa_experimental_exports_are_discoverable() -> None:
    import voiage
    from voiage import methods

    assert voiage.deterministic_sensitivity is deterministic_sensitivity
    assert methods.deterministic_sensitivity is deterministic_sensitivity
