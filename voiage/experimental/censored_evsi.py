"""Experimental censored sequential EVSI over scoped paired draws."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import json
from typing import TYPE_CHECKING, Literal, TypedDict, cast

import polars as pl

from voiage._runtime import compute_censored_sequential_evsi
from voiage.contracts.preparation import _long_net_benefit_values, _resolve_binding

from .paired_draws import _COLUMNS, _validate_draws

if TYPE_CHECKING:
    from collections.abc import Sequence

    from voiage.contracts.normalized_input import NormalizedInputBundle


class TobitDraw(TypedDict):
    """Measurement parameters conditional on one draw."""

    draw_id: int | str
    mean: float
    standard_deviation: float
    detection_probability: float


class TobitAcquisition(TypedDict):
    """A Gaussian acquisition action and its declared finite partition."""

    action_id: str
    limit: float
    cutpoints: Sequence[float]
    outcome_ids: Sequence[str]
    states: Sequence[TobitDraw]


@dataclass(frozen=True)
class CensoredSequentialEvsiResult:
    """Gross information value and the draw scope used to evaluate it."""

    status: Literal["complete", "indeterminate"]
    perspective: str
    subgroup_id: str
    draw_ids: tuple[int | str, ...]
    strategy_ids: tuple[str, ...]
    current_value: Fraction
    expected_value: Fraction | None
    evsi: Fraction | None
    branches_examined: int


def censored_sequential_evsi(
    draws: NormalizedInputBundle,
    *,
    stages: Sequence[Sequence[TobitAcquisition]],
    perspective: str,
    subgroup_id: str,
    max_branches: int,
) -> CensoredSequentialEvsiResult:
    """Evaluate a finite acquisition schedule for one paired-draw scope.

    Use a long net-benefit binding and an explicit relative prior weight for
    each draw. Acquisition state rows may arrive in any order. They use the
    original draw IDs. Each acquisition lists outcome IDs for unavailability,
    censoring, and the successive bins beginning at its quantification limit.

    The Rust solver calculates the gross value of the declared partition and
    returns an indeterminate result if it exhausts the outcome branch budget.
    """
    if type(max_branches) is not int or max_branches <= 0:
        raise ValueError("A positive integer outcome branch budget is required")
    method = "censored_sequential_evsi"
    payoff = _resolve_binding(
        draws.manifest.bindings, "net_benefit", method_family=method
    )
    weight = _resolve_binding(draws.manifest.bindings, "weight", method_family=method)
    if (
        payoff.layout != "long"
        or payoff.sample_id_field_id != "draw_id"
        or payoff.strategy_field_id != "strategy_id"
        or payoff.value_field_id != "net_benefit"
        or payoff.table_id != weight.table_id
        or len(weight.field_ids) != 1
        or weight.field_ids[0] in _COLUMNS
    ):
        raise ValueError(
            "Censored EVSI requires aligned paired-draw payoff and weight bindings"
        )
    weight_field = weight.field_ids[0]
    table = draws.table(payoff.table_id).select((*_COLUMNS, weight_field))
    frame = cast("pl.DataFrame", pl.from_arrow(table, rechunk=False))
    selected = frame.filter(
        (pl.col("perspective") == perspective) & (pl.col("subgroup_id") == subgroup_id)
    ).to_arrow(compat_level=pl.CompatLevel.newest())
    if not selected.num_rows:
        raise ValueError("No paired draws match the requested perspective and subgroup")
    _validate_draws(selected, set(payoff.strategy_names), weight_field)
    for name in ("net_benefit", weight_field):
        if any(float(value) != value for value in selected[name].to_pylist()):
            raise ValueError(
                f"Values in {name!r} must be exactly representable as binary64"
            )

    weights: dict[int | str, float] = dict(
        zip(
            selected["draw_id"].to_pylist(),
            selected[weight_field].to_pylist(),
            strict=True,
        )
    )
    values, strategy_ids = _long_net_benefit_values(selected, payoff)
    state_ids = [_state_id(draw_id) for draw_id in weights]
    states = [
        {"state_id": identifier, "prior_weight": prior, "net_benefits": row}
        for identifier, prior, row in zip(
            state_ids, weights.values(), values.tolist(), strict=True
        )
    ]
    experiment = {
        "strategy_ids": strategy_ids,
        "states": states,
        "stages": [
            [_encode_acquisition(action) for action in stage] for stage in stages
        ],
    }
    payload = compute_censored_sequential_evsi(
        json.dumps(experiment, allow_nan=False), max_branches=max_branches
    )
    status = payload["status"]
    if (
        payload["contract_version"] != 1
        or payload["state_ids"] != state_ids
        or payload["strategy_ids"] != strategy_ids
        or status not in ("complete", "indeterminate")
    ):
        raise RuntimeError(
            "The native censored EVSI result uses an incompatible contract"
        )
    return CensoredSequentialEvsiResult(
        status=cast('Literal["complete", "indeterminate"]', status),
        perspective=perspective,
        subgroup_id=subgroup_id,
        draw_ids=tuple(weights),
        strategy_ids=tuple(strategy_ids),
        current_value=_fraction(payload["current_value"]),
        expected_value=_fraction(payload["expected_value"])
        if status == "complete"
        else None,
        evsi=_fraction(payload["evsi"]) if status == "complete" else None,
        branches_examined=cast("int", payload["branches_examined"]),
    )


def _state_id(draw_id: int | str) -> str:
    """Keep integer and text identifiers distinct at the native boundary."""
    if type(draw_id) is not int and not isinstance(draw_id, str):
        raise ValueError("Observation state IDs require an integer or text draw ID")
    return json.dumps(draw_id, ensure_ascii=False)


def _encode_acquisition(action: TobitAcquisition) -> dict[str, object]:
    states = [
        {
            "state_id": _state_id(state["draw_id"]),
            "mean": state["mean"],
            "standard_deviation": state["standard_deviation"],
            "detection_probability": state["detection_probability"],
        }
        for state in action["states"]
    ]
    return {**action, "states": states}


def _fraction(value: object) -> Fraction:
    numerator, denominator = cast("list[str]", value)
    return Fraction(int(numerator), int(denominator))
