# Adapted from edithatogo/vop_poc_nz at ad84d309 (Apache-2.0).
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from voiage.methods.perspective_tensor import (
    METHOD_CONTRACT_VERSION,
    NetBenefitTensor,
    TiePolicy,
)

FIXTURE = Path(__file__).parent / "fixtures" / "perspective_conformance_v1.json"


def load_fixture(fixture_id: str) -> dict[str, object]:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert data["method_contract_version"] == METHOD_CONTRACT_VERSION
    return next(item for item in data["fixtures"] if item["id"] == fixture_id)


def tensor(item: dict[str, object]) -> NetBenefitTensor:
    return NetBenefitTensor(
        values=np.asarray(item["values"], dtype=float),
        strategies=tuple(item["strategies"]),
        perspectives=tuple(item["perspectives"]),
    )


def test_directional_regret_fixture() -> None:
    item = load_fixture("directional_regret")
    value = tensor(item)
    expected = item["expected"]
    assert value.evop(
        choose_under="health_system", evaluate_under="societal"
    ).per_person == pytest.approx(expected["health_system_to_societal"])
    assert value.evop(
        choose_under="societal", evaluate_under="health_system"
    ).per_person == pytest.approx(expected["societal_to_health_system"])
    assert value.discordance_probability("health_system", "societal") == pytest.approx(
        expected["discordance_probability"]
    )


def test_tie_split_fixture() -> None:
    item = load_fixture("tie_split_acceptability")
    rows = {
        row.strategy: row
        for row in tensor(item).perspective_acceptability_frontier(
            tie_policy=TiePolicy.SPLIT
        )
    }
    expected = item["expected"]
    for strategy in ("A", "B"):
        assert rows[strategy].probability_optimal == pytest.approx(
            expected["probability_optimal"][strategy]
        )
        assert (
            rows[strategy].expected_value_rank
            == expected["expected_value_rank"][strategy]
        )


def test_exact_switch_fixture() -> None:
    item = load_fixture("exact_mixture_switch")
    value = tensor(item)
    segments = value.exact_perspective_frontier(
        left_perspective="left", right_perspective="right"
    )
    expected = item["expected"]
    assert len(segments) == 2
    for actual, wanted in zip(segments, expected["segments"], strict=True):
        assert actual.lower_right_weight == pytest.approx(wanted["lower"])
        assert actual.upper_right_weight == pytest.approx(wanted["upper"])
        assert list(actual.optimal_strategies) == wanted["strategies"]
    switch = value.exact_switch_points(
        left_perspective="left", right_perspective="right"
    )[0]
    assert switch.right_weight == pytest.approx(expected["switch_weight"])


def test_additional_reference_outputs() -> None:
    from voiage.methods.perspective_tensor import expected_value_switch_points

    value = tensor(load_fixture("exact_mixture_switch"))
    options = {"left_perspective": "left", "right_perspective": "right"}
    grid = value.perspective_mixture_frontier(**options, grid_size=3)
    actual = {
        "grid": [r.as_dict() for r in grid],
        "exact": [r.as_dict() for r in expected_value_switch_points(value, **options)],
        "sampled": [r.as_dict() for r in expected_value_switch_points(grid)],
        "mcda": value.mcda_feature_records(reference_perspective="left", population=37),
        "scaled": value.evop(
            choose_under="left", evaluate_under="right", population=37
        ).as_dict(),
    }
    assert actual == json.loads(
        FIXTURE.with_name("perspective_tensor_reference.json").read_text()
    )
