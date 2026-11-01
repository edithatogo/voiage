# Adapted from edithatogo/vop_poc_nz at ad84d309 (Apache-2.0).
from __future__ import annotations

import numpy as np
import pytest

from voiage.methods.perspective_tensor import (
    DecisionRule,
    NetBenefitTensor,
    PerspectiveError,
    TiePolicy,
    tensor_from_records,
)


def test_expected_value_evop_is_directional_strategy_regret() -> None:
    tensor = NetBenefitTensor(
        values=np.array(
            [
                [[10.0, 0.0], [0.0, 100.0]],
                [[10.0, 0.0], [0.0, 100.0]],
                [[10.0, 0.0], [0.0, 100.0]],
            ]
        ),
        strategies=("A", "B"),
        perspectives=("health_system", "societal"),
    )

    hs_to_soc = tensor.evop(choose_under="health_system", evaluate_under="societal")
    soc_to_hs = tensor.evop(choose_under="societal", evaluate_under="health_system")

    assert hs_to_soc.chosen_strategy == "A"
    assert hs_to_soc.target_strategy == "B"
    assert hs_to_soc.per_person == pytest.approx(100.0)
    assert soc_to_hs.per_person == pytest.approx(10.0)
    assert hs_to_soc.per_person != soc_to_hs.per_person


def test_regret_matrix_has_zero_diagonal() -> None:
    tensor = NetBenefitTensor(
        values=np.array(
            [
                [[5.0, 5.0], [2.0, 2.0]],
                [[4.0, 4.0], [3.0, 3.0]],
            ]
        ),
        strategies=("A", "B"),
        perspectives=("p1", "p2"),
    )

    matrix = tensor.regret_matrix()
    diagonal = [row for row in matrix if row["choose_under"] == row["evaluate_under"]]
    assert diagonal
    assert all(row["per_person"] == pytest.approx(0.0) for row in diagonal)


def test_expected_value_and_per_draw_rules_are_distinct() -> None:
    # Mean values align under p1 and p2, but draw-level optima frequently differ.
    tensor = NetBenefitTensor(
        values=np.array(
            [
                [[100.0, 0.0], [0.0, 90.0]],
                [[0.0, 90.0], [100.0, 0.0]],
                [[100.0, 0.0], [0.0, 90.0]],
                [[0.0, 90.0], [100.0, 0.0]],
            ]
        ),
        strategies=("A", "B"),
        perspectives=("p1", "p2"),
    )

    expected = tensor.evop(choose_under="p1", evaluate_under="p2")
    per_draw = tensor.evop(
        choose_under="p1",
        evaluate_under="p2",
        decision_rule=DecisionRule.PER_DRAW,
    )

    assert expected.decision_rule is DecisionRule.EXPECTED_VALUE
    assert per_draw.decision_rule is DecisionRule.PER_DRAW
    assert expected.per_person == pytest.approx(0.0)
    assert per_draw.per_person == pytest.approx(90.0)
    assert expected.discordance_probability == pytest.approx(1.0)


def test_perspective_acceptability_frontier_reports_probability_optimal() -> None:
    tensor = NetBenefitTensor(
        values=np.array(
            [
                [[5.0], [0.0]],
                [[0.0], [5.0]],
                [[5.0], [0.0]],
                [[5.0], [0.0]],
            ]
        ),
        strategies=("A", "B"),
        perspectives=("societal",),
    )

    rows = {row.strategy: row for row in tensor.perspective_acceptability_frontier()}
    assert rows["A"].probability_optimal == pytest.approx(0.75)
    assert rows["B"].probability_optimal == pytest.approx(0.25)
    assert rows["A"].expected_value_rank == 1


def test_weighted_perspective_can_be_used_for_stakeholder_mixture() -> None:
    tensor = NetBenefitTensor(
        values=np.array([[[10.0, 0.0]], [[20.0, 10.0]]]),
        strategies=("A",),
        perspectives=("health_system", "societal"),
    )

    mixed = tensor.with_weighted_perspective(
        "deliberative", {"health_system": 1.0, "societal": 3.0}
    )

    assert mixed.perspectives == ("health_system", "societal", "deliberative")
    assert mixed.values[:, 0, 2].tolist() == pytest.approx([2.5, 12.5])


def test_tensor_from_records_requires_dense_values() -> None:
    records = [
        {"draw": 0, "strategy": "A", "perspective": "p", "net_benefit": 1.0},
        {"draw": 0, "strategy": "B", "perspective": "p", "net_benefit": 2.0},
        {"draw": 1, "strategy": "A", "perspective": "p", "net_benefit": 3.0},
    ]

    with pytest.raises(PerspectiveError, match="not dense"):
        tensor_from_records(records)


def test_tensor_from_records_round_trips_dense_values() -> None:
    records = [
        {"draw": 0, "strategy": "A", "perspective": "p", "net_benefit": 1.0},
        {"draw": 0, "strategy": "B", "perspective": "p", "net_benefit": 2.0},
        {"draw": 1, "strategy": "A", "perspective": "p", "net_benefit": 3.0},
        {"draw": 1, "strategy": "B", "perspective": "p", "net_benefit": 4.0},
    ]

    tensor = tensor_from_records(records, case_id="fixture")
    assert tensor.case_id == "fixture"
    assert tensor.values.shape == (2, 2, 1)
    assert tensor.strategies == ("A", "B")


def test_weight_validation_and_population_validation() -> None:
    tensor = NetBenefitTensor(
        values=np.array([[[1.0, 2.0]]]),
        strategies=("A",),
        perspectives=("p1", "p2"),
    )
    with pytest.raises(PerspectiveError, match="non-negative"):
        tensor.with_weighted_perspective("bad", {"p1": -1.0, "p2": 2.0})
    with pytest.raises(PerspectiveError, match="population"):
        tensor.evop(choose_under="p1", evaluate_under="p2", population=-1)


def test_paf_splits_exact_ties() -> None:
    tensor = NetBenefitTensor(
        values=np.array([[[5.0], [5.0]], [[5.0], [0.0]]]),
        strategies=("A", "B"),
        perspectives=("p",),
    )
    rows = {row.strategy: row for row in tensor.perspective_acceptability_frontier()}
    assert rows["A"].probability_optimal == pytest.approx(0.75)
    assert rows["B"].probability_optimal == pytest.approx(0.25)


def test_evop_split_ties_averages_source_tied_strategies() -> None:
    tensor = NetBenefitTensor(
        values=np.array(
            [
                [[1.0, 0.0], [1.0, 10.0]],
                [[1.0, 0.0], [1.0, 10.0]],
            ]
        ),
        strategies=("a", "b"),
        perspectives=("source", "target"),
    )
    split = tensor.evop(
        choose_under="source",
        evaluate_under="target",
        selection_tie_policy=TiePolicy.SPLIT,
    )
    first = tensor.evop(
        choose_under="source",
        evaluate_under="target",
        selection_tie_policy=TiePolicy.FIRST,
    )
    assert split.per_person == pytest.approx(5.0)
    assert first.per_person == pytest.approx(10.0)
    assert split.tie_detected is True


def test_error_policy_rejects_tied_draws() -> None:
    tensor = NetBenefitTensor(
        values=np.ones((2, 2, 2)), strategies=("A", "B"), perspectives=("left", "right")
    )
    with pytest.raises(PerspectiveError, match="Tied"):
        tensor.discordance_probability("left", "right", tie_policy=TiePolicy.ERROR)
    with pytest.raises(PerspectiveError, match="[Tt]ie"):
        tensor.perspective_acceptability_frontier(tie_policy=TiePolicy.ERROR)
