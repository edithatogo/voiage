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


@pytest.mark.parametrize(
    ("values", "labels", "lenses", "draws", "message"),
    [
        (np.ones((2, 2)), ("a", "b"), ("p",), None, "shape"),
        (np.ones((0, 2, 1)), ("a", "b"), ("p",), None, "at least one"),
        (np.ones((2, 2, 1)), ("a",), ("p",), None, "strategy-name count"),
        (np.ones((2, 2, 1)), ("a", "b"), ("p", "q"), None, "perspective-name count"),
        (np.ones((2, 2, 1)), ("a", "a"), ("p",), None, "strategy names must be unique"),
        (
            np.ones((2, 1, 2)),
            ("a",),
            ("p", "p"),
            None,
            "perspective names must be unique",
        ),
        (np.ones((2, 1, 1)), ("a",), ("p",), ("one",), "draw ID count"),
        (np.full((2, 1, 1), np.inf), ("a",), ("p",), None, "finite"),
    ],
)
def test_tensor_rejects_ambiguous_or_nonfinite_inputs(
    values, labels, lenses, draws, message
):
    with pytest.raises(PerspectiveError, match=message):
        NetBenefitTensor(values, labels, lenses, draw_ids=draws)


@pytest.mark.parametrize(
    ("name", "weights", "normalize", "message"),
    [
        ("p", {"p": 1}, True, "already exists"),
        ("mix", {}, True, "empty"),
        ("mix", {"p": np.inf}, True, "finite"),
        ("mix", {"p": 0}, True, "positive total"),
        ("mix", {"p": 0}, False, "positive total"),
        ("mix", {"absent": 1}, True, "Unknown perspective"),
    ],
)
def test_weighted_tensor_rejects_invalid_mixtures(name, weights, normalize, message):
    tensor = NetBenefitTensor(np.ones((2, 2, 1)), ("a", "b"), ("p",))
    with pytest.raises(PerspectiveError, match=message):
        tensor.with_weighted_perspective(name, weights, normalize=normalize)


def test_expected_and_draw_specific_optima_apply_the_selected_tie_policy():
    tensor = NetBenefitTensor(np.array([[[4], [0]], [[1], [1]]]), ("a", "b"), ("p",))
    assert tensor.optimal_strategy_index("p", tie_policy="error") == 0
    np.testing.assert_array_equal(
        tensor.optimal_strategy_index("p", decision_rule="per_draw"), [0, 0]
    )
    with pytest.raises(PerspectiveError, match="tied"):
        tensor.optimal_strategy_index("p", decision_rule="per_draw", tie_policy="error")
    tied = NetBenefitTensor(np.ones((2, 2, 1)), ("a", "b"), ("p",))
    with pytest.raises(PerspectiveError, match="tied"):
        tied.optimal_strategy_index("p", tie_policy="error")
    assert tensor.strategy_index("b") == 1
    with pytest.raises(PerspectiveError, match="Unknown strategy"):
        tensor.strategy_index("absent")


def test_tie_rules_and_draw_level_loss_distribution():
    from voiage.methods import perspective_tensor as api

    t = NetBenefitTensor(np.array([[[1, 0], [1, 4]]]), ("a", "b"), ("p", "q"))
    options = {"choose_under": "p", "evaluate_under": "q"}
    for rule in DecisionRule:
        with pytest.raises(PerspectiveError, match="Tied"):
            api.evop(t, **options, decision_rule=rule, selection_tie_policy="error")
    split = api.evop(
        t,
        **options,
        decision_rule="per_draw",
        selection_tie_policy="split",
        keep_distribution=True,
    )
    assert split.as_dict(include_distribution=True)["per_draw_loss"] == [2]
    assert t.discordance_probability("p", "q", tie_policy="first") == 1
    rows = api.perspective_acceptability_frontier(t, tie_policy="first")
    assert rows[0].as_dict()["probability_optimal"] == 1
    np.testing.assert_array_equal(api.expected_net_benefit(t, "q"), [0, 4])
    weighted = api.with_weighted_perspective(t, "mix", {"p": 1, "q": 1})
    np.testing.assert_allclose(weighted.nmb("mix"), [[0.5, 2.5]])
    assert api.regret_matrix(t) == t.regret_matrix()
    assert api.mcda_feature_records(
        t, reference_perspective="p"
    ) == t.mcda_feature_records(reference_perspective="p")


def test_dominated_crossings_preserve_a_single_frontier_segment():
    from voiage.methods import perspective_tensor as api

    t = NetBenefitTensor(
        np.array([[[10, 10], [0, 2], [2, 0], [0, 0]]]),
        ("best", "up", "down", "flat"),
        ("p", "q"),
    )
    options = {"left_perspective": "p", "right_perspective": "q"}
    (segment,) = api.exact_perspective_frontier(t, **options)
    assert segment.as_dict()["optimal_strategies"] == ["best"]
    assert (segment.lower_right_weight, segment.upper_right_weight) == (0, 1)
    (collapsed,) = api.exact_perspective_frontier(t, **options, atol=1)
    assert collapsed == segment
    with pytest.raises(PerspectiveError, match="grid_size"):
        api.perspective_mixture_frontier(t, **options, grid_size=1)
    # Parallel lines and an intersection outside the mixture interval.
    no_switch = NetBenefitTensor(
        np.array([[[10, 11], [0, 1], [1, 1]]]), t.strategies[:3], t.perspectives
    )
    assert not no_switch.exact_switch_points(**options)


def test_records_require_unique_keys_and_named_columns():
    with pytest.raises(PerspectiveError, match="empty"):
        tensor_from_records([])
    with pytest.raises(PerspectiveError, match="Missing required record column"):
        tensor_from_records([{"draw": 0}])
    row = {"draw": 0, "strategy": "a", "perspective": "p", "net_benefit": 1}
    with pytest.raises(PerspectiveError, match="Duplicate"):
        tensor_from_records([row, row])


def test_switch_point_exposes_legacy_names_and_exact_weight():
    from voiage.methods.perspective_tensor import PerspectiveSwitchPoint

    switch = PerspectiveSwitchPoint("p", "q", 0.4, ("a", "b"), ("c",))
    assert switch.from_strategy == "a|b"
    assert switch.to_strategy == "c"
    assert switch.lower_right_weight == switch.upper_right_weight == switch.right_weight


def test_chained_weighted_perspectives_keep_their_provenance():
    original = NetBenefitTensor(np.ones((1, 1, 1)), ("a",), ("p",))
    first = original.with_weighted_perspective("one", {"p": 1})
    second = first.with_weighted_perspective("two", {"one": 2})
    assert second.attrs["weighted_perspectives"] == {
        "one": {"weights": {"p": 1}, "normalize": True},
        "two": {"weights": {"one": 2}, "normalize": True},
    }
    assert set(first.attrs["weighted_perspectives"]) == {"one"}
