"""Tests for dominance and ICER analysis."""

import numpy as np
import pytest

from voiage.exceptions import InputError
from voiage.methods.dominance import (
    DominanceResult,
    calculate_dominance,
    calculate_extended_dominance,
    calculate_icers,
    calculate_strong_dominance,
    cost_effectiveness_frontier,
)


@pytest.fixture
def dominance_example() -> tuple[np.ndarray, np.ndarray, list[str]]:
    costs = np.array([100.0, 200.0, 500.0, 800.0, 900.0])
    effects = np.array([1.0, 2.0, 2.5, 4.0, 3.5])
    names = ["A", "B", "C", "D", "E"]
    return costs, effects, names


def test_calculate_dominance_classifies_frontier_and_dominated_strategies(
    dominance_example: tuple[np.ndarray, np.ndarray, list[str]],
) -> None:
    costs, effects, names = dominance_example

    result = calculate_dominance(costs, effects, names)

    assert isinstance(result, DominanceResult)
    assert result.frontier_indices == [0, 1, 3]
    assert result.strongly_dominated_indices == [4]
    assert result.extended_dominated_indices == [2]
    assert result.status == [
        "frontier",
        "frontier",
        "extended_dominated",
        "frontier",
        "strongly_dominated",
    ]
    assert result.incremental_costs.tolist() == pytest.approx([100.0, 600.0])
    assert result.incremental_effects.tolist() == pytest.approx([1.0, 2.0])
    assert result.icers.tolist() == pytest.approx([100.0, 300.0])
    assert result.reporting["reporting_standard"] == "CHEERS-VOI"
    assert result.reporting["analysis_type"] == "calculate_dominance"


def test_dominance_helper_functions(
    dominance_example: tuple[np.ndarray, np.ndarray, list[str]],
) -> None:
    costs, effects, _names = dominance_example

    assert calculate_strong_dominance(costs, effects) == [4]
    assert calculate_extended_dominance(costs, effects) == [2]
    assert cost_effectiveness_frontier(costs, effects) == [0, 1, 3]

    incremental_costs, incremental_effects, icers = calculate_icers(
        costs,
        effects,
        [0, 1, 3],
    )
    assert incremental_costs.tolist() == pytest.approx([100.0, 600.0])
    assert incremental_effects.tolist() == pytest.approx([1.0, 2.0])
    assert icers.tolist() == pytest.approx([100.0, 300.0])


def test_calculate_icers_handles_single_frontier_strategy() -> None:
    incremental_costs, incremental_effects, icers = calculate_icers(
        [100.0, 200.0],
        [1.0, 1.0],
        [0],
    )

    assert incremental_costs.size == 0
    assert incremental_effects.size == 0
    assert icers.size == 0


def test_calculate_dominance_rejects_invalid_inputs() -> None:
    with pytest.raises(InputError, match="1D"):
        calculate_dominance(np.asarray([[1.0, 2.0]]), [1.0, 2.0])

    with pytest.raises(InputError, match="same length"):
        calculate_dominance([1.0, 2.0], [1.0])

    with pytest.raises(InputError, match="At least two"):
        calculate_dominance([1.0], [1.0])

    with pytest.raises(InputError, match="finite"):
        calculate_dominance([1.0, np.nan], [1.0, 2.0])

    with pytest.raises(InputError, match="strategy_names"):
        calculate_dominance([1.0, 2.0], [1.0, 2.0], strategy_names=["A"])
