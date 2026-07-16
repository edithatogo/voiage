"""Runtime tests for adaptive learning and bandit VOI."""

import numpy as np
import pytest

from voiage.analysis import DecisionAnalysis
from voiage.exceptions import InputError
from voiage.methods.adaptive_learning_bandit import (
    value_of_adaptive_learning_bandit,
)
from voiage.schema import ValueArray

REWARDS = [[0.5, 0.6, 0.7, 0.8], [0.7, 0.8, 0.9, 1.0]]


def test_bandit_ucb_reports_regret_and_sampling_diagnostics() -> None:
    result = value_of_adaptive_learning_bandit(
        REWARDS,
        policy="ucb",
        exploration_cost=0.01,
        confidence=2.0,
        arm_names=["control", "adaptive"],
    )
    assert result.method_maturity == "fixture-backed"
    assert result.selected_arms.tolist() == [0, 1, 1, 0]
    assert result.total_reward == pytest.approx(3.0)
    assert result.regret == pytest.approx(0.4)
    assert result.sampling_burden == 4
    assert result.diagnostics["parity_status"] == "deferred"


@pytest.mark.parametrize("policy", ["thompson", "epsilon-greedy"])
def test_bandit_policies_are_deterministic(policy: str) -> None:
    first = value_of_adaptive_learning_bandit(REWARDS, policy=policy, seed=7)
    second = value_of_adaptive_learning_bandit(REWARDS, policy=policy, seed=7)
    np.testing.assert_array_equal(first.selected_arms, second.selected_arms)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"policy": "bad"}, "policy"),
        ({"horizon": 0}, "horizon"),
        ({"exploration_cost": -1.0}, "exploration_cost"),
        ({"epsilon": 2.0}, "epsilon"),
        ({"confidence": -1.0}, "confidence"),
        ({"arm_names": ["only"]}, "arm_names"),
    ],
)
def test_bandit_rejects_invalid_inputs(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(InputError, match=message):
        value_of_adaptive_learning_bandit(REWARDS, **kwargs)  # type: ignore[arg-type]


def test_bandit_stopping_rule_records_early_stop() -> None:
    result = value_of_adaptive_learning_bandit(REWARDS, stop_regret=1.0)
    assert result.stopping_step < 4
    assert len(result.selected_arms) == result.stopping_step


def test_bandit_wrapper_and_epsilon_exploration() -> None:
    analysis = DecisionAnalysis(
        ValueArray.from_numpy(np.asarray(REWARDS).T, ["control", "adaptive"])
    )
    result = analysis.value_of_adaptive_learning_bandit(
        policy="epsilon-greedy", epsilon=1.0, seed=3
    )
    assert result.policy == "epsilon-greedy"
    assert len(result.selected_arms) == 4


@pytest.mark.parametrize(
    ("rewards", "kwargs", "message"),
    [
        ([[1.0, 2.0]], {}, "non-empty"),
        ([[np.nan, 2.0]], {}, "finite"),
        (REWARDS, {"stop_regret": -1.0}, "stop_regret"),
    ],
)
def test_bandit_rejects_shape_and_nonfinite_inputs(
    rewards: list[list[float]], kwargs: dict[str, object], message: str
) -> None:
    if rewards == [[1.0, 2.0]]:
        rewards = []
    with pytest.raises(InputError, match=message):
        value_of_adaptive_learning_bandit(rewards, **kwargs)  # type: ignore[arg-type]
