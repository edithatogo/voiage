"""Fixture-backed value of adaptive learning for bandit allocation policies."""

from dataclasses import dataclass

import numpy as np

from voiage.config import DEFAULT_DTYPE
from voiage.exceptions import raise_input_error
from voiage.reporting import build_cheers_reporting


@dataclass(frozen=True)
class AdaptiveLearningBanditResult:
    """Structured result for sequential bandit learning value."""

    value: float
    policy: str
    arm_names: list[str]
    selected_arms: np.ndarray
    cumulative_rewards: np.ndarray
    total_reward: float
    baseline_reward: float
    regret: float
    opportunity_cost: float
    exploration_cost: float
    decision_switch_frequency: float
    sampling_burden: int
    stopping_step: int
    method_maturity: str
    diagnostics: dict[str, object]
    reporting: dict[str, object]


def value_of_adaptive_learning_bandit(
    reward_samples: np.ndarray | list[list[float]],
    *,
    policy: str = "ucb",
    horizon: int | None = None,
    exploration_cost: float = 0.0,
    epsilon: float = 0.1,
    confidence: float = 2.0,
    stop_regret: float | None = None,
    arm_names: list[str] | None = None,
    seed: int = 0,
) -> AdaptiveLearningBanditResult:
    """Estimate the value of sequential allocation and adaptive learning.

    ``reward_samples`` is an arm-by-sample matrix. Policies are deterministic
    for a fixed seed, making this surface suitable for contract fixtures while
    remaining explicitly fixture-backed pending parity and open-data evidence.
    """
    rewards = np.asarray(reward_samples, dtype=DEFAULT_DTYPE)
    if rewards.ndim != 2 or min(rewards.shape) < 1:
        raise_input_error("reward_samples must be a non-empty arm x sample matrix.")
    if not np.all(np.isfinite(rewards)):
        raise_input_error("reward_samples must be finite.")
    if policy not in {"thompson", "ucb", "epsilon-greedy"}:
        raise_input_error("policy must be thompson, ucb, or epsilon-greedy.")
    if horizon is None:
        horizon = int(rewards.shape[1])
    if horizon < 1 or horizon > rewards.shape[1]:
        raise_input_error("horizon must be between one and the sample count.")
    if exploration_cost < 0 or not np.isfinite(exploration_cost):
        raise_input_error("exploration_cost must be finite and non-negative.")
    if not 0 <= epsilon <= 1 or not np.isfinite(epsilon):
        raise_input_error("epsilon must be finite and between zero and one.")
    if confidence < 0 or not np.isfinite(confidence):
        raise_input_error("confidence must be finite and non-negative.")
    if stop_regret is not None and (stop_regret < 0 or not np.isfinite(stop_regret)):
        raise_input_error("stop_regret must be finite and non-negative.")

    names = arm_names or [f"arm_{idx + 1}" for idx in range(rewards.shape[0])]
    if len(names) != rewards.shape[0]:
        raise_input_error("arm_names length must match arm count.")
    rng = np.random.default_rng(seed)
    counts = np.zeros(rewards.shape[0], dtype=int)
    means = np.zeros(rewards.shape[0], dtype=DEFAULT_DTYPE)
    selected: list[int] = []
    outcomes: list[float] = []
    stop = horizon
    best_mean = float(np.max(np.mean(rewards[:, :horizon], axis=1)))
    for step in range(horizon):
        if step < rewards.shape[0]:
            arm = step
        elif policy == "ucb":
            bonus = confidence * np.sqrt(np.log(step + 1) / np.maximum(counts, 1))
            arm = int(np.argmax(means + bonus))
        elif policy == "epsilon-greedy" and rng.random() < epsilon:
            arm = int(rng.integers(rewards.shape[0]))
        elif policy == "thompson":
            draws = rng.normal(means, 1.0 / np.sqrt(np.maximum(counts, 1)))
            arm = int(np.argmax(draws))
        else:
            arm = int(np.argmax(means))
        outcome = float(rewards[arm, step])
        counts[arm] += 1
        means[arm] += (outcome - means[arm]) / counts[arm]
        selected.append(arm)
        outcomes.append(outcome)
        if stop_regret is not None:
            current_regret = (step + 1) * best_mean - float(np.sum(outcomes))
            if current_regret <= stop_regret and step + 1 >= rewards.shape[0]:
                stop = step + 1
                break

    selected_array = np.asarray(selected, dtype=int)
    cumulative = np.cumsum(np.asarray(outcomes, dtype=DEFAULT_DTYPE))
    total_reward = float(cumulative[-1])
    baseline_reward = best_mean * len(selected)
    regret = max(0.0, baseline_reward - total_reward)
    switches = int(np.sum(np.diff(selected_array) != 0)) if len(selected) > 1 else 0
    value = max(0.0, total_reward - baseline_reward - exploration_cost * len(selected))
    diagnostics: dict[str, object] = {
        "policy": policy,
        "horizon": len(selected),
        "expected_value_of_continued_learning": value,
        "stopping_rule": stop_regret is not None,
        "parity_status": "deferred",
        "open_data_status": "blocked: no licensed online allocation trace committed",
    }
    return AdaptiveLearningBanditResult(
        value=value,
        policy=policy,
        arm_names=list(names),
        selected_arms=selected_array,
        cumulative_rewards=cumulative,
        total_reward=total_reward,
        baseline_reward=baseline_reward,
        regret=regret,
        opportunity_cost=regret,
        exploration_cost=exploration_cost * len(selected),
        decision_switch_frequency=switches / max(1, len(selected) - 1),
        sampling_burden=int(np.sum(counts)),
        stopping_step=stop,
        method_maturity="fixture-backed",
        diagnostics=diagnostics,
        reporting=build_cheers_reporting(
            analysis_type="value_of_adaptive_learning_bandit",
            method_family="adaptive_learning_bandit",
            method_maturity="fixture-backed",
            diagnostics=diagnostics,
        ),
    )
