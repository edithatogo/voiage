"""Check the planning examples by finite enumeration, not package execution."""

from __future__ import annotations

from fractions import Fraction
from itertools import product
import json
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = (
    ROOT
    / "conductor/tracks/simulation_outcome_boundary_20261107"
    / "vop-voh-reference-cases.json"
)


@pytest.fixture(scope="module")
def cases() -> dict[str, dict[str, Any]]:
    data = json.loads(EXAMPLES.read_text(), parse_float=Fraction)
    assert data["status"] == "planning_examples_not_runtime_or_accepted_registry"
    result = {case["id"]: case for case in data["cases"]}
    assert len(result) == len(data["cases"]) == 7
    return result


def _policy_values(
    values: list[list[list[int]]],
    weights: list[Fraction],
    state_mass: list[Fraction],
) -> tuple[Fraction, Fraction]:
    groups = len(weights)
    strategies = len(values[0][0])

    def evaluate(policy: tuple[int, ...]) -> Fraction:
        return sum(
            state_mass[state] * weights[group] * values[state][group][policy[group]]
            for state in range(len(values))
            for group in range(groups)
        )

    common = max(evaluate((strategy,) * groups) for strategy in range(strategies))
    tailored = max(
        evaluate(policy) for policy in product(range(strategies), repeat=groups)
    )
    return common, tailored


@pytest.mark.parametrize(
    "case_id",
    [
        "population-weighted",
        "information-interaction",
        "signed-difference",
        "no-heterogeneity",
    ],
)
def test_finite_policy_and_information_values(
    cases: dict[str, dict[str, Any]], case_id: str
) -> None:
    case = cases[case_id]
    values, weights, probabilities = (
        case["values"],
        case["population_weights"],
        case["state_probabilities"],
    )
    assert sum(weights) == sum(probabilities) == 1
    assert all(weight >= 0 for weight in weights + probabilities)
    c0, cf = _policy_values(values, weights, probabilities)
    perfect = [
        _policy_values([state], weights, [probability])
        for state, probability in zip(values, probabilities, strict=True)
    ]
    p0, pf = (sum(result[index] for result in perfect) for index in (0, 1))
    actual = {
        "C0": c0,
        "Cf": cf,
        "P0": p0,
        "Pf": pf,
        "static": cf - c0,
        "dynamic": pf - p0,
        "EVPI0": p0 - c0,
        "EVPIf": pf - cf,
        "dynamic_minus_static": (pf - p0) - (cf - c0),
    }
    assert actual["dynamic_minus_static"] == actual["EVPIf"] - actual["EVPI0"]
    if "signal_likelihood" in case:
        likelihood = case["signal_likelihood"]
        assert all(sum(row) == 1 and all(x >= 0 for x in row) for row in likelihood)
        signal_values = [
            _policy_values(
                values,
                weights,
                [
                    p * likelihood[state][signal]
                    for state, p in enumerate(probabilities)
                ],
            )
            for signal in range(len(likelihood[0]))
        ]
        s0, sf = (sum(result[index] for result in signal_values) for index in (0, 1))
        cost = case["study_cost_per_person"]
        actual.update(
            S0=s0,
            Sf=sf,
            EVSI0=s0 - c0,
            EVSIf=sf - cf,
            net_EVSI0=s0 - c0 - cost,
            net_EVSIf=sf - cf - cost,
        )
        assert (sf - s0) - (cf - c0) == actual["EVSIf"] - actual["EVSI0"]
    assert {name: actual[name] for name in case["expected"]} == case["expected"]


def test_directional_means_and_switch(cases: dict[str, dict[str, Any]]) -> None:
    case = cases["directional"]
    p, q = case["means"]["p"], case["means"]["q"]
    p_choice, q_choice = p.index(max(p)), q.index(max(q))
    switch = Fraction(p[0] - p[1], (p[0] - p[1]) - (q[0] - q[1]))
    assert case["expected"] == {
        "p_to_q": q[q_choice] - q[p_choice],
        "q_to_p": p[p_choice] - p[q_choice],
        "q_mixture_switch": switch,
    }


def test_current_and_per_draw_decisions_are_distinct(
    cases: dict[str, dict[str, Any]],
) -> None:
    case = cases["current-versus-draw"]
    p, q = case["values"]["p"], case["values"]["q"]
    probabilities = case["state_probabilities"]
    means = [
        [
            sum(probabilities[s] * outcomes[s][a] for s in range(len(p)))
            for a in range(2)
        ]
        for outcomes in (p, q)
    ]
    p_choice, q_choice = (row.index(max(row)) for row in means)
    current = means[1][q_choice] - means[1][p_choice]
    per_draw = sum(
        probabilities[s] * (max(q[s]) - q[s][p[s].index(max(p[s]))])
        for s in range(len(p))
    )
    assert case["expected"] == {
        "current_p_to_q": current,
        "per_draw_p_to_q": per_draw,
    }
    assert current != per_draw


def test_joint_perspective_and_subgroup_policies(
    cases: dict[str, dict[str, Any]],
) -> None:
    case = cases["joint-perspective-group"]
    p, q = case["values"]["p"], case["values"]["q"]
    weights = case["population_weights"]
    policies = list(product(range(2), repeat=len(weights)))

    def evaluate(values: list[list[int]], policy: tuple[int, ...]) -> Fraction:
        return sum(weights[g] * values[g][policy[g]] for g in range(len(weights)))

    common = [policy for policy in policies if len(set(policy)) == 1]
    p_common = max(common, key=lambda policy: evaluate(p, policy))
    q_common = max(common, key=lambda policy: evaluate(q, policy))
    p_tailored = max(policies, key=lambda policy: evaluate(p, policy))
    q_tailored = max(policies, key=lambda policy: evaluate(q, policy))
    assert case["expected"] == {
        "common_p_to_q": evaluate(q, q_common) - evaluate(q, p_common),
        "tailored_p_to_q": evaluate(q, q_tailored) - evaluate(q, p_tailored),
        "p_static_VOH": evaluate(p, p_tailored) - evaluate(p, p_common),
        "q_static_VOH": evaluate(q, q_tailored) - evaluate(q, q_common),
    }
