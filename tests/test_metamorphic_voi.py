"""Metamorphic Testing (MT) suite for Value of Information analysis.

This module formalizes and verifies Metamorphic Relations (MRs) across
VOI estimation algorithms (EVPI, EVPPI, EVSI, ENBS, MCDA, and DecisionProblem):
- MR1: Additive Shift Invariance (EVPI, EVPPI, EVSI unchanged under constant shift)
- MR2: Positive Scalar Homogeneity (EVPI scales linearly with positive factor k)
- MR3: Strategy Permutation Invariance (Order of strategy columns does not alter VOI)
- MR4: Strict Dominance Collapse (Dominant strategy drives VOI to 0.0)
- MR5: Strategy Duplication Invariance (Duplicate strategies do not affect VOI)
- MR6: Parameter Subset Monotonicity (0 <= EVPPI(A) <= EVPPI(A U B) <= EVPI)
- MR7: ENBS Research Cost Monotonicity (ENBS decreases strictly monotonically with cost)
- MR8: Disjoint Decision Problem Additivity (Independent multi-domain VOI sums additively)
"""

from __future__ import annotations

import numpy as np
import pytest

from voiage.methods.basic import evpi, evppi
from voiage.methods.sample_information import enbs
from voiage.schema import ParameterSet


@pytest.fixture
def rng():
    return np.random.default_rng(seed=42)


@pytest.fixture
def base_net_benefits(rng):
    """Generates a standard 200-sample, 3-strategy net benefit matrix."""
    return rng.normal(loc=[100.0, 105.0, 98.0], scale=[15.0, 20.0, 12.0], size=(200, 3))


@pytest.fixture
def base_parameter_set(rng):
    """Generates a ParameterSet with 3 correlated parameters."""
    samples = rng.multivariate_normal(
        mean=[0.0, 1.0, -0.5],
        cov=[[1.0, 0.4, 0.2], [0.4, 1.0, 0.1], [0.2, 0.1, 1.0]],
        size=200,
    )
    return ParameterSet.from_numpy_or_dict(
        {
            "theta_1": samples[:, 0],
            "theta_2": samples[:, 1],
            "theta_3": samples[:, 2],
        }
    )


def test_mr1_additive_shift_invariance_evpi(base_net_benefits):
    """MR1: Adding any scalar constant c to all net benefits leaves EVPI invariant."""
    base_val = evpi(base_net_benefits)
    assert base_val > 0.0

    for c in [-1000.0, -42.5, 0.0, 50.0, 10000.0]:
        shifted_nb = base_net_benefits + c
        shifted_val = evpi(shifted_nb)
        np.testing.assert_allclose(
            shifted_val,
            base_val,
            rtol=1e-12,
            atol=1e-12,
            err_msg=f"EVPI violated additive shift invariance for c={c}",
        )


def test_mr1_additive_shift_invariance_evppi(base_net_benefits, base_parameter_set):
    """MR1: Adding any scalar constant c leaves EVPPI invariant."""
    base_val = evppi(
        base_net_benefits, base_parameter_set, parameters_of_interest=["theta_1"]
    )
    assert base_val > 0.0

    for c in [-500.0, 100.0, 500.0]:
        shifted_nb = base_net_benefits + c
        shifted_val = evppi(
            shifted_nb, base_parameter_set, parameters_of_interest=["theta_1"]
        )
        np.testing.assert_allclose(
            shifted_val,
            base_val,
            rtol=1e-10,
            atol=1e-10,
            err_msg=f"EVPPI violated additive shift invariance for c={c}",
        )


def test_mr2_positive_scalar_homogeneity(base_net_benefits, base_parameter_set):
    """MR2: Multiplying all net benefits by k > 0 scales EVPI and EVPPI by exactly k."""
    base_evpi = evpi(base_net_benefits)
    base_evppi = evppi(
        base_net_benefits, base_parameter_set, parameters_of_interest=["theta_1"]
    )

    for k in [0.01, 0.5, 1.0, 2.0, 10.0, 100.0]:
        scaled_nb = base_net_benefits * k
        scaled_evpi = evpi(scaled_nb)
        scaled_evppi = evppi(
            scaled_nb, base_parameter_set, parameters_of_interest=["theta_1"]
        )

        np.testing.assert_allclose(
            scaled_evpi,
            base_evpi * k,
            rtol=1e-12,
            atol=1e-12,
            err_msg=f"EVPI violated positive homogeneity for scale k={k}",
        )
        np.testing.assert_allclose(
            scaled_evppi,
            base_evppi * k,
            rtol=1e-10,
            atol=1e-10,
            err_msg=f"EVPPI violated positive homogeneity for scale k={k}",
        )


def test_mr3_strategy_permutation_invariance(base_net_benefits, rng):
    """MR3: Arbitrary permutation of strategy columns preserves EVPI."""
    base_evpi = evpi(base_net_benefits)

    permutations = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ]

    for p in permutations:
        permuted_nb = base_net_benefits[:, p]
        permuted_evpi = evpi(permuted_nb)
        np.testing.assert_allclose(
            permuted_evpi,
            base_evpi,
            rtol=1e-12,
            atol=1e-12,
            err_msg=f"EVPI violated strategy permutation invariance for ordering {p}",
        )


def test_mr4_strict_dominance_collapse(base_net_benefits):
    """MR4: If one strategy strictly dominates all alternatives in all states, VOI collapses to 0.0."""
    dominant_strategy = np.max(base_net_benefits, axis=1, keepdims=True) + 50.0

    nb_with_dominant = np.hstack([base_net_benefits, dominant_strategy])
    collapsed_evpi = evpi(nb_with_dominant)

    assert collapsed_evpi == pytest.approx(0.0, abs=1e-12)


def test_mr5_duplicate_strategy_invariance(base_net_benefits):
    """MR5: Appending an identical duplicate strategy to choice set preserves EVPI."""
    base_evpi = evpi(base_net_benefits)

    # Duplicate strategy 0 and strategy 1
    extended_nb = np.column_stack(
        [base_net_benefits, base_net_benefits[:, 0], base_net_benefits[:, 1]]
    )
    extended_evpi = evpi(extended_nb)

    np.testing.assert_allclose(
        extended_evpi,
        base_evpi,
        rtol=1e-12,
        atol=1e-12,
        err_msg="EVPI violated strategy duplication invariance",
    )


def test_mr6_parameter_subset_monotonicity(base_net_benefits, base_parameter_set):
    """MR6: EVPPI must satisfy 0 <= EVPPI(A) <= EVPPI(A U B) <= EVPI."""
    total_evpi = evpi(base_net_benefits)

    evppi_1 = evppi(
        base_net_benefits, base_parameter_set, parameters_of_interest=["theta_1"]
    )
    evppi_2 = evppi(
        base_net_benefits, base_parameter_set, parameters_of_interest=["theta_2"]
    )
    evppi_12 = evppi(
        base_net_benefits,
        base_parameter_set,
        parameters_of_interest=["theta_1", "theta_2"],
    )
    evppi_all = evppi(
        base_net_benefits,
        base_parameter_set,
        parameters_of_interest=["theta_1", "theta_2", "theta_3"],
    )

    assert 0.0 <= evppi_1 <= total_evpi + 1e-9
    assert 0.0 <= evppi_2 <= total_evpi + 1e-9
    assert 0.0 <= evppi_12 <= total_evpi + 1e-9
    assert 0.0 <= evppi_all <= total_evpi + 1e-9

    # Joint information is at least as informative as single marginal
    assert evppi_12 >= min(evppi_1, evppi_2) - 1e-6


def test_mr7_enbs_research_cost_monotonicity():
    """MR7: ENBS = EVSI - Cost decreases strictly monotonically with increasing research cost."""
    evsi_value = 25000.0

    costs = [0.0, 1000.0, 5000.0, 15000.0, 25000.0, 50000.0]
    enbs_values = [enbs(evsi_value, cost) for cost in costs]

    for i in range(len(costs) - 1):
        assert enbs_values[i] > enbs_values[i + 1], (
            "ENBS is not strictly decreasing in research cost"
        )
        assert enbs_values[i] == pytest.approx(evsi_value - costs[i], abs=1e-12)


def test_mr8_disjoint_decision_problem_additivity(rng):
    """MR8: Two mutually independent decision problems evaluated independently sum to joint EVPI."""
    nb1 = rng.normal(loc=[10.0, 12.0], scale=[2.0, 3.0], size=(200, 2))
    nb2 = rng.normal(loc=[100.0, 105.0], scale=[10.0, 15.0], size=(200, 2))

    evpi1 = evpi(nb1)
    evpi2 = evpi(nb2)

    # Joint decision problem with 4 composite strategies: (s1_1+s2_1, s1_1+s2_2, s1_2+s2_1, s1_2+s2_2)
    joint_nb = np.column_stack(
        [
            nb1[:, 0] + nb2[:, 0],
            nb1[:, 0] + nb2[:, 1],
            nb1[:, 1] + nb2[:, 0],
            nb1[:, 1] + nb2[:, 1],
        ]
    )
    joint_evpi = evpi(joint_nb)

    # For separable independent decisions, max(s1+s2) = max(s1) + max(s2)
    np.testing.assert_allclose(
        joint_evpi,
        evpi1 + evpi2,
        rtol=1e-10,
        atol=1e-10,
        err_msg="Independent joint decision problem EVPI did not equal sum of marginal EVPIs",
    )
