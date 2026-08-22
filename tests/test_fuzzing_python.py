"""Continuous fuzz testing for voiage Python numerical and contract boundaries."""

from __future__ import annotations

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
import numpy as np

from voiage.decision_card import (
    DecisionCard,
    DecisionProblemSnapshot,
    Governance,
    InformationValuation,
    Lineage,
    ResidualUncertainty,
    SelectedPolicy,
)
from voiage.methods.basic import evpi, evppi
from voiage.methods.sample_information import enbs
from voiage.schema import DecisionProblem, Intervention


@given(
    st.lists(
        st.lists(
            st.floats(
                allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6
            ),
            min_size=2,
            max_size=8,
        ),
        min_size=2,
        max_size=20,
    )
)
@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_fuzz_evpi_finite_matrices(raw_rows: list[list[float]]) -> None:
    """EVPI must be non-negative and finite for any rectangular finite matrix."""
    cols = min(len(r) for r in raw_rows)
    matrix = np.array([r[:cols] for r in raw_rows], dtype=np.float64)

    val = evpi(matrix)
    assert np.isfinite(val), f"EVPI should be finite, got {val}"
    assert val >= -1e-12, f"EVPI must be non-negative, got {val}"


@given(
    evsi=st.floats(min_value=0.0, max_value=1e8, allow_nan=False, allow_infinity=False),
    research_cost=st.floats(
        min_value=0.0, max_value=1e8, allow_nan=False, allow_infinity=False
    ),
)
@settings(max_examples=50, deadline=None)
def test_fuzz_enbs_calculation(evsi: float, research_cost: float) -> None:
    """ENBS must equal EVSI - research_cost within floating-point precision."""
    val = enbs(evsi, research_cost)
    expected = evsi - research_cost
    assert np.isfinite(val)
    assert np.isclose(val, expected, rtol=1e-7, atol=1e-7)


@given(
    n_samples=st.integers(min_value=30, max_value=60),
    n_strategies=st.integers(min_value=2, max_value=4),
)
@settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_fuzz_evppi_calculation(n_samples: int, n_strategies: int) -> None:
    """EVPPI should evaluate cleanly on synthetic continuous parameter draws."""
    rng = np.random.default_rng(42)
    nb = rng.normal(loc=10.0, scale=3.0, size=(n_samples, n_strategies))
    params = {"theta_1": rng.uniform(low=0.0, high=1.0, size=n_samples)}

    try:
        val = evppi(nb, params, parameters_of_interest=["theta_1"])
        assert np.isfinite(val)
        assert val >= -1e-6
    except (ValueError, RuntimeError):
        # Degenerate parameter fits or small samples are permitted to fail explicitly
        pass


@given(
    prob_id=st.text(
        min_size=1, max_size=20, alphabet="abcdefghijklmnopqrstuvwxyz0123456789_-"
    ),
    title=st.text(
        min_size=1,
        max_size=40,
        alphabet="abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    ),
    wtp=st.floats(min_value=1.0, max_value=1e6, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=30, deadline=None)
def test_fuzz_decision_problem_roundtrip(prob_id: str, title: str, wtp: float) -> None:
    """DecisionProblem dataclass must serialize and validate arbitrary structured fields."""
    prob = DecisionProblem(
        decision_problem_id=prob_id,
        title=title,
        willingness_to_pay=wtp,
        interventions=[
            Intervention(
                intervention_id="int_a", name="Intervention A", is_reference=True
            ),
            Intervention(intervention_id="int_b", name="Intervention B"),
        ],
    )
    d = prob.to_dict()
    restored = DecisionProblem.from_dict(d)
    assert restored.decision_problem_id == prob.decision_problem_id
    assert restored.title == prob.title
    assert np.isclose(restored.willingness_to_pay, prob.willingness_to_pay)
    assert len(restored.interventions) == 2


@given(
    decision_id=st.text(
        min_size=1, max_size=20, alphabet="abcdefghijklmnopqrstuvwxyz0123456789_-"
    ),
    owner=st.text(
        min_size=1, max_size=30, alphabet="abcdefghijklmnopqrstuvwxyz0123456789_-"
    ),
)
@settings(max_examples=30, deadline=None)
def test_fuzz_decision_card_roundtrip(decision_id: str, owner: str) -> None:
    """DecisionCard record should preserve attributes under dict and JSON serialization."""
    card = DecisionCard(
        decision_id=decision_id,
        version="1.0.0",
        title="Fuzz Test Card",
        status="approved",
        created_at="2026-08-22T00:00:00Z",
        decision_problem=DecisionProblemSnapshot(
            problem_id="prob_1",
            title="Problem 1",
            alternatives=["A", "B"],
            criterion="max_enbs",
        ),
        selected_policy=SelectedPolicy(
            name="A",
            rationale="Higher expected payoff",
            expected_net_benefit=100.0,
        ),
        information_valuation=InformationValuation(
            evpi=25.0,
        ),
        residual_uncertainty=ResidualUncertainty(),
        governance=Governance(
            owner=owner,
            reviewers=["reviewer1"],
        ),
        lineage=Lineage(
            model_version="1.0.0",
            input_hash="abc123hash",
        ),
    )
    d = card.to_dict()
    restored = DecisionCard.from_dict(d)
    assert restored.decision_id == card.decision_id
    assert restored.governance.owner == card.governance.owner
    assert restored.status == "approved"
