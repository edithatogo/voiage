"""Deterministic tests for health economics helper calculations."""

import jax.numpy as jnp
import numpy as np
import pytest

import voiage.health_economics as health_economics_module
from voiage.health_economics import (
    HealthEconomicsAnalysis,
    HealthState,
    Treatment,
    calculate_icer_simple,
)


def _build_analysis() -> HealthEconomicsAnalysis:
    return HealthEconomicsAnalysis(
        willingness_to_pay=50_000.0,
        currency="AUD",
    )


def test_calculate_qaly_uses_default_duration_without_discounting() -> None:
    """QALY should fall back to the state's duration when no horizon is given."""
    analysis = _build_analysis()
    state = HealthState(
        state_id="stable",
        description="Stable disease",
        utility=0.8,
        cost=1_000.0,
        duration=5.0,
    )

    qaly = analysis.calculate_qaly(state, time_horizon=None, discount_rate=0.0)

    assert qaly == pytest.approx(4.0)


def test_calculate_qaly_applies_discounting_and_caps_value() -> None:
    """Discounted QALY should match the current continuous-discount formula."""
    analysis = _build_analysis()
    state = HealthState(
        state_id="stable",
        description="Stable disease",
        utility=0.8,
        cost=1_000.0,
        duration=5.0,
    )

    qaly = analysis.calculate_qaly(state, time_horizon=4.0, discount_rate=0.03)
    expected = 0.8 * (1 - np.exp(-0.03 * 4.0)) / 0.03

    assert qaly == pytest.approx(expected)
    assert qaly <= 4.0


def test_calculate_cost_applies_discounting_and_caps_value() -> None:
    """Discounted costs should match the current continuous-discount formula."""
    analysis = _build_analysis()
    state = HealthState(
        state_id="stable",
        description="Stable disease",
        utility=0.8,
        cost=2_000.0,
        duration=5.0,
    )

    cost = analysis.calculate_cost(state, time_horizon=4.0, discount_rate=0.05)
    expected = 2_000.0 * (1 - np.exp(-0.05 * 4.0)) / 0.05

    assert cost == pytest.approx(expected)
    assert cost <= 8_000.0


def test_calculate_icer_returns_infinity_for_non_positive_incremental_qaly() -> None:
    """Dominated interventions should report an infinite ICER."""
    analysis = _build_analysis()
    treatment = Treatment(
        name="New treatment",
        description="Less effective option",
        effectiveness=0.6,
        cost_per_cycle=2_000.0,
        cycles_required=1,
    )
    comparator = Treatment(
        name="Comparator",
        description="Current standard",
        effectiveness=0.8,
        cost_per_cycle=1_500.0,
        cycles_required=1,
    )
    health_states = [
        HealthState(
            state_id="stable",
            description="Stable disease",
            utility=0.9,
            cost=0.0,
            duration=1.0,
        )
    ]

    icer = analysis.calculate_icer(treatment, comparator, health_states)

    assert icer == float("inf")


def test_calculate_net_monetary_benefit_uses_default_states() -> None:
    """NMB should use the internally generated default health states."""
    analysis = _build_analysis()
    treatment = Treatment(
        name="Drug A",
        description="Reference treatment",
        effectiveness=0.85,
        cost_per_cycle=120.0,
        cycles_required=10,
    )

    nmb = analysis.calculate_net_monetary_benefit(treatment)

    assert isinstance(nmb, float)
    assert np.isfinite(nmb)


def test_create_cost_effectiveness_acceptability_curve_shapes_and_bounds() -> None:
    """The CEAC helper should return paired arrays over the requested WTP range."""
    analysis = _build_analysis()
    treatment = Treatment(
        name="Drug A",
        description="Reference treatment",
        effectiveness=0.85,
        cost_per_cycle=120.0,
        cycles_required=10,
    )
    health_states = [
        HealthState(
            state_id="stable",
            description="Stable disease",
            utility=0.9,
            cost=25.0,
            duration=1.0,
        )
    ]

    wtp_values, probabilities = analysis.create_cost_effectiveness_acceptability_curve(
        treatment=treatment,
        health_states=health_states,
        wtp_range=(0.0, 100_000.0),
        num_points=25,
    )

    assert np.asarray(wtp_values).shape == (25,)
    probability_array = np.asarray(probabilities)
    assert probability_array.ndim == 0 or probability_array.shape == (25,)
    assert np.all(probability_array >= 0.0)
    assert np.all(probability_array <= 1.0)


@pytest.mark.parametrize(
    ("effectiveness", "expected_first_state"),
    [
        (0.9, "Healthy"),
        (0.6, "Improved"),
        (0.4, "Slight"),
    ],
)
def test_create_default_health_states_covers_effectiveness_branches(
    effectiveness: float,
    expected_first_state: str,
) -> None:
    """Default health states should change with the treatment effectiveness band."""
    analysis = _build_analysis()
    treatment = Treatment(
        name="Band test",
        description="Treatment to drive default-state branching",
        effectiveness=effectiveness,
        cost_per_cycle=100.0,
        cycles_required=5,
    )

    states = analysis._create_default_health_states(treatment)

    assert [state.state_id for state in states] == [
        expected_first_state,
        states[1].state_id,
        "Dead",
    ]
    assert len(states) == 3


def test_calculate_treatment_totals_accumulates_costs_and_weighted_qalys() -> None:
    """Treatment totals should combine direct treatment costs and weighted state QALYs."""
    analysis = _build_analysis()
    treatment = Treatment(
        name="Drug A",
        description="Treatment with side effects",
        effectiveness=0.8,
        cost_per_cycle=150.0,
        cycles_required=4,
        side_effect_cost=75.0,
    )
    health_states = [
        HealthState(
            state_id="stable",
            description="Stable disease",
            utility=0.8,
            cost=100.0,
            duration=2.0,
        ),
        HealthState(
            state_id="improved",
            description="Improved disease",
            utility=0.6,
            cost=50.0,
            duration=1.0,
        ),
    ]

    total_cost, total_qaly = analysis._calculate_treatment_totals(
        treatment, health_states
    )
    expected_qaly = sum(
        analysis.calculate_qaly(state) * state.utility for state in health_states
    )

    assert total_cost == pytest.approx(675.0)
    assert total_qaly == pytest.approx(expected_qaly)


def test_budget_impact_analysis_returns_consistent_summary() -> None:
    """Budget impact analysis should report annualised and total impacts consistently."""
    analysis = _build_analysis()
    treatment = Treatment(
        name="Drug A",
        description="Reference treatment",
        effectiveness=0.7,
        cost_per_cycle=200.0,
        cycles_required=3,
        side_effect_cost=50.0,
    )

    result = analysis.budget_impact_analysis(
        treatment=treatment,
        population_size=1_000,
        adoption_rate=0.25,
        time_horizon=5.0,
        annual_budget=100_000.0,
    )

    assert result["patients_treated"] == pytest.approx(250.0)
    assert result["total_budget_impact"] == pytest.approx(812_500.0)
    assert result["annual_budget_impact"] == pytest.approx(162_500.0)
    assert result["budget_impact_percentage"] == pytest.approx(162.5)
    assert result["sustainability_score"] == pytest.approx(0.0)


def test_probabilistic_sensitivity_analysis_returns_expected_summary_keys() -> None:
    """PSA should produce stable summary dictionaries for each output dimension."""
    analysis = _build_analysis()
    treatment = Treatment(
        name="Drug A",
        description="Reference treatment",
        effectiveness=0.75,
        cost_per_cycle=175.0,
        cycles_required=4,
        side_effect_utility=0.02,
        side_effect_cost=40.0,
    )
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        analysis, "calculate_qaly", lambda state: jnp.asarray(state.utility * 2.0)
    )
    monkeypatch.setattr(
        analysis, "calculate_cost", lambda state: jnp.asarray(state.cost)
    )
    monkeypatch.setattr(analysis, "calculate_icer", lambda _: jnp.asarray(123.0))
    monkeypatch.setattr(
        analysis,
        "calculate_net_monetary_benefit",
        lambda _, health_states: jnp.asarray(
            health_states[0].utility * 50_000.0 - health_states[0].cost
        ),
    )

    try:
        result = analysis.probabilistic_sensitivity_analysis(
            treatment=treatment,
            num_simulations=32,
        )
    finally:
        monkeypatch.undo()

    assert set(result) == {
        "qaly_distribution",
        "cost_distribution",
        "icer_distribution",
        "net_monetary_benefit",
        "simulation_parameters",
    }
    for key in (
        "qaly_distribution",
        "cost_distribution",
        "icer_distribution",
        "net_monetary_benefit",
    ):
        summary = result[key]
        assert {"mean", "q025", "q975"} <= set(summary)
        assert all(np.isfinite(value) for value in summary.values())
    assert result["simulation_parameters"]["num_simulations"] == 32


def test_health_decision_outcomes_uses_custom_decision_function() -> None:
    """Custom decision functions should bypass the default health-state workflow."""
    analysis = _build_analysis()
    treatment = Treatment(
        name="Drug A",
        description="Reference treatment",
        effectiveness=0.8,
        cost_per_cycle=120.0,
        cycles_required=2,
    )
    comparator = Treatment(
        name="Comparator",
        description="Current standard",
        effectiveness=0.7,
        cost_per_cycle=100.0,
        cycles_required=2,
    )

    result = analysis._health_decision_outcomes(
        [treatment, comparator],
        lambda selected_treatment, **kwargs: {
            "chosen": selected_treatment.name,
            "context": kwargs["label"],
        },
        label="screened",
    )

    assert result == [
        {"chosen": "Drug A", "context": "screened"},
        {"chosen": "Comparator", "context": "screened"},
    ]


def test_health_decision_outcomes_returns_default_summary_when_no_callback() -> None:
    """Default health decision outcomes should expose the expected summary metrics."""
    analysis = _build_analysis()
    treatment = Treatment(
        name="Drug A",
        description="Reference treatment",
        effectiveness=0.85,
        cost_per_cycle=120.0,
        cycles_required=4,
    )
    comparator = Treatment(
        name="Comparator",
        description="Current standard",
        effectiveness=0.65,
        cost_per_cycle=100.0,
        cycles_required=4,
    )

    result = analysis._health_decision_outcomes(
        [treatment, comparator],
        decision_function="default",
    )

    assert len(result) == 2
    for outcome in result:
        assert set(outcome) == {"qaly", "cost", "nmb", "icer"}
        assert all(np.isfinite(outcome[key]) for key in ("qaly", "cost", "nmb"))
        assert isinstance(outcome["icer"], float)


def test_create_voi_analysis_for_health_decisions_populates_stubbed_analysis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The VOI factory should configure the returned analysis object with inputs and callback."""

    class StubDecisionAnalysis:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs
            self.decision_function = None

    analysis = _build_analysis()
    treatment = Treatment(
        name="Drug A",
        description="Reference treatment",
        effectiveness=0.85,
        cost_per_cycle=120.0,
        cycles_required=4,
    )

    monkeypatch.setattr(
        health_economics_module,
        "DecisionAnalysis",
        StubDecisionAnalysis,
    )

    voi_analysis = analysis.create_voi_analysis_for_health_decisions(
        treatments=[treatment],
        decision_outcome_function=lambda selected_treatment, **kwargs: {
            "name": selected_treatment.name,
            "label": kwargs["label"],
        },
        additional_parameters={"extra_parameter": "kept"},
    )

    assert isinstance(voi_analysis, StubDecisionAnalysis)
    assert voi_analysis.kwargs["backend"] == "jax"
    assert voi_analysis.kwargs["treatments"] == [treatment]
    assert voi_analysis.kwargs["willingness_to_pay"] == 50_000.0
    assert voi_analysis.kwargs["currency"] == "AUD"
    assert voi_analysis.kwargs["analysis_type"] == "health_economics"
    assert voi_analysis.kwargs["extra_parameter"] == "kept"
    assert callable(voi_analysis.decision_function)
    assert voi_analysis.decision_function({}, label="checked") == [
        {"name": "Drug A", "label": "checked"}
    ]


def test_calculate_icer_simple_returns_expected_ratio_and_infinity() -> None:
    """The simple ICER helper should mirror the public branch behaviour exactly."""
    assert calculate_icer_simple(10_000.0, 2.0, 8_000.0, 1.0) == pytest.approx(2_000.0)
    assert calculate_icer_simple(10_000.0, 1.0, 8_000.0, 1.0) == float("inf")
