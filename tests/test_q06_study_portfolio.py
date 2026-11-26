"""Tests for Q06 research design scheduling, portfolio optimization, and equity/sequential extensions."""

from __future__ import annotations

import pytest

from voiage.exceptions import InputError
from voiage.experimental.study_design import (
    discounted_population_enbs,
    enumerate_study_portfolio,
    equity_weighted_study_enbs,
    sequential_adaptive_stopping_regret,
)


def test_q06_discounted_population_enbs_and_expired_horizon() -> None:
    enbs_ok = discounted_population_enbs(
        per_patient_evsi=2.0,
        research_cost=5000.0,
        discount_rate=0.035,
        recruitment_delay_years=1.0,
        time_horizon_years=10.0,
        incident_population_per_year=1000.0,
    )
    assert enbs_ok > 0.0

    enbs_expired = discounted_population_enbs(
        per_patient_evsi=2.0,
        research_cost=5000.0,
        discount_rate=0.035,
        recruitment_delay_years=10.0,
        time_horizon_years=10.0,
        incident_population_per_year=1000.0,
    )
    assert enbs_expired == pytest.approx(-5000.0)


def test_q06_portfolio_overlap_guard_and_optimal_selection() -> None:
    with pytest.raises(InputError):
        enumerate_study_portfolio(
            study_ids=["s1", "s2", "s3"],
            costs=[100.0, 100.0, 80.0],
            standalone_evsis=[220.0, 210.0, 150.0],
            parameter_groups=["efficacy", "efficacy", "utility"],
            interactions=[],
            budget=250.0,
            require_explicit_overlap_guard=True,
        )

    opt = enumerate_study_portfolio(
        study_ids=["s1", "s2", "s3"],
        costs=[100.0, 100.0, 80.0],
        standalone_evsis=[220.0, 210.0, 150.0],
        parameter_groups=["efficacy", "efficacy", "utility"],
        interactions=[(0, 1, -180.0)],
        budget=250.0,
        require_explicit_overlap_guard=True,
    )
    assert opt["selected_indices"] == [0, 2]
    assert float(opt["enbs"]) == pytest.approx(190.0)


def test_q06_equity_weighted_and_sequential_regret() -> None:
    eq = equity_weighted_study_enbs(
        subgroup_evsi=[100.0, 200.0],
        equity_weights=[1.0, 3.0],
        research_cost=50.0,
    )
    assert eq == pytest.approx(125.0)

    seq = sequential_adaptive_stopping_regret(
        stage_evsi_remaining=[100.0, 180.0, 190.0],
        stage_cumulative_costs=[20.0, 60.0, 120.0],
    )
    assert seq["optimal_stage"] == 1
    assert float(seq["immediate_stop_regret"]) == pytest.approx(40.0)

    with pytest.raises(InputError):
        discounted_population_enbs(
            per_patient_evsi=-1.0,
            research_cost=5000.0,
            discount_rate=0.035,
            recruitment_delay_years=1.0,
            time_horizon_years=10.0,
            incident_population_per_year=1000.0,
        )
    with pytest.raises(InputError):
        equity_weighted_study_enbs(
            subgroup_evsi=[100.0],
            equity_weights=[0.0],
            research_cost=50.0,
        )
    with pytest.raises(InputError):
        sequential_adaptive_stopping_regret(
            stage_evsi_remaining=[],
            stage_cumulative_costs=[],
        )
