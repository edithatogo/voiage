"""R01-R03 frontier research contracts and comparative evaluation tests."""

from __future__ import annotations

import pytest

from voiage.experimental.study_design import (
    enumerate_study_portfolio,
    sequential_adaptive_stopping_regret,
)


def test_r01_r03_frontier_regret_and_portfolio_bounds() -> None:
    seq = sequential_adaptive_stopping_regret(
        stage_evsi_remaining=[80.0, 150.0, 155.0],
        stage_cumulative_costs=[10.0, 40.0, 100.0],
    )
    assert seq["optimal_stage"] == 1
    assert float(seq["optimal_net_value"]) == pytest.approx(110.0)
    assert float(seq["immediate_stop_regret"]) == pytest.approx(40.0)

    port = enumerate_study_portfolio(
        study_ids=["trial_a", "impl_b"],
        costs=[50.0, 30.0],
        standalone_evsis=[120.0, 60.0],
        parameter_groups=["efficacy", "implementation"],
        interactions=[],
        budget=100.0,
        require_explicit_overlap_guard=True,
    )
    assert port["selected_indices"] == [0, 1]
    assert float(port["enbs"]) == pytest.approx(100.0)
