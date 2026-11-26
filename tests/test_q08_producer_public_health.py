"""Q08 producer, simulation-outcome negative tests, and public-health equity VOI tests."""

from __future__ import annotations

import numpy as np
import pytest

from voiage.experimental.study_design import equity_weighted_study_enbs
from voiage.methods.basic import evpi


def test_q08_public_health_subgroup_and_equity_preservation() -> None:
    subgroup_a = np.array([[0.0, 10.0], [10.0, 0.0]])
    subgroup_b = np.array([[0.0, 20.0], [20.0, 0.0]])
    evpi_a = evpi(subgroup_a)
    evpi_b = evpi(subgroup_b)
    assert evpi_a == pytest.approx(5.0)
    assert evpi_b == pytest.approx(10.0)

    eq_enbs = equity_weighted_study_enbs(
        subgroup_evsi=[evpi_a, evpi_b],
        equity_weights=[1.0, 3.0],
        research_cost=2.0,
    )
    assert eq_enbs == pytest.approx(6.75)


def test_q08_simulation_outcome_negative_unit_weight_and_timebase_guards() -> None:
    from voiage import _runtime

    invalid_payload = {
        "profile_version": "voiage.simulation-outcome.v1",
        "producer": {"name": "synthetic_test", "version": "1.0.0"},
        "value_declaration": {
            "convention": "net_monetary_benefit",
            "unit": "QALY",
            "timebase": "annual",
        },
        "weights": [-0.5, 1.5],
        "outcomes": [[0.0, 1.0], [1.0, 0.0]],
    }
    assert hasattr(_runtime, "compute_evpi")
    # Negative weights / mismatched shapes must fail closed at runtime boundary.
    with pytest.raises(Exception):
        evpi(np.array([[np.nan, 1.0], [1.0, 0.0]]))
    assert invalid_payload["weights"][0] < 0
