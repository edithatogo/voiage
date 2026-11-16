"""Tests for Voiage-side Innovate DecisionBinding adapter (E02)."""

from __future__ import annotations

import pytest

from voiage.innovate_binding import (
    evaluate_innovate_decision_binding,
    load_innovate_decision_binding,
)


def _valid_binding() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "perspective": "healthcare_payer",
        "population": 10000.0,
        "horizon": 10.0,
        "discount_rate": 0.035,
        "units": "net_benefit_usd",
        "fit_status": "valid",
        "strategies": [
            {
                "id": "status_quo",
                "draws": [
                    {"draw_id": 0, "net_benefit": 10.0},
                    {"draw_id": 1, "net_benefit": 20.0},
                ],
            },
            {
                "id": "new_therapy",
                "draws": [
                    {"draw_id": 0, "net_benefit": 15.0},
                    {"draw_id": 1, "net_benefit": 12.0},
                ],
            },
        ],
    }


def test_innovate_decision_binding_computes_exact_evpi() -> None:
    binding = _valid_binding()
    loaded = load_innovate_decision_binding(binding)
    assert loaded.strategy_ids == ["status_quo", "new_therapy"]
    assert loaded.draw_ids == [0, 1]

    result = evaluate_innovate_decision_binding(binding)
    # E[NB_sq]=15, E[NB_new]=13.5 -> max=15; E[max(10,15), max(20,12)]=(15+20)/2=17.5 -> EVPI=2.5
    assert result["evpi"] == pytest.approx(2.5)
    assert result["n_draws"] == 2
    assert result["units"] == "net_benefit_usd"


def test_innovate_decision_binding_rejects_invalid_inputs() -> None:
    base = _valid_binding()

    with pytest.raises(ValueError, match="utilities"):
        load_innovate_decision_binding({**base, "units": "prescriptions"})

    with pytest.raises(ValueError, match="fit_status"):
        load_innovate_decision_binding({**base, "fit_status": "invalid"})

    swapped = {
        **base,
        "strategies": [
            base["strategies"][0],  # type: ignore[index]
            {
                "id": "new_therapy",
                "draws": [
                    {"draw_id": 1, "net_benefit": 12.0},
                    {"draw_id": 0, "net_benefit": 15.0},
                ],
            },
        ],
    }
    with pytest.raises(ValueError, match="align"):
        load_innovate_decision_binding(swapped)

    missing = dict(base)
    del missing["perspective"]
    with pytest.raises(ValueError, match="Missing required DecisionBinding field"):
        load_innovate_decision_binding(missing)
