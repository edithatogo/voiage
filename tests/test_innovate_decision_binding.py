"""Tests for Voiage-side Innovate DecisionBinding adapter (E02)."""

from __future__ import annotations

from typing import Any

import pytest

from voiage.innovate_binding import (
    evaluate_innovate_decision_binding,
    load_innovate_decision_binding,
)


def _valid_binding() -> dict[str, Any]:
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


@pytest.mark.parametrize("version", ["2.0", "", None, 1.0])
def test_innovate_decision_binding_rejects_unsupported_schema_versions(
    version: object,
) -> None:
    binding = {**_valid_binding(), "schema_version": version}

    with pytest.raises(ValueError, match="schema version"):
        load_innovate_decision_binding(binding)


@pytest.mark.parametrize("perspective", [None, "", "   ", 5])
def test_innovate_decision_binding_requires_a_labelled_perspective(
    perspective: object,
) -> None:
    binding = {**_valid_binding(), "perspective": perspective}

    with pytest.raises(ValueError, match="perspective"):
        load_innovate_decision_binding(binding)


@pytest.mark.parametrize("strategy_id", [None, "", "   ", 5])
def test_innovate_decision_binding_requires_string_strategy_ids(
    strategy_id: object,
) -> None:
    binding = _valid_binding()
    binding["strategies"][0]["id"] = strategy_id

    with pytest.raises(ValueError, match="strategy id"):
        load_innovate_decision_binding(binding)


@pytest.mark.parametrize("units", [["net_benefit_usd"], {"net_benefit": 1}])
def test_innovate_decision_binding_requires_string_units(units: object) -> None:
    binding = {**_valid_binding(), "units": units}

    with pytest.raises(ValueError, match="units"):
        load_innovate_decision_binding(binding)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("population", 0),
        ("population", -1),
        ("population", float("nan")),
        ("population", float("inf")),
        ("horizon", 0),
        ("horizon", -1),
        ("horizon", float("nan")),
        ("horizon", float("inf")),
        ("discount_rate", -0.01),
        ("discount_rate", 1.01),
        ("discount_rate", float("nan")),
        ("discount_rate", float("inf")),
    ],
)
def test_innovate_decision_binding_rejects_invalid_scaling_metadata(
    field: str,
    value: float,
) -> None:
    binding = {**_valid_binding(), field: value}

    with pytest.raises(ValueError, match=field):
        load_innovate_decision_binding(binding)


@pytest.mark.parametrize("discount_rate", [0.0, 1.0])
def test_innovate_decision_binding_accepts_scaling_boundaries(
    discount_rate: float,
) -> None:
    binding = {
        **_valid_binding(),
        "population": 0.5,
        "horizon": 0.5,
        "discount_rate": discount_rate,
    }

    loaded = load_innovate_decision_binding(binding)

    assert loaded.population == 0.5
    assert loaded.horizon == 0.5
    assert loaded.discount_rate == discount_rate


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_innovate_decision_binding_rejects_nonfinite_net_benefits(value: float) -> None:
    binding = _valid_binding()
    binding["strategies"][0]["draws"][0]["net_benefit"] = value

    with pytest.raises(ValueError, match="finite"):
        load_innovate_decision_binding(binding)


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


def test_innovate_decision_binding_matches_frozen_diffusion_fixture() -> None:
    # Hand-authored Innovate fixture, manifest SHA-256 b61feb8ed857dc1ab5f9b6d18d75091fe5b458c345cdace66338d0ae6afcbc5e.
    # At time 3 the synthetic utility equals cumulative adoption, minus 0.05
    # adoption-equivalent units for accelerated access. This is not a clinical
    # or economic valuation; the independent exact EVPI is 1/600.
    binding = {
        **_valid_binding(),
        "perspective": "synthetic_fixture",
        "population": 1000,
        "horizon": 3,
        "discount_rate": 0,
        "units": "net_benefit_adoption_equivalent",
        "strategies": [
            {
                "id": "baseline_access",
                "draws": [
                    {"draw_id": draw_id, "net_benefit": value}
                    for draw_id, value in zip(
                        (1, 2, 3), (0.235, 0.258, 0.280), strict=True
                    )
                ],
            },
            {
                "id": "accelerated_access",
                "draws": [
                    {"draw_id": draw_id, "net_benefit": value - 0.05}
                    for draw_id, value in zip(
                        (1, 2, 3), (0.282, 0.306, 0.339), strict=True
                    )
                ],
            },
        ],
    }

    result = evaluate_innovate_decision_binding(binding)

    assert result["evpi"] == pytest.approx(1 / 600, abs=1e-14)
    assert result["strategy_ids"] == ["baseline_access", "accelerated_access"]
    assert result["n_draws"] == 3


def test_innovate_decision_binding_scales_evpi_for_population() -> None:
    binding = {**_valid_binding(), "population": 4, "horizon": 3, "discount_rate": 0.1}

    result = evaluate_innovate_decision_binding(binding, population_scaled=True)

    discounted_years = sum(1 / 1.1**year for year in range(1, 4))
    assert result["evpi"] == pytest.approx(2.5 * 4 * discounted_years)
    assert result["strategy_ids"] == ["status_quo", "new_therapy"]


@pytest.mark.parametrize(
    "strategies",
    [None, [], [{"id": "only", "draws": [{"draw_id": 0, "net_benefit": 1.0}]}]],
)
def test_innovate_decision_binding_requires_two_strategies(strategies: object) -> None:
    binding = {**_valid_binding(), "strategies": strategies}
    with pytest.raises(ValueError, match="At least two strategies"):
        load_innovate_decision_binding(binding)


def test_innovate_decision_binding_rejects_duplicate_strategy_ids() -> None:
    binding = _valid_binding()
    binding["strategies"][1]["id"] = binding["strategies"][0]["id"]

    with pytest.raises(ValueError, match="Duplicate strategy id.*status_quo"):
        load_innovate_decision_binding(binding)


@pytest.mark.parametrize("strategy_index", [0, 1])
def test_innovate_decision_binding_rejects_empty_draws(strategy_index: int) -> None:
    binding = _valid_binding()
    strategy = binding["strategies"][strategy_index]
    strategy["draws"] = []

    with pytest.raises(ValueError, match=f"Strategy '{strategy['id']}' has no draws"):
        load_innovate_decision_binding(binding)


def test_innovate_decision_binding_rejects_invalid_inputs() -> None:
    base = _valid_binding()

    with pytest.raises(ValueError, match="utilities"):
        load_innovate_decision_binding({**base, "units": "prescriptions"})

    with pytest.raises(ValueError, match="fit_status"):
        load_innovate_decision_binding({**base, "fit_status": "invalid"})

    swapped = {
        **base,
        "strategies": [
            base["strategies"][0],
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


@pytest.mark.parametrize(
    ("first_ids", "second_ids"),
    [([0, 0], [0, 0]), ([0, 1], [0, 0]), ([0, 0], [0, 1])],
)
def test_innovate_decision_binding_rejects_duplicate_draw_ids(
    first_ids: list[int], second_ids: list[int]
) -> None:
    binding = {
        **_valid_binding(),
        "strategies": [
            {
                "id": "status_quo",
                "draws": [
                    {"draw_id": draw_id, "net_benefit": value}
                    for draw_id, value in zip(first_ids, (0.0, 10.0), strict=True)
                ],
            },
            {
                "id": "new_therapy",
                "draws": [
                    {"draw_id": draw_id, "net_benefit": value}
                    for draw_id, value in zip(second_ids, (10.0, 0.0), strict=True)
                ],
            },
        ],
    }
    with pytest.raises(ValueError, match="Duplicate draw ID"):
        load_innovate_decision_binding(binding)


@pytest.mark.parametrize("draw_id", [[], {}, float("nan")])
def test_innovate_decision_binding_rejects_invalid_draw_ids(draw_id: object) -> None:
    binding = {
        **_valid_binding(),
        "strategies": [
            {"id": "a", "draws": [{"draw_id": draw_id, "net_benefit": 1.0}]},
            {"id": "b", "draws": [{"draw_id": draw_id, "net_benefit": 2.0}]},
        ],
    }
    with pytest.raises(ValueError, match="draw ID"):
        load_innovate_decision_binding(binding)
