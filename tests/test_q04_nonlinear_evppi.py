"""Q04 qualification tests for Rust nonlinear EVPPI and MARS artifact replay."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from voiage import _runtime
from voiage.exceptions import DimensionMismatchError, InputError


def _normal_matrix(n_rows: int, n_cols: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_rows, n_cols))


def test_nonlinear_evppi_and_mars_replay_recover_analytic_thresholds() -> None:
    x = _normal_matrix(2_400, 1, seed=20_261_108)
    nb = np.column_stack([np.zeros(x.shape[0]), x[:, 0] ** 2 - 1.0])
    analytic_quad = math.sqrt(2.0 / math.pi) * math.exp(-0.5)

    linear = _runtime.compute_evppi(nb.tolist(), x.tolist())
    assert linear < 0.02

    quad = _runtime.compute_evppi_nonlinear(
        nb.tolist(),
        x.tolist(),
        basis="quadratic_interaction",
        fold_count=5,
        ridge_lambda=1e-8,
    )
    assert quad["estimator"] == "quadratic_interaction"
    assert quad["contract_version"] == 1
    assert quad["applicability_warnings"] == []
    assert quad["evppi_held_out"] == pytest.approx(analytic_quad, abs=0.06)
    assert quad["held_out_realized_gain"] == pytest.approx(analytic_quad, abs=0.06)
    assert quad["held_out_r2_by_strategy"][1] > 0.99
    assert quad["held_out_decision_loss_vs_perfect"] < 0.01

    spline = _runtime.compute_evppi_nonlinear(
        nb.tolist(),
        x.tolist(),
        basis="piecewise_linear_spline",
        knots_per_parameter=8,
        fold_count=5,
        ridge_lambda=1e-4,
    )
    assert spline["estimator"] == "piecewise_linear_spline"
    assert spline["evppi_held_out"] == pytest.approx(analytic_quad, abs=0.08)
    assert spline["held_out_r2_by_strategy"][1] > 0.90

    # Pure multiplicative interaction: NB_1 = x_1 * x_2 -> 1/pi, linear OLS ~ 0.
    x2 = _normal_matrix(2_000, 2, seed=99_001)
    nb2 = np.column_stack([np.zeros(x2.shape[0]), x2[:, 0] * x2[:, 1]])
    inter = _runtime.compute_evppi_nonlinear(
        nb2.tolist(),
        x2.tolist(),
        basis="quadratic_interaction",
        fold_count=5,
        ridge_lambda=1e-6,
    )
    assert inter["linear_evppi_reference"] < 0.06
    assert inter["evppi_held_out"] > inter["linear_evppi_reference"] + 0.20
    assert inter["evppi_held_out"] == pytest.approx(1.0 / math.pi, abs=0.05)

    # Exact producer-owned MARS artifact replay (NB_1 = -0.8 + 2*(x-0.5)_+ + 1.5*(-0.5-x)_+).
    xs = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.5, 2.5]
    mars_nb = [
        [0.0, -0.8 + 2.0 * max(0.0, val - 0.5) + 1.5 * max(0.0, -0.5 - val)]
        for val in xs
    ]
    mars_artifact = {
        "schema_version": "voiage.mars-artifact.v1",
        "producer": "external_mars_trainer",
        "strategies": [
            {"intercept": 0.0, "terms": []},
            {
                "intercept": -0.8,
                "terms": [
                    {
                        "coefficient": 2.0,
                        "factors": [
                            {"parameter_index": 0, "knot": 0.5, "direction": 1}
                        ],
                    },
                    {
                        "coefficient": 1.5,
                        "factors": [
                            {"parameter_index": 0, "knot": -0.5, "direction": -1}
                        ],
                    },
                ],
            },
        ],
    }
    replay = _runtime.compute_evppi_mars_artifact(
        mars_nb, [[val] for val in xs], json.dumps(mars_artifact)
    )
    assert replay["estimator"] == "mars_artifact_replay"
    assert replay["contract_version"] == 1
    assert replay["rmse_by_strategy"][1] < 1e-12
    assert replay["r2_by_strategy"][1] == pytest.approx(1.0, abs=1e-12)
    assert replay["decision_loss_vs_perfect"] < 1e-12


def test_nonlinear_evppi_and_mars_reject_invalid_inputs() -> None:
    nb = [[0.0, 1.0], [1.0, 0.0], [0.5, 0.5], [0.2, 0.8]]
    params = [[0.1], [0.2], [0.3], [0.4]]

    with pytest.raises(ValueError, match="unsupported nonlinear EVPPI basis"):
        _runtime.compute_evppi_nonlinear(nb, params, basis="unknown_basis")

    with pytest.raises(InputError, match="fold_count"):
        _runtime.compute_evppi_nonlinear(nb, params, fold_count=1)

    with pytest.raises(DimensionMismatchError, match="sample counts"):
        _runtime.compute_evppi_nonlinear(nb, params[:-1])

    with pytest.raises(ValueError, match="invalid MARS artifact JSON"):
        _runtime.compute_evppi_mars_artifact(nb, params, "{corrupt")

    bad_dir = {
        "schema_version": "voiage.mars-artifact.v1",
        "producer": "external_mars_trainer",
        "strategies": [
            {"intercept": 0.0, "terms": []},
            {
                "intercept": 0.0,
                "terms": [
                    {
                        "coefficient": 1.0,
                        "factors": [
                            {"parameter_index": 0, "knot": 0.0, "direction": 0}
                        ],
                    }
                ],
            },
        ],
    }
    with pytest.raises(InputError, match="direction"):
        _runtime.compute_evppi_mars_artifact(nb, params, json.dumps(bad_dir))

    with pytest.raises(DimensionMismatchError, match="sample counts"):
        _runtime.compute_evppi_mars_artifact(nb, params[:-1], json.dumps(bad_dir))

    # Surrogate-consistent baseline: constant parameters or constant MARS artifact
    # must yield zero EVPPI rather than counting calibration offset as information.
    const_nb = [[0.0, 0.0], [0.0, 3.0], [0.0, 0.0]]
    const_p = [[1.0], [1.0], [1.0]]
    const_res = _runtime.compute_evppi_nonlinear(
        const_nb, const_p, fold_count=2, ridge_lambda=0.0
    )
    assert const_res["evppi_held_out"] == pytest.approx(0.0, abs=1e-12)
    assert const_res["held_out_realized_gain"] <= 0.0
    const_mars = {
        "schema_version": "voiage.mars-artifact.v1",
        "producer": "external_constant_surrogate",
        "strategies": [
            {"intercept": 0.0, "terms": []},
            {"intercept": 2.0, "terms": []},
        ],
    }
    const_replay = _runtime.compute_evppi_mars_artifact(
        const_nb, const_p, json.dumps(const_mars)
    )
    assert const_replay["evppi"] == pytest.approx(0.0, abs=1e-12)
