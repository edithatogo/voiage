"""Tests for Q05 coherent EVSI contracts, non-normal study models, IS, and QMC."""

from __future__ import annotations

import numpy as np
import pytest

from voiage.contracts.study_design import (
    EvsiDiagnosticsV1,
    EvsiStudyResultV1,
    PosteriorSampleBoundaryV1,
    SampleSizeSpecV1,
)
from voiage.exceptions import InputError
from voiage.methods.sample_information import (
    beta_binomial_two_arm_evsi,
    evsi_importance_sampling,
    evsi_qmc_normal_normal,
    exponential_time_to_event_evsi,
    gamma_poisson_evsi,
    normal_normal_two_arm_evsi,
)


def test_q05_contracts_enforce_sample_size_and_f02_diagnostics() -> None:
    per_arm = SampleSizeSpecV1(convention="per_arm", per_arm=25)
    total = SampleSizeSpecV1(convention="total_equal_two_arm", total=50)
    assert per_arm.resolved_total() == 50
    assert total.resolved_total() == 50

    with pytest.raises(ValueError):
        SampleSizeSpecV1(convention="per_arm", total=50)
    with pytest.raises(ValueError):
        SampleSizeSpecV1(convention="total_equal_two_arm", per_arm=25)
    with pytest.raises(ValueError):
        SampleSizeSpecV1(convention="total_equal_two_arm", total=51)
    with pytest.raises(ValueError):
        SampleSizeSpecV1(convention="multi_arm", per_arm_sizes=(10,))

    multi = SampleSizeSpecV1(convention="multi_arm", per_arm_sizes=(10, 20, 30))
    assert multi.resolved_total() == 60

    res = EvsiStudyResultV1(
        estimator="importance_sampling",
        study_model="two_arm_binary",
        sample_size=per_arm,
        evsi=12.5,
        expected_current_value=100.0,
        expected_sample_value=112.5,
        diagnostics=EvsiDiagnosticsV1(ess=85.0, khat=0.35, decision_error=0.04),
    )
    assert res.diagnostics.khat == 0.35

    ok_post = PosteriorSampleBoundaryV1(
        framework="pymc",
        parameter_names=("alpha", "beta"),
        draw_count=1000,
        chain_count=4,
        min_bulk_ess=400.0,
        max_rhat=1.01,
        divergent_transitions=0,
    )
    assert ok_post.chain_count == 4

    with pytest.raises(ValueError):
        PosteriorSampleBoundaryV1(
            framework="stan",
            parameter_names=("alpha", "alpha"),
            draw_count=1000,
            chain_count=4,
            min_bulk_ess=400.0,
            max_rhat=1.01,
            divergent_transitions=0,
        )
    with pytest.raises(ValueError):
        PosteriorSampleBoundaryV1(
            framework="stan",
            parameter_names=("alpha",),
            draw_count=1000,
            chain_count=4,
            min_bulk_ess=400.0,
            max_rhat=1.25,
            divergent_transitions=0,
        )
    with pytest.raises(ValueError):
        PosteriorSampleBoundaryV1(
            framework="stan",
            parameter_names=("alpha",),
            draw_count=1000,
            chain_count=4,
            min_bulk_ess=400.0,
            max_rhat=1.01,
            divergent_transitions=2,
        )


def test_q05_native_study_models_and_qmc_bindings() -> None:
    bb_small = beta_binomial_two_arm_evsi(
        alpha_0=2.0,
        beta_0=8.0,
        alpha_1=2.5,
        beta_1=7.5,
        per_arm_sample_size=10,
        net_benefit_slope=1000.0,
        net_benefit_intercept=-50.0,
    )
    bb_large = beta_binomial_two_arm_evsi(
        alpha_0=2.0,
        beta_0=8.0,
        alpha_1=2.5,
        beta_1=7.5,
        per_arm_sample_size=40,
        net_benefit_slope=1000.0,
        net_benefit_intercept=-50.0,
    )
    assert 0.0 < bb_small < bb_large

    gp = gamma_poisson_evsi(
        prior_shape=4.0,
        prior_rate=10.0,
        exposure_per_unit=1.0,
        total_sample_size=20,
        net_benefit_slope=500.0,
        net_benefit_intercept=-200.0,
    )
    assert gp > 0.0

    tte = exponential_time_to_event_evsi(
        prior_shape=3.0,
        prior_rate=6.0,
        expected_events=20.0,
        net_benefit_slope=400.0,
        net_benefit_intercept=-200.0,
    )
    assert tte > 0.0

    exact = normal_normal_two_arm_evsi(
        prior_mean=0.1,
        prior_standard_deviation=0.5,
        outcome_standard_deviation=1.0,
        total_sample_size=100,
        net_benefit_slope=1000.0,
        net_benefit_intercept=-100.0,
    )
    qmc = evsi_qmc_normal_normal(
        prior_mean=0.1,
        prior_standard_deviation=0.5,
        outcome_standard_deviation=1.0,
        total_sample_size=100,
        net_benefit_slope=1000.0,
        net_benefit_intercept=-100.0,
        num_points=1024,
    )
    assert abs(float(qmc["evsi"]) - exact) < 2.0


def test_q05_importance_sampling_binding_and_fail_closed_degeneracy() -> None:
    nb = np.array(
        [[0.0, float(k)] for k in [1, 2, 3, -1, -2, 1.5, 0.5, -0.5, 2.5, -1.5]]
    )
    healthy_lw = np.array([[0.1, 0.2, 0.3, -0.1, -0.2, 0.15, 0.05, -0.05, 0.25, -0.15]])
    res = evsi_importance_sampling(
        nb, healthy_lw, min_ess_threshold=5.0, max_khat_threshold=0.7
    )
    assert float(res["min_ess"]) >= 5.0
    assert float(res["evsi"]) >= 0.0

    bad_lw = np.array([[20.0] + [-20.0] * 9])
    with pytest.raises(InputError):
        evsi_importance_sampling(
            nb, bad_lw, min_ess_threshold=5.0, max_khat_threshold=0.7
        )
    with pytest.raises(InputError):
        evsi_importance_sampling(np.array([1.0, 2.0]), bad_lw)
    with pytest.raises(InputError):
        beta_binomial_two_arm_evsi(
            alpha_0=2.0,
            beta_0=8.0,
            alpha_1=2.5,
            beta_1=7.5,
            per_arm_sample_size=0,
            net_benefit_slope=1000.0,
            net_benefit_intercept=-50.0,
        )
    with pytest.raises(InputError):
        gamma_poisson_evsi(
            prior_shape=4.0,
            prior_rate=10.0,
            exposure_per_unit=1.0,
            total_sample_size=21,
            net_benefit_slope=500.0,
            net_benefit_intercept=-200.0,
        )
    with pytest.raises(InputError):
        evsi_qmc_normal_normal(
            prior_mean=0.1,
            prior_standard_deviation=0.5,
            outcome_standard_deviation=1.0,
            total_sample_size=99,
            net_benefit_slope=1000.0,
            net_benefit_intercept=-100.0,
            num_points=1024,
        )
    with pytest.raises(InputError):
        evsi_qmc_normal_normal(
            prior_mean=0.1,
            prior_standard_deviation=0.5,
            outcome_standard_deviation=1.0,
            total_sample_size=100,
            net_benefit_slope=1000.0,
            net_benefit_intercept=-100.0,
            num_points=0,
        )
