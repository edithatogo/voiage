//! Q05 coherent EVSI qualification tests.

use voiage_domain::{EvsiDiagnostics, SampleMatrix, SampleSizeConvention};
use voiage_numerics::{
    beta_binomial_two_arm_evsi, compare_evsi_mc_vs_qmc, evsi_importance_sampling,
    evsi_importance_sampling_raw, evsi_moment_based, evsi_multilevel_normal_normal,
    evsi_qmc_normal_normal, evsi_regression, exponential_time_to_event_evsi, gamma_poisson_evsi,
    normal_normal_two_arm_evsi,
};

#[test]
fn q05_sample_size_convention_distinguishes_per_arm_and_total() {
    let per_arm = SampleSizeConvention::PerArm { per_arm: 25 };
    let total_eq = SampleSizeConvention::TotalEqualTwoArm { total: 50 };
    assert_eq!(per_arm.total_sample_size().unwrap(), 50);
    assert_eq!(total_eq.total_sample_size().unwrap(), 50);
    assert_eq!(per_arm.per_arm_sizes().unwrap(), vec![25, 25]);
    assert_eq!(total_eq.per_arm_sizes().unwrap(), vec![25, 25]);

    assert!(SampleSizeConvention::TotalEqualTwoArm { total: 51 }
        .validate()
        .is_err());
    assert!(SampleSizeConvention::PerArm { per_arm: 0 }
        .validate()
        .is_err());
}

#[test]
fn q05_diagnostics_reject_out_of_range_values() {
    let ok = EvsiDiagnostics {
        ess: Some(120.0),
        khat: Some(0.35),
        decision_error: Some(0.05),
        monte_carlo_se: Some(0.01),
        bias_estimate: Some(-0.002),
        rmse_estimate: Some(0.011),
        coverage: Some(0.95),
    };
    assert!(ok.validate().is_ok());

    let bad_err = EvsiDiagnostics {
        decision_error: Some(1.2),
        ..ok
    };
    assert!(bad_err.validate().is_err());
}

#[test]
fn q05_conjugate_study_models_are_nonnegative_and_monotone_in_sample_size() {
    let small = SampleSizeConvention::PerArm { per_arm: 10 };
    let large = SampleSizeConvention::PerArm { per_arm: 40 };

    let bb_small =
        beta_binomial_two_arm_evsi(2.0, 8.0, 2.5, 7.5, &small, 1000.0, -50.0).unwrap();
    let bb_large =
        beta_binomial_two_arm_evsi(2.0, 8.0, 2.5, 7.5, &large, 1000.0, -50.0).unwrap();
    assert!(bb_small > 0.0);
    assert!(bb_large > bb_small);

    let gp_small = gamma_poisson_evsi(4.0, 10.0, 1.0, &small, 500.0, -200.0).unwrap();
    let gp_large = gamma_poisson_evsi(4.0, 10.0, 1.0, &large, 500.0, -200.0).unwrap();
    assert!(gp_small > 0.0);
    assert!(gp_large > gp_small);

    let tte_small = exponential_time_to_event_evsi(3.0, 6.0, 10.0, 400.0, -200.0).unwrap();
    let tte_large = exponential_time_to_event_evsi(3.0, 6.0, 40.0, 400.0, -200.0).unwrap();
    assert!(tte_small > 0.0);
    assert!(tte_large > tte_small);
}

#[test]
fn q05_importance_sampling_passes_healthy_weights_and_fails_closed_on_degeneracy() {
    // 10 prior samples, 2 strategies
    let nb = SampleMatrix::try_from(vec![
        vec![0.0, 1.0],
        vec![0.0, 2.0],
        vec![0.0, 3.0],
        vec![0.0, -1.0],
        vec![0.0, -2.0],
        vec![0.0, 1.5],
        vec![0.0, 0.5],
        vec![0.0, -0.5],
        vec![0.0, 2.5],
        vec![0.0, -1.5],
    ])
    .unwrap();

    // Well-behaved weights across 2 outer datasets
    let healthy_lw = SampleMatrix::try_from(vec![
        vec![0.1, 0.2, 0.3, -0.1, -0.2, 0.15, 0.05, -0.05, 0.25, -0.15],
        vec![-0.1, -0.2, -0.3, 0.1, 0.2, -0.15, -0.05, 0.05, -0.25, 0.15],
    ])
    .unwrap();
    let res = evsi_importance_sampling(&nb, &healthy_lw, 5.0, 0.7).unwrap();
    assert!(res.min_ess >= 5.0);
    assert!(res.max_khat <= 0.7);
    assert!(res.evsi >= 0.0);

    // Degenerate weights: one weight dominates (ESS ~ 1.0)
    let degenerate_lw = SampleMatrix::try_from(vec![vec![
        20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0,
    ]])
    .unwrap();
    let raw = evsi_importance_sampling_raw(&nb, &degenerate_lw).unwrap();
    assert!(raw.min_ess < 1.1);
    assert!(evsi_importance_sampling(&nb, &degenerate_lw, 5.0, 0.7).is_err());
}

#[test]
fn q05_qmc_and_multilevel_beat_plain_mc_at_equal_budget() {
    let mut mc_sq_err = 0.0;
    let mut qmc_abs_err = 0.0;
    for seed in 1_u64..=16 {
        let cmp = compare_evsi_mc_vs_qmc(0.1, 0.5, 1.0, 100, 1000.0, -100.0, 1024, seed).unwrap();
        mc_sq_err += cmp.mc_abs_error * cmp.mc_abs_error;
        qmc_abs_err = cmp.qmc_abs_error;
    }
    let mc_rmse = (mc_sq_err / 16.0).sqrt();
    assert!(qmc_abs_err < mc_rmse);
    assert!(qmc_abs_err < 2.0);

    let ml = evsi_multilevel_normal_normal(0.1, 0.5, 1.0, 100, 1000.0, -100.0, 32, 6).unwrap();
    let exact = normal_normal_two_arm_evsi(0.1, 0.5, 1.0, 100, 1000.0, -100.0).unwrap();
    assert!((ml.evsi - exact).abs() < 2.0);
}

#[test]
fn q05_regression_and_moment_estimators_stay_finite_and_ordered() {
    let nb = SampleMatrix::try_from(vec![
        vec![0.0, 1.0],
        vec![0.0, 2.0],
        vec![0.0, -1.0],
        vec![0.0, 0.5],
        vec![0.0, -0.5],
        vec![0.0, 1.5],
    ])
    .unwrap();
    let params = SampleMatrix::try_from(vec![
        vec![0.1],
        vec![0.2],
        vec![-0.1],
        vec![0.05],
        vec![-0.05],
        vec![0.15],
    ])
    .unwrap();
    let targets = SampleMatrix::try_from(vec![
        vec![1.0],
        vec![2.0],
        vec![0.0],
        vec![0.5],
        vec![0.0],
        vec![1.5],
    ])
    .unwrap();

    let mom = evsi_moment_based(&nb, &params, 20).unwrap();
    let reg = evsi_regression(&targets, &params, &params).unwrap();
    assert!(mom.evsi >= 0.0 && mom.evsi <= mom.expected_perfect_information + 1e-9);
    assert!(reg.expected_sample_value.is_finite());

    let qmc = evsi_qmc_normal_normal(0.0, 1.0, 2.0, 50, 100.0, 0.0, 128).unwrap();
    assert!(qmc.evsi > 0.0);
}
