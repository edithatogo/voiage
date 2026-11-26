//! Q04 nonlinear EVPPI and MARS artifact replay integration tests.

use std::f64::consts::{E, PI};

use voiage_domain::SampleMatrix;
use voiage_numerics::{
    evppi, evppi_from_mars_artifact, evppi_nonlinear, EvppiBasis, EvppiNonlinearConfig,
    MarsArtifact, MarsHingeFactor, MarsStrategyModel, MarsTerm,
};

struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn next_unit(&mut self) -> f64 {
        #[allow(clippy::cast_precision_loss)]
        let mantissa = (self.next_u64() >> 11) as f64;
        let scale = f64::from(1_u32 << 26) * f64::from(1_u32 << 27);
        (mantissa + 0.5) / scale
    }

    fn next_normal_pair(&mut self) -> (f64, f64) {
        let u1 = self.next_unit();
        let u2 = self.next_unit();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * PI * u2;
        (r * theta.cos(), r * theta.sin())
    }
}

fn matrix(rows: Vec<Vec<f64>>) -> SampleMatrix {
    SampleMatrix::try_from(rows).expect("valid sample matrix")
}

#[test]
fn quadratic_threshold_beats_linear_ols_and_tracks_analytic_reference() {
    let mut rng = SplitMix64(20_261_108);
    let n = 2_400;
    let mut nb_rows = Vec::with_capacity(n);
    let mut x_rows = Vec::with_capacity(n);
    for _ in 0..(n / 2) {
        let (z1, z2) = rng.next_normal_pair();
        for x in [z1, z2] {
            nb_rows.push(vec![0.0, x * x - 1.0]);
            x_rows.push(vec![x]);
        }
    }
    let nb = matrix(nb_rows);
    let params = matrix(x_rows);
    let analytic = (2.0 / PI).sqrt() / E.sqrt();

    let linear = evppi(&nb, &params).expect("linear evppi succeeds");
    assert!(
        linear < 0.02,
        "linear OLS should miss symmetric quadratic payoff: {linear}"
    );

    let quad = evppi_nonlinear(
        &nb,
        &params,
        &EvppiNonlinearConfig {
            basis: EvppiBasis::QuadraticInteraction,
            fold_count: 5,
            ridge_lambda: 1.0e-8,
        },
    )
    .expect("quadratic nonlinear EVPPI succeeds");
    assert_eq!(quad.estimator, "quadratic_interaction");
    assert_eq!(quad.basis_width, 3);
    assert_eq!(quad.applicability_warnings, Vec::<String>::new());
    assert!((quad.evppi_held_out - analytic).abs() < 0.05);
    assert!((quad.held_out_realized_gain - analytic).abs() < 0.05);
    assert!(quad.held_out_r2_by_strategy[1] > 0.99);
    assert!(quad.held_out_decision_loss_vs_perfect < 0.01);

    let spline = evppi_nonlinear(
        &nb,
        &params,
        &EvppiNonlinearConfig {
            basis: EvppiBasis::PiecewiseLinearSpline {
                knots_per_parameter: 8,
            },
            fold_count: 5,
            ridge_lambda: 1.0e-4,
        },
    )
    .expect("spline nonlinear EVPPI succeeds");
    assert_eq!(spline.estimator, "piecewise_linear_spline");
    assert!((spline.evppi_held_out - analytic).abs() < 0.08);
    assert!(spline.held_out_r2_by_strategy[1] > 0.90);

    let repeat = evppi_nonlinear(
        &nb,
        &params,
        &EvppiNonlinearConfig {
            basis: EvppiBasis::QuadraticInteraction,
            fold_count: 5,
            ridge_lambda: 1.0e-8,
        },
    )
    .expect("repeat succeeds");
    assert_eq!(quad, repeat);
}

#[test]
fn interaction_and_correlated_noise_preserve_invariants_and_decision_loss() {
    let mut rng = SplitMix64(99_001);
    let n = 2_000;
    let mut nb_rows = Vec::with_capacity(n);
    let mut x_rows = Vec::with_capacity(n);
    for _ in 0..n {
        let (x1, x2) = rng.next_normal_pair();
        nb_rows.push(vec![0.0, x1 * x2]);
        x_rows.push(vec![x1, x2]);
    }
    let nb = matrix(nb_rows);
    let params = matrix(x_rows);
    let analytic_interaction = 1.0 / PI;

    let res = evppi_nonlinear(
        &nb,
        &params,
        &EvppiNonlinearConfig {
            basis: EvppiBasis::QuadraticInteraction,
            fold_count: 5,
            ridge_lambda: 1.0e-6,
        },
    )
    .expect("interaction EVPPI succeeds");
    assert!(res.linear_evppi_reference < 0.06);
    assert!(res.evppi_held_out > res.linear_evppi_reference + 0.20);
    assert!((res.evppi_held_out - analytic_interaction).abs() < 0.05);
    assert!(res.held_out_r2_by_strategy[1] > 0.99);

    let mut corr_nb_rows = Vec::with_capacity(n);
    let mut corr_x_rows = Vec::with_capacity(n);
    for _ in 0..n {
        let (z1, z2) = rng.next_normal_pair();
        let (nuisance, _) = rng.next_normal_pair();
        let x1 = z1;
        let x2 = 0.8 * z1 + 0.6 * z2;
        corr_nb_rows.push(vec![0.0, x1 * x2 - 0.8 + nuisance]);
        corr_x_rows.push(vec![x1, x2]);
    }
    let corr_nb_m = matrix(corr_nb_rows.clone());
    let corr_x_m = matrix(corr_x_rows);
    let corr_res = evppi_nonlinear(
        &corr_nb_m,
        &corr_x_m,
        &EvppiNonlinearConfig {
            basis: EvppiBasis::QuadraticInteraction,
            fold_count: 5,
            ridge_lambda: 1.0e-4,
        },
    )
    .expect("correlated EVPPI succeeds");
    let evpi = corr_res.expected_perfect_information - corr_res.expected_current_value;
    assert!(corr_res.held_out_realized_gain >= 0.0);
    assert!(corr_res.evppi_held_out <= evpi + 1.0e-9);
    assert!(corr_res.held_out_decision_loss_vs_perfect > 0.05);

    let shifted = matrix(
        corr_nb_rows
            .iter()
            .map(|r| vec![r[0] + 25.0, r[1] + 25.0])
            .collect(),
    );
    let shift_res = evppi_nonlinear(
        &shifted,
        &corr_x_m,
        &EvppiNonlinearConfig {
            basis: EvppiBasis::QuadraticInteraction,
            fold_count: 5,
            ridge_lambda: 1.0e-4,
        },
    )
    .expect("shifted succeeds");
    assert!((shift_res.evppi_held_out - corr_res.evppi_held_out).abs() < 1.0e-9);
    assert!((shift_res.held_out_realized_gain - corr_res.held_out_realized_gain).abs() < 1.0e-9);

    let scaled = matrix(
        corr_nb_rows
            .iter()
            .map(|r| vec![r[0] * 3.0, r[1] * 3.0])
            .collect(),
    );
    let scale_res = evppi_nonlinear(
        &scaled,
        &corr_x_m,
        &EvppiNonlinearConfig {
            basis: EvppiBasis::QuadraticInteraction,
            fold_count: 5,
            ridge_lambda: 1.0e-4,
        },
    )
    .expect("scaled succeeds");
    assert!((scale_res.evppi_held_out - 3.0 * corr_res.evppi_held_out).abs() < 1.0e-9);
    assert!(
        (scale_res.held_out_r2_by_strategy[1] - corr_res.held_out_r2_by_strategy[1]).abs() < 1.0e-9
    );
}

#[test]
fn applicability_warnings_and_nonlinear_rejections_fail_closed() {
    let nb_small = matrix(vec![
        vec![0.0, 2.0],
        vec![0.0, -3.0],
        vec![0.0, 1.5],
        vec![0.0, -0.5],
        vec![0.0, 4.0],
        vec![0.0, -4.0],
        vec![0.0, 0.2],
        vec![0.0, -1.2],
        vec![0.0, 3.0],
        vec![0.0, -2.0],
    ]);
    let p_const = matrix((1..=10).map(|i| vec![f64::from(i), 5.0]).collect());
    let diag = evppi_nonlinear(
        &nb_small,
        &p_const,
        &EvppiNonlinearConfig {
            basis: EvppiBasis::PiecewiseLinearSpline {
                knots_per_parameter: 4,
            },
            fold_count: 5,
            ridge_lambda: 1.0e-3,
        },
    )
    .expect("diagnostic warnings evaluate cleanly");
    assert!(diag
        .applicability_warnings
        .iter()
        .any(|w| w == "low_sample_to_basis_ratio"));
    assert!(diag
        .applicability_warnings
        .iter()
        .any(|w| w == "constant_parameter_column"));

    assert!(evppi_nonlinear(
        &nb_small,
        &p_const,
        &EvppiNonlinearConfig {
            basis: EvppiBasis::QuadraticInteraction,
            fold_count: 1,
            ridge_lambda: 0.0,
        },
    )
    .is_err());
    assert!(evppi_nonlinear(
        &nb_small,
        &p_const,
        &EvppiNonlinearConfig {
            basis: EvppiBasis::PiecewiseLinearSpline {
                knots_per_parameter: 0,
            },
            fold_count: 2,
            ridge_lambda: 0.0,
        },
    )
    .is_err());
}

#[test]
fn mars_artifact_replay_and_invalid_artifacts() {
    let xs = [-2.0_f64, -1.0, -0.5, 0.0, 0.5, 1.5, 2.5];
    let nb_mars = matrix(
        xs.iter()
            .map(|&x| {
                vec![
                    0.0,
                    -0.8 + 2.0 * (x - 0.5).max(0.0) + 1.5 * (-0.5 - x).max(0.0),
                ]
            })
            .collect(),
    );
    let p_mars = matrix(xs.iter().map(|&x| vec![x]).collect());
    let artifact = MarsArtifact {
        schema_version: "voiage.mars-artifact.v1".to_string(),
        producer: "earth_r_4_4".to_string(),
        strategies: vec![
            MarsStrategyModel {
                intercept: 0.0,
                terms: vec![],
            },
            MarsStrategyModel {
                intercept: -0.8,
                terms: vec![
                    MarsTerm {
                        coefficient: 2.0,
                        factors: vec![MarsHingeFactor {
                            parameter_index: 0,
                            knot: 0.5,
                            direction: 1,
                        }],
                    },
                    MarsTerm {
                        coefficient: 1.5,
                        factors: vec![MarsHingeFactor {
                            parameter_index: 0,
                            knot: -0.5,
                            direction: -1,
                        }],
                    },
                ],
            },
        ],
    };
    let replay = evppi_from_mars_artifact(&nb_mars, &p_mars, &artifact).expect("mars replay");
    assert_eq!(replay.estimator, "mars_artifact_replay");
    assert!(replay.rmse_by_strategy[1] < 1.0e-12);
    assert!((replay.r2_by_strategy[1] - 1.0).abs() < 1.0e-12);
    assert!(replay.decision_loss_vs_perfect < 1.0e-12);

    let mut bad_art = artifact;
    bad_art.strategies[1].terms[0].factors[0].direction = 0;
    assert!(evppi_from_mars_artifact(&nb_mars, &p_mars, &bad_art).is_err());
}

#[test]
fn surrogate_consistent_baseline_prevents_calibration_inflation() {
    let nb = matrix(vec![vec![0.0, 0.0], vec![0.0, 3.0], vec![0.0, 0.0]]);
    let params = matrix(vec![vec![1.0], vec![1.0], vec![1.0]]);
    let res = evppi_nonlinear(
        &nb,
        &params,
        &EvppiNonlinearConfig {
            basis: EvppiBasis::QuadraticInteraction,
            fold_count: 2,
            ridge_lambda: 0.0,
        },
    )
    .expect("constant parameter case evaluates cleanly");
    assert!(res.evppi_held_out.abs() < 1.0e-12);
    assert!(res.held_out_realized_gain <= 0.0);

    let const_mars = MarsArtifact {
        schema_version: "voiage.mars-artifact.v1".to_string(),
        producer: "external_constant_surrogate".to_string(),
        strategies: vec![
            MarsStrategyModel {
                intercept: 0.0,
                terms: vec![],
            },
            MarsStrategyModel {
                intercept: 2.0,
                terms: vec![],
            },
        ],
    };
    let replay = evppi_from_mars_artifact(&nb, &params, &const_mars).expect("constant MARS");
    assert!(replay.evppi.abs() < 1.0e-12);
    assert!(replay.realized_decision_gain.abs() < 1.0e-12);
}
