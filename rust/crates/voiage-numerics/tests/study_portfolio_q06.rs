//! Q06 study-design scheduling, discounting, and portfolio-optimization tests.

use voiage_numerics::{
    discounted_population_enbs, enumerate_study_portfolio, equity_weighted_study_enbs,
    sequential_adaptive_stopping_regret, PortfolioInteraction, PortfolioStudySpec,
    StudyScheduleConfig,
};

#[test]
fn q06_discounted_population_enbs_handles_delay_and_horizon() {
    let base = StudyScheduleConfig {
        discount_rate: 0.035,
        recruitment_delay_years: 1.0,
        time_horizon_years: 10.0,
        incident_population_per_year: 1000.0,
    };
    let val = discounted_population_enbs(2.0, 5000.0, &base).unwrap();
    assert!(val > 0.0);

    let expired = StudyScheduleConfig {
        recruitment_delay_years: 10.0,
        ..base
    };
    let val_expired = discounted_population_enbs(2.0, 5000.0, &expired).unwrap();
    assert!((val_expired - (-5000.0)).abs() < 1e-12);
}

#[test]
fn q06_portfolio_enumeration_guards_overlaps_and_selects_optimal_subset() {
    let studies = vec![
        PortfolioStudySpec {
            study_id: "s1".to_string(),
            cost: 100.0,
            standalone_evsi: 220.0,
            parameter_group: "efficacy".to_string(),
        },
        PortfolioStudySpec {
            study_id: "s2".to_string(),
            cost: 100.0,
            standalone_evsi: 210.0,
            parameter_group: "efficacy".to_string(),
        },
        PortfolioStudySpec {
            study_id: "s3".to_string(),
            cost: 80.0,
            standalone_evsi: 150.0,
            parameter_group: "utility".to_string(),
        },
    ];

    // Without explicit interaction guard for overlapping "efficacy" studies, fail closed.
    assert!(enumerate_study_portfolio(&studies, &[], 250.0, true).is_err());

    // With strong negative redundancy (-180) between s1 and s2, optimal within budget 250 is {s1, s3}.
    let interactions = vec![PortfolioInteraction {
        study_a: 0,
        study_b: 1,
        joint_evsi_adjustment: -180.0,
    }];
    let opt = enumerate_study_portfolio(&studies, &interactions, 250.0, true).unwrap();
    assert_eq!(opt.selected_indices, vec![0, 2]);
    assert!((opt.enbs - (220.0 + 150.0 - 180.0)).abs() < 1e-9);
}

#[test]
fn q06_equity_weighted_enbs_and_sequential_stopping_regret() {
    let eq = equity_weighted_study_enbs(&[100.0, 200.0], &[1.0, 3.0], 50.0).unwrap();
    assert!((eq - (175.0 - 50.0)).abs() < 1e-12);

    let seq =
        sequential_adaptive_stopping_regret(&[100.0, 180.0, 190.0], &[20.0, 60.0, 120.0]).unwrap();
    assert_eq!(seq.optimal_stage, 1);
    assert!((seq.optimal_net_value - 120.0).abs() < 1e-12);
    assert!((seq.immediate_stop_regret - 40.0).abs() < 1e-12);
}

#[test]
fn q06_error_branches_fail_closed() {
    let base = StudyScheduleConfig {
        discount_rate: 0.0,
        recruitment_delay_years: 1.0,
        time_horizon_years: 5.0,
        incident_population_per_year: 100.0,
    };
    assert!(discounted_population_enbs(1.0, 10.0, &base).unwrap() > 0.0);
    assert!(discounted_population_enbs(-1.0, 10.0, &base).is_err());
    assert!(discounted_population_enbs(1.0, -1.0, &base).is_err());

    assert!(enumerate_study_portfolio(&[], &[], 100.0, true).is_err());
    assert!(equity_weighted_study_enbs(&[], &[], 0.0).is_err());
    assert!(equity_weighted_study_enbs(&[10.0], &[0.0], 0.0).is_err());
    assert!(sequential_adaptive_stopping_regret(&[], &[]).is_err());
    assert!(sequential_adaptive_stopping_regret(&[-1.0], &[0.0]).is_err());
}
