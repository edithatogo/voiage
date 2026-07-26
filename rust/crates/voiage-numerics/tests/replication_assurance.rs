//! Independent-replication convergence contracts for stochastic estimators.

use voiage_numerics::summarize_replications;

#[test]
fn replicated_estimates_report_between_run_error_and_convergence() {
    let summary = summarize_replications(&[1.0, 1.1, 0.9, 1.0], 0.2)
        .expect("four finite independent estimates");

    assert_eq!(summary.replications, 4);
    assert!((summary.mean - 1.0).abs() <= 1.0e-12);
    assert!(summary.variance > 0.0);
    assert!(summary.standard_error > 0.0);
    assert!(summary.converged);
    assert!(summary.split_mean_relative_difference <= 0.2);
}

#[test]
fn replication_summary_requires_two_estimates_and_a_positive_tolerance() {
    assert!(summarize_replications(&[1.0], 0.1).is_err());
    assert!(summarize_replications(&[1.0, 1.1], 0.0).is_err());
    assert!(summarize_replications(&[1.0, f64::NAN], 0.1).is_err());
}
