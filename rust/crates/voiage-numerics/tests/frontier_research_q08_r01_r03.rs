//! Q08 risk-prediction and R01-R03 frontier research qualification tests.

use voiage_numerics::{
    decision_aware_simulation_allocation, joint_research_implementation_capacity,
    risk_prediction_threshold_evpi, transport_bias_evsi_bounds,
};

#[test]
fn q08_risk_prediction_threshold_evpi_is_nonnegative_and_bounded() {
    let probs = [0.05, 0.15, 0.25, 0.40, 0.60];
    let val = risk_prediction_threshold_evpi(&probs, 0.20).unwrap();
    assert!(val > 0.0 && val < 1.0);
    assert!(risk_prediction_threshold_evpi(&probs, 1.0).is_err());
}

#[test]
fn r01_decision_aware_allocation_concentrates_on_tight_margins() {
    let res =
        decision_aware_simulation_allocation(&[1.0, 1.0, 1.0], &[10.0, 0.05, 5.0], 100).unwrap();
    assert_eq!(res.allocations.iter().sum::<usize>(), 100);
    assert!(res.allocations[1] > res.allocations[0]);
    assert!(res.sequential_error_bound > 0.0);
}

#[test]
fn r02_joint_research_implementation_capacity_selects_feasible_optimum() {
    let res = joint_research_implementation_capacity(
        &[(120.0, 40.0)],
        &[(200.0, 50.0, 100.0), (50.0, 10.0, 20.0)],
        &[(50.0, 10.0), (120.0, 30.0)],
    )
    .unwrap();
    assert_eq!(res.research_index, Some(0));
    assert_eq!(res.implementation_index, 0);
    assert_eq!(res.capacity_index, 1);
    assert!((res.net_benefit - 200.0).abs() < 1e-12);
}

#[test]
fn r03_transport_bias_bounds_nest_nominal_evsi() {
    let b = transport_bias_evsi_bounds(50.0, 100.0, 0.2).unwrap();
    assert!((b.lower_bound_evsi - 40.0).abs() < 1e-12);
    assert!((b.upper_bound_evsi - 60.0).abs() < 1e-12);
    assert_eq!(b.maturity, "experimental_frontier_v1");
}
