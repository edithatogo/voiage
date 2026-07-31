use voiage_numerics::{
    expected_utility_information, ExpectedUtilityInformationInput, InformationStructure,
    SolverSettings, UtilityDescriptor,
};

fn base_input() -> ExpectedUtilityInformationInput {
    ExpectedUtilityInformationInput {
        schema_version: "expected-utility-information-input-v1".into(),
        decision_problem_id: "pathology".into(),
        stakeholder_scope_id: "stakeholder".into(),
        action_ids: vec!["zeta".into(), "alpha".into()],
        state_ids: vec!["low".into(), "high".into()],
        payoffs: vec![vec![0.0, 0.0], vec![0.0, 0.0]],
        state_probabilities: vec![0.7, 0.3],
        initial_wealth: 10.0,
        payoff_unit: "USD".into(),
        currency: Some("USD".into()),
        price_date: Some("2026-07-31".into()),
        information_cost_location: "ex_ante_sure_transfer".into(),
        information: InformationStructure {
            kind: "clairvoyant".into(),
            signal_ids: vec!["low".into(), "high".into()],
            signal_state_probabilities: vec![vec![0.7, 0.0], vec![0.0, 0.3]],
        },
        terminal_outcome_floor: Some(1.0),
        solver: SolverSettings::default(),
    }
}

#[test]
fn complete_ties_are_sorted_by_action_id() {
    let result = expected_utility_information(
        &base_input(),
        &UtilityDescriptor::Affine {
            slope: 1.0,
            intercept: 0.0,
        },
    )
    .expect("valid tied problem");

    assert_eq!(result.current_policy.tie_set, vec!["alpha", "zeta"]);
    assert_eq!(result.current_policy.representative_action_id, "alpha");
    assert_eq!(result.bpi.value, Some(0.0));
    assert_eq!(result.bpi_root.status, "zero_boundary");
}

#[test]
fn positive_affine_rescaling_preserves_prices_and_policies() {
    let mut problem = base_input();
    problem.payoffs = vec![vec![0.0, 4.0], vec![2.0, 0.0]];
    let canonical = expected_utility_information(
        &problem,
        &UtilityDescriptor::Affine {
            slope: 1.0,
            intercept: 0.0,
        },
    )
    .unwrap();
    let rescaled = expected_utility_information(
        &problem,
        &UtilityDescriptor::Affine {
            slope: 17.0,
            intercept: -81.0,
        },
    )
    .unwrap();

    assert_eq!(canonical.current_policy, rescaled.current_policy);
    assert!((canonical.bpi.value.unwrap() - rescaled.bpi.value.unwrap()).abs() < 1e-9);
    assert!((canonical.spi.value.unwrap() - rescaled.spi.value.unwrap()).abs() < 1e-9);
    assert!((rescaled.eui.value.unwrap() - 17.0 * canonical.eui.value.unwrap()).abs() < 1e-9);
}

#[test]
fn utility_domain_and_probability_failures_are_fail_closed() {
    let mut domain = base_input();
    domain.initial_wealth = 0.0;
    let error = expected_utility_information(
        &domain,
        &UtilityDescriptor::Log {
            reference_wealth: 1.0,
        },
    )
    .expect_err("zero terminal wealth is outside log domain");
    assert_eq!(error.code(), "utility_domain");

    let mut probabilities = base_input();
    probabilities.state_probabilities = vec![0.7, 0.4];
    let error = expected_utility_information(
        &probabilities,
        &UtilityDescriptor::Affine {
            slope: 1.0,
            intercept: 0.0,
        },
    )
    .expect_err("probabilities must sum to one");
    assert_eq!(error.code(), "invalid_probability");
}

#[test]
fn bounded_search_reports_unbracketed_without_fabricating_a_price() {
    let mut problem = base_input();
    problem.payoffs = vec![vec![0.0, 100.0], vec![100.0, 0.0]];
    problem.solver.initial_upper = 0.01;
    problem.solver.maximum_price = 0.02;
    let result = expected_utility_information(
        &problem,
        &UtilityDescriptor::Affine {
            slope: 1.0,
            intercept: 0.0,
        },
    )
    .unwrap();

    assert_eq!(result.bpi.status, "failed");
    assert_eq!(result.bpi.value, None);
    assert_eq!(result.bpi_root.status, "not_bracketed");
    assert!(result.bpi_root.estimate.is_none());
}

#[test]
fn result_retains_stakeholder_scope_and_cross_problem_requirements() {
    let result = expected_utility_information(
        &base_input(),
        &UtilityDescriptor::Affine {
            slope: 1.0,
            intercept: 0.0,
        },
    )
    .unwrap();
    assert_eq!(result.comparability.stakeholder_scope_id, "stakeholder");
    assert!(!result.comparability.cross_problem_comparable);
    assert!(result
        .comparability
        .required_shared_fields
        .contains(&"stakeholder_scope_id".into()));
}

#[test]
fn bounded_solver_reports_iteration_and_evaluation_limits() {
    let mut problem = base_input();
    problem.payoffs = vec![vec![0.0, 5.0], vec![0.0, -9.0]];
    problem.state_probabilities = vec![0.8, 0.2];
    problem.information.signal_state_probabilities = vec![vec![0.8, 0.0], vec![0.0, 0.2]];
    problem.solver.maximum_iterations = 1;
    let result = expected_utility_information(
        &problem,
        &UtilityDescriptor::Log {
            reference_wealth: 1.0,
        },
    )
    .unwrap();
    assert_eq!(result.bpi_root.status, "max_iterations");
    assert_eq!(result.bpi.value, None);

    problem.solver.maximum_iterations = 200;
    problem.solver.maximum_evaluations = 2;
    let result = expected_utility_information(
        &problem,
        &UtilityDescriptor::Log {
            reference_wealth: 1.0,
        },
    )
    .unwrap();
    assert_eq!(result.bpi_root.status, "max_evaluations");
    assert_eq!(result.bpi.value, None);
}
