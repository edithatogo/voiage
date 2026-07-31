//! Independent references for expected-utility information pricing.

use voiage_numerics::{
    expected_utility_information, ExpectedUtilityInformationInput, InformationStructure,
    SolverSettings, UtilityDescriptor,
};

fn clairvoyant_joint(probabilities: &[f64]) -> Vec<Vec<f64>> {
    probabilities
        .iter()
        .enumerate()
        .map(|(row, probability)| {
            probabilities
                .iter()
                .enumerate()
                .map(|(column, _)| if row == column { *probability } else { 0.0 })
                .collect()
        })
        .collect()
}

#[allow(clippy::needless_pass_by_value)]
fn input(
    payoffs: Vec<Vec<f64>>,
    probabilities: Vec<f64>,
    wealth: f64,
) -> ExpectedUtilityInformationInput {
    let states = probabilities.len();
    let actions = payoffs.first().map_or(0, Vec::len);
    ExpectedUtilityInformationInput {
        schema_version: "expected-utility-information-input-v1".into(),
        decision_problem_id: "reference-problem".into(),
        stakeholder_scope_id: "reference-stakeholder".into(),
        action_ids: (0..actions)
            .map(|index| format!("action-{index}"))
            .collect(),
        state_ids: (0..states).map(|index| format!("state-{index}")).collect(),
        payoffs,
        state_probabilities: probabilities.clone(),
        initial_wealth: wealth,
        payoff_unit: "USD".into(),
        currency: Some("USD".into()),
        price_date: Some("2026-07-31".into()),
        information_cost_location: "ex_ante_sure_transfer".into(),
        information: InformationStructure {
            kind: "clairvoyant".into(),
            signal_ids: (0..states).map(|index| format!("state-{index}")).collect(),
            signal_state_probabilities: clairvoyant_joint(&probabilities),
        },
        terminal_outcome_floor: Some(1.0),
        solver: SolverSettings::default(),
    }
}

#[test]
fn affine_clairvoyance_reduces_to_monetary_evpi() {
    let problem = input(vec![vec![0.0, 4.0], vec![2.0, 0.0]], vec![0.5, 0.5], 10.0);
    let result = expected_utility_information(
        &problem,
        &UtilityDescriptor::Affine {
            slope: 2.0,
            intercept: 7.0,
        },
    )
    .expect("valid affine reference");

    assert!((result.eui.value.unwrap() - 2.0).abs() < 1e-12);
    assert!((result.cei.value.unwrap() - 1.0).abs() < 1e-12);
    assert!((result.bpi.value.unwrap() - 1.0).abs() < 1e-8);
    assert!((result.spi.value.unwrap() - 1.0).abs() < 1e-8);
    assert!((result.ppi.value.unwrap() - (1.0 / 12.0)).abs() < 1e-12);
    assert_eq!(
        result.affine_reduction.monetary_measure,
        Some("evpi".into())
    );
    assert!((result.affine_reduction.value.unwrap() - 1.0).abs() < 1e-12);
}

#[test]
fn logarithmic_reference_preserves_buy_sell_asymmetry() {
    let problem = input(vec![vec![0.0, 5.0], vec![0.0, -9.0]], vec![0.8, 0.2], 10.0);
    let result = expected_utility_information(
        &problem,
        &UtilityDescriptor::Log {
            reference_wealth: 1.0,
        },
    )
    .expect("valid logarithmic reference");

    assert!((result.eui.value.unwrap() - 0.324_372_086_5).abs() < 1e-9);
    assert!((result.cei.value.unwrap() - 3.831_618_672_2).abs() < 1e-8);
    assert!((result.bpi.value.unwrap() - 3.752_188_661_0).abs() < 1e-7);
    assert!((result.spi.value.unwrap() - 3.408_503_026_1).abs() < 1e-7);
    assert!((result.ppi.value.unwrap() - 0.123_478_254_240_394_05).abs() < 1e-10);
    assert_ne!(result.bpi.value, result.spi.value);
    assert_eq!(result.affine_reduction.status, "unavailable");
}

#[test]
fn exponential_reference_has_translation_invariant_prices() {
    let problem = input(vec![vec![0.0, 4.0], vec![2.0, 0.0]], vec![0.5, 0.5], 10.0);
    let result = expected_utility_information(
        &problem,
        &UtilityDescriptor::Exponential {
            risk_tolerance: 10.0,
            reference_wealth: 0.0,
        },
    )
    .expect("valid exponential reference");

    let baseline = -0.5 * (-1.0_f64).exp() - 0.5 * (-1.4_f64).exp();
    let informed = -0.5 * (-1.4_f64).exp() - 0.5 * (-1.2_f64).exp();
    let expected_cei = -10.0 * (-informed).ln() - (-10.0 * (-baseline).ln());
    assert!((result.eui.value.unwrap() - (informed - baseline)).abs() < 1e-12);
    assert!((result.cei.value.unwrap() - expected_cei).abs() < 1e-12);
    assert!((result.bpi.value.unwrap() - expected_cei).abs() < 1e-8);
    assert!((result.spi.value.unwrap() - expected_cei).abs() < 1e-8);
}

#[test]
fn finite_signal_uses_joint_probabilities_not_clairvoyant_shortcut() {
    let mut problem = input(vec![vec![0.0, 4.0], vec![2.0, 0.0]], vec![0.5, 0.5], 10.0);
    problem.information = InformationStructure {
        kind: "finite_signal".into(),
        signal_ids: vec!["negative".into(), "positive".into()],
        signal_state_probabilities: vec![vec![0.4, 0.1], vec![0.1, 0.4]],
    };
    let result = expected_utility_information(
        &problem,
        &UtilityDescriptor::Affine {
            slope: 1.0,
            intercept: 0.0,
        },
    )
    .expect("valid finite signal");

    assert!((result.eui.value.unwrap() - 0.4).abs() < 1e-12);
    assert_eq!(result.information_kind, "finite_signal");
    assert_eq!(
        result.affine_reduction.monetary_measure,
        Some("evsi".into())
    );
}
