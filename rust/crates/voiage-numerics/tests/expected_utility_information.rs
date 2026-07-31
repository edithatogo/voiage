//! Independent references for expected-utility information pricing.

use voiage_numerics::{
    expected_utility_information, ExpectedUtilityInformationInput, InformationStructure,
    SolverSettings, UtilityDescriptor,
};

use serde_json::Value;

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
    // At this price the risky action is outside the log domain under the
    // adverse signal, but safe remains feasible. Support-conditional action
    // infeasibility must not invalidate the valid contingent policy.
    assert_eq!(result.bpi_root.status, "converged");
    assert!(result
        .bpi_root
        .evaluated_policies
        .iter()
        .flat_map(|evaluation| &evaluation.policies)
        .flat_map(|policy| &policy.domain_exclusions)
        .any(|exclusion| {
            exclusion.signal_id.as_deref() == Some("state-1")
                && exclusion.action_id == "action-1"
                && exclusion.state_ids == ["state-1"]
                && exclusion.reason == "utility_domain"
        }));
    assert!(!result.bpi_root.policy_switched);
    assert!(result.bpi_root.transitions.is_empty());
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

fn strings(value: &Value) -> Vec<String> {
    value
        .as_array()
        .expect("string array")
        .iter()
        .map(|item| item.as_str().expect("string").to_owned())
        .collect()
}

fn numbers(value: &Value) -> Vec<f64> {
    value
        .as_array()
        .expect("number array")
        .iter()
        .map(|item| item.as_f64().expect("number"))
        .collect()
}

fn fixture_problem(payload: &Value) -> (ExpectedUtilityInformationInput, UtilityDescriptor) {
    let request = &payload["request"];
    let information = &request["information"];
    let solver = &request["solver"];
    let utility = &request["utility"];
    let problem = ExpectedUtilityInformationInput {
        schema_version: request["schema_version"].as_str().unwrap().into(),
        decision_problem_id: request["decision_problem_id"].as_str().unwrap().into(),
        stakeholder_scope_id: request["stakeholder_scope_id"].as_str().unwrap().into(),
        action_ids: strings(&request["action_ids"]),
        state_ids: strings(&request["state_ids"]),
        payoffs: request["payoffs"]
            .as_array()
            .unwrap()
            .iter()
            .map(numbers)
            .collect(),
        state_probabilities: numbers(&request["state_probabilities"]),
        initial_wealth: request["initial_wealth"].as_f64().unwrap(),
        payoff_unit: request["payoff_unit"].as_str().unwrap().into(),
        currency: request["currency"].as_str().map(Into::into),
        price_date: request["price_date"].as_str().map(Into::into),
        information_cost_location: request["information_cost_location"]
            .as_str()
            .unwrap()
            .into(),
        information: InformationStructure {
            kind: information["kind"].as_str().unwrap().into(),
            signal_ids: strings(&information["signal_ids"]),
            signal_state_probabilities: information["signal_state_probabilities"]
                .as_array()
                .unwrap()
                .iter()
                .map(numbers)
                .collect(),
        },
        terminal_outcome_floor: request["terminal_outcome_floor"].as_f64(),
        solver: SolverSettings {
            initial_upper: solver["initial_upper"].as_f64().unwrap(),
            expansion_factor: solver["expansion_factor"].as_f64().unwrap(),
            maximum_price: solver["maximum_price"].as_f64().unwrap(),
            absolute_price_tolerance: solver["absolute_price_tolerance"].as_f64().unwrap(),
            relative_price_tolerance: solver["relative_price_tolerance"].as_f64().unwrap(),
            utility_tolerance: solver["utility_tolerance"].as_f64().unwrap(),
            maximum_iterations: usize::try_from(solver["maximum_iterations"].as_u64().unwrap())
                .unwrap(),
            maximum_evaluations: usize::try_from(solver["maximum_evaluations"].as_u64().unwrap())
                .unwrap(),
        },
    };
    let descriptor = match utility["family"].as_str().unwrap() {
        "affine" => UtilityDescriptor::Affine {
            slope: utility["slope"].as_f64().unwrap(),
            intercept: utility["intercept"].as_f64().unwrap(),
        },
        "log" => UtilityDescriptor::Log {
            reference_wealth: utility["reference_wealth"].as_f64().unwrap(),
        },
        family => panic!("unsupported normative fixture utility: {family}"),
    };
    (problem, descriptor)
}

#[test]
fn committed_normative_fixtures_drive_rust_conformance() {
    let fixtures = [
        include_str!(
            "../../../../specs/frontier/expected-utility-information-pricing/v1/fixtures/normative/affine-clairvoyant.json"
        ),
        include_str!(
            "../../../../specs/frontier/expected-utility-information-pricing/v1/fixtures/normative/log-buy-sell-asymmetry.json"
        ),
    ];
    for fixture in fixtures {
        let payload: Value = serde_json::from_str(fixture).expect("valid normative fixture");
        let (problem, utility) = fixture_problem(&payload);
        let result = expected_utility_information(&problem, &utility).expect("fixture executes");
        let expected = &payload["expected"];
        for (actual, field, tolerance) in [
            (result.eui.value.unwrap(), "eui", 1.0e-9),
            (result.cei.value.unwrap(), "cei", 1.0e-8),
            (result.bpi.value.unwrap(), "bpi", 1.0e-7),
            (result.spi.value.unwrap(), "spi", 1.0e-7),
            (result.ppi.value.unwrap(), "ppi", 1.0e-10),
        ] {
            let reference = expected[field].as_f64().expect("numeric reference");
            assert!((actual - reference).abs() < tolerance, "{field}");
        }
    }
}
