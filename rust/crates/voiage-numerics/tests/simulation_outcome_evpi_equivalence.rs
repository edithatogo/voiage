//! Analytical EVPI equivalence and joint-draw permutation tests across `O07`.

use serde_json::Value;
use voiage_domain::{
    ObjectiveDirection, SampleCube, SampleVector, SimulationOutcomeProfileInput,
    ValidatedSimulationOutcomeProfile, ValueConvention, WeightSemantics,
};
use voiage_numerics::{
    adapt_simulation_outcome, evpi, simulation_outcome_evpi, structural_evpi,
    SimulationOutcomeAdapterError,
};

fn load_evpi_fixtures() -> Value {
    let raw = include_str!("../../../../specs/ecosystem/simulation-outcome/v1/evpi-fixtures.json");
    serde_json::from_str(raw).expect("valid O07 fixture JSON")
}

#[test]
fn adapter_evpi_matches_analytical_reference_and_existing_rust_kernels() {
    let fixtures = load_evpi_fixtures();
    let tol = fixtures["tolerance"].as_f64().expect("tolerance float");
    let cases = fixtures["accepted"]
        .as_array()
        .expect("accepted fixture array");
    assert!(cases.len() >= 3);

    for case in cases {
        let case_id = case["case_id"].as_str().expect("case_id");
        let expected = case["expected_evpi"].as_f64().expect("expected_evpi");
        let input: SimulationOutcomeProfileInput =
            serde_json::from_value(case["profile"].clone()).expect("profile input");
        let validated =
            ValidatedSimulationOutcomeProfile::validate(&input).expect("valid outcome profile");

        let result = simulation_outcome_evpi(&validated).expect("adapter evpi");
        assert!(
            (result.evpi() - expected).abs() <= tol,
            "{case_id}: adapter evpi {} != expected {expected}",
            result.evpi()
        );
        assert_eq!(result.provenance(), validated.provenance());
        assert_eq!(result.value_declaration(), validated.value_declaration());
        assert_eq!(result.strategy_ids(), validated.strategy_ids());

        let handoff = adapt_simulation_outcome(&validated).expect("handoff");
        let kernel_evpi = match result.weight_semantics() {
            WeightSemantics::Equal => evpi(handoff.value_array().net_benefit()).expect("core evpi"),
            WeightSemantics::Probability => {
                let cube = SampleCube::try_from(
                    handoff
                        .value_array()
                        .net_benefit()
                        .rows()
                        .map(|row| vec![row.to_vec()])
                        .collect::<Vec<_>>(),
                )
                .expect("cube");
                let probs =
                    SampleVector::try_from(handoff.draw_weights().to_vec()).expect("probabilities");
                structural_evpi(&cube, &probs).expect("structural evpi")
            }
            WeightSemantics::RelativeWeight => {
                let weights = handoff.draw_weights();
                let max_w = weights.iter().copied().fold(0.0_f64, f64::max);
                let scaled: Vec<f64> = weights.iter().map(|w| w / max_w).collect();
                let denom: f64 = scaled.iter().sum();
                let probs = SampleVector::try_from(
                    scaled.into_iter().map(|w| w / denom).collect::<Vec<_>>(),
                )
                .expect("normalized weights");
                let cube = SampleCube::try_from(
                    handoff
                        .value_array()
                        .net_benefit()
                        .rows()
                        .map(|row| vec![row.to_vec()])
                        .collect::<Vec<_>>(),
                )
                .expect("cube");
                structural_evpi(&cube, &probs).expect("structural evpi")
            }
        };
        assert!(
            (result.evpi() - kernel_evpi).abs() <= tol,
            "{case_id}: adapter evpi {} != kernel evpi {kernel_evpi}",
            result.evpi()
        );
    }
}

#[test]
fn joint_permutation_preserves_evpi_and_independent_permutation_diverges() {
    let fixtures = load_evpi_fixtures();
    let tol = fixtures["tolerance"].as_f64().expect("tolerance float");

    let invariant_case = &fixtures["joint_permutation_invariant"];
    let invariant_input: SimulationOutcomeProfileInput =
        serde_json::from_value(invariant_case["profile"].clone()).expect("invariant profile");
    let invariant_validated =
        ValidatedSimulationOutcomeProfile::validate(&invariant_input).expect("valid invariant");
    let invariant_result = simulation_outcome_evpi(&invariant_validated).expect("invariant evpi");
    let baseline_expected = invariant_case["expected_evpi"]
        .as_f64()
        .expect("baseline expected evpi");
    assert!((invariant_result.evpi() - baseline_expected).abs() <= tol);

    let counter = &fixtures["independent_strategy_permutation_counterexample"];
    let counter_input: SimulationOutcomeProfileInput =
        serde_json::from_value(counter["profile"].clone()).expect("counterexample profile");
    let counter_validated =
        ValidatedSimulationOutcomeProfile::validate(&counter_input).expect("valid counterexample");
    let counter_result = simulation_outcome_evpi(&counter_validated).expect("counterexample evpi");
    let counter_expected = counter["counterexample_evpi"]
        .as_f64()
        .expect("counterexample expected evpi");
    assert!((counter_result.evpi() - counter_expected).abs() <= tol);
    assert!((counter_result.evpi() - invariant_result.evpi()).abs() > 1.0);
}

#[test]
fn minimized_loss_and_weight_sum_drift_fail_explicitly() {
    let fixtures = load_evpi_fixtures();
    let loss_input: SimulationOutcomeProfileInput =
        serde_json::from_value(fixtures["rejected_minimize_loss"].clone()).expect("loss profile");
    let loss_validated =
        ValidatedSimulationOutcomeProfile::validate(&loss_input).expect("domain-valid loss");
    let loss_err = simulation_outcome_evpi(&loss_validated).expect_err("minimize rejected");
    assert_eq!(
        loss_err,
        SimulationOutcomeAdapterError::UnsupportedObjectiveDirection {
            convention: ValueConvention::Loss,
            objective: ObjectiveDirection::Minimize,
        }
    );

    let mut prob_input: SimulationOutcomeProfileInput =
        serde_json::from_value(fixtures["accepted"][1]["profile"].clone()).expect("prob profile");
    for record in &mut prob_input.outcomes {
        if record.draw_key == "s0" {
            record.weight += 5.0e-10;
        }
    }
    let drifted = ValidatedSimulationOutcomeProfile::validate(&prob_input)
        .expect("passes domain 1e-9 tolerance");
    let err = simulation_outcome_evpi(&drifted)
        .expect_err("exceeds kernel 1e-12 probability sum check without rescaling");
    assert!(matches!(err, SimulationOutcomeAdapterError::Numerical(_)));
}
