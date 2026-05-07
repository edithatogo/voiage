use serde::Deserialize;
use voiage_core::{
    evsi_stochastic_contract, validate_reporting_payload, AnalysisEnvelope, ApproximationStatus,
    DiagnosticStatus, EvsiSummary, MethodMaturity, TrialDesign, ValueArray,
};

#[derive(Debug, Deserialize)]
struct EvsiStochasticKernelInput {
    analysis_id: String,
    trial_design: TrialDesign,
    current_net_benefit: ValueArray,
    seed: u64,
    method: Option<String>,
}

fn load_input() -> EvsiStochasticKernelInput {
    serde_json::from_str(include_str!("fixtures/evsi_stochastic_kernel.input.json"))
        .expect("stochastic EVSI input should deserialize")
}

fn load_expected() -> AnalysisEnvelope<EvsiSummary> {
    serde_json::from_str(include_str!(
        "fixtures/evsi_stochastic_kernel.expected.json"
    ))
    .expect("stochastic EVSI expected envelope should deserialize")
}

#[test]
fn evsi_stochastic_kernel_matches_the_expected_envelope() {
    let input = load_input();
    let expected = load_expected();

    let actual = evsi_stochastic_contract(
        input.analysis_id,
        &input.trial_design,
        &input.current_net_benefit,
        input.seed,
        input.method,
    )
    .expect("valid stochastic EVSI contract");

    assert_eq!(actual, expected);
    assert_eq!(actual.analysis_type, "evsi");
    assert_eq!(actual.diagnostics.status, DiagnosticStatus::Approximate);
    assert_eq!(
        actual.method_metadata.method_maturity,
        MethodMaturity::Approximate
    );
    assert_eq!(
        actual.method_metadata.approximation_status,
        ApproximationStatus::Approximate
    );
    assert!(actual.result.evsi >= 0.0);
    validate_reporting_payload(&actual.reporting).expect("reporting should remain valid");

    let serialized = serde_json::to_string_pretty(&actual).expect("serializes deterministically");
    assert_eq!(
        serialized,
        include_str!("fixtures/evsi_stochastic_kernel.expected.json").trim_end()
    );
}

#[test]
fn evsi_stochastic_kernel_fixture_is_stable() {
    let expected = load_expected();
    assert!(expected.result.evsi >= 0.0);
    assert_eq!(
        expected.reporting.estimator.as_deref(),
        Some("strided-bootstrap-summary")
    );
}
