use voiage_core::{
    evsi_contract, validate_reporting_payload, ApproximationStatus, DiagnosticStatus, DomainError,
    EvsiSummary, MethodMaturity, TrialArm, TrialDesign, ValueArray,
};

fn sample_value_array() -> ValueArray {
    ValueArray::new(
        "current-net-benefit-001",
        vec!["Strategy A".to_string(), "Strategy B".to_string()],
        vec![
            vec![8.0, 3.0],
            vec![7.0, 4.0],
            vec![1.0, 9.0],
            vec![2.0, 10.0],
        ],
    )
    .expect("valid sample array")
}

fn sample_trial_design() -> TrialDesign {
    TrialDesign::new(
        "screening-trial-design-001",
        vec![
            TrialArm::new("control", "Control", 1).expect("valid arm"),
            TrialArm::new("treatment", "Treatment", 2).expect("valid arm"),
        ],
    )
    .expect("valid trial design")
}

#[test]
fn evsi_contract_builds_a_deterministic_analysis_envelope() {
    let trial_design = sample_trial_design();
    let current_net_benefit = sample_value_array();

    let envelope = evsi_contract("evsi-001", &trial_design, &current_net_benefit, None)
        .expect("valid EVSI contract");

    assert_eq!(envelope.analysis_id, "evsi-001");
    assert_eq!(
        envelope.decision_problem_id.as_deref(),
        Some("screening-trial-design-001")
    );
    assert_eq!(envelope.analysis_type, "evsi");
    assert_eq!(envelope.method.as_deref(), Some("deterministic-summary"));
    assert_eq!(
        envelope.method_metadata.method_maturity,
        MethodMaturity::Approximate
    );
    assert_eq!(
        envelope.method_metadata.approximation_status,
        ApproximationStatus::Approximate
    );
    assert_eq!(envelope.diagnostics.status, DiagnosticStatus::Ok);
    assert_eq!(
        envelope.result,
        EvsiSummary {
            evsi: 1.166666666666666,
            trial_design_id: "screening-trial-design-001".to_string(),
            sample_size: 3,
            expected_current_value: 6.5,
            expected_sample_value: 7.666666666666666,
            expected_perfect_information: 8.5,
            method: Some("deterministic-summary".to_string()),
        }
    );
    validate_reporting_payload(&envelope.reporting).expect("reporting should be valid");
    assert_eq!(envelope.reporting.reporting_standard, "CHEERS-VOI");
    assert_eq!(envelope.reporting.analysis_type, "evsi");
    assert_eq!(envelope.reporting.method_family, "sample-information");
    assert_eq!(
        envelope.reporting.method_maturity,
        MethodMaturity::Approximate
    );
    assert_eq!(
        envelope.reporting.estimator.as_deref(),
        Some("deterministic-summary")
    );
    assert_eq!(
        envelope
            .reporting
            .provenance
            .get("trial_design_id")
            .map(String::as_str),
        Some("screening-trial-design-001")
    );
    assert_eq!(
        envelope
            .reporting
            .reproducibility
            .get("chunking_rule")
            .map(String::as_str),
        Some("contiguous_chunks_by_trial_sample_size")
    );
}

#[test]
fn evsi_contract_rejects_empty_trial_design() {
    let current_net_benefit = sample_value_array();
    let trial_design = TrialDesign {
        trial_design_id: "screening-trial-design-001".to_string(),
        arms: Vec::new(),
    };

    let err = evsi_contract("evsi-001", &trial_design, &current_net_benefit, None)
        .expect_err("empty designs are rejected");

    assert_eq!(err, DomainError::EmptyCollection("arms"));
}

#[test]
fn evsi_contract_rejects_invalid_value_array_shapes() {
    let trial_design = sample_trial_design();

    let mut mismatched_sample_count = sample_value_array();
    mismatched_sample_count.sample_count = 3;
    let sample_err = evsi_contract("evsi-002", &trial_design, &mismatched_sample_count, None)
        .expect_err("sample count mismatch is rejected");
    assert_eq!(
        sample_err,
        DomainError::SampleCountMismatch {
            expected: 4,
            actual: 3,
        }
    );

    let invalid_rows = ValueArray {
        value_array_id: "current-net-benefit-002".to_string(),
        sample_count: 2,
        strategy_names: vec!["Strategy A".to_string(), "Strategy B".to_string()],
        net_benefit: vec![vec![1.0, 2.0], vec![3.0]],
    };
    let ragged_err = evsi_contract("evsi-003", &trial_design, &invalid_rows, None)
        .expect_err("ragged rows are rejected");
    assert_eq!(ragged_err, DomainError::RaggedMatrix("net_benefit"));

    let invalid_values = ValueArray {
        value_array_id: "current-net-benefit-003".to_string(),
        sample_count: 2,
        strategy_names: vec!["Strategy A".to_string(), "Strategy B".to_string()],
        net_benefit: vec![vec![1.0, f64::NAN], vec![3.0, 4.0]],
    };
    let finite_err = evsi_contract("evsi-004", &trial_design, &invalid_values, None)
        .expect_err("non-finite values are rejected");
    assert_eq!(finite_err, DomainError::NonFinite("net_benefit"));
}
