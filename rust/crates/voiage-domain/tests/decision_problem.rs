//! Contract tests for decision problems and reproducibility metadata.

use std::collections::BTreeMap;

use voiage_domain::{
    DecisionProblem, ExecutionMode, Identifier, Intervention, Provenance, ReproducibilityMetadata,
    Seed, Threshold,
};

fn intervention(id: &str, name: &str) -> Intervention {
    Intervention::new(
        Identifier::new(id).unwrap(),
        Identifier::new(name).unwrap(),
        None,
        false,
        None,
    )
}

fn provenance(fixture_id: Option<&str>) -> Provenance {
    Provenance::new(
        Identifier::new("1.0.0").unwrap(),
        Identifier::new("rust-core-v1").unwrap(),
        Identifier::new("evpi").unwrap(),
        BTreeMap::from([("tolerance".to_owned(), "1e-10".to_owned())]),
        fixture_id.map(|value| Identifier::new(value).unwrap()),
        Some(Identifier::new("rust").unwrap()),
    )
}

#[test]
fn decision_problem_requires_unique_nonempty_interventions() {
    let result = DecisionProblem::new(
        Identifier::new("screening-001").unwrap(),
        Identifier::new("Screening programme").unwrap(),
        Identifier::new("AUD").unwrap(),
        Threshold::new(50_000.0).unwrap(),
        None,
        vec![
            intervention("usual-care", "Usual care"),
            intervention("screen", "Screen"),
        ],
    )
    .unwrap();

    assert_eq!(result.decision_problem_id().as_str(), "screening-001");
    assert_eq!(result.interventions().len(), 2);
    assert_eq!(
        result.willingness_to_pay().get().to_bits(),
        50_000.0_f64.to_bits()
    );

    assert!(DecisionProblem::new(
        Identifier::new("screening-001").unwrap(),
        Identifier::new("Screening programme").unwrap(),
        Identifier::new("AUD").unwrap(),
        Threshold::new(50_000.0).unwrap(),
        None,
        vec![],
    )
    .is_err());
    assert!(DecisionProblem::new(
        Identifier::new("screening-001").unwrap(),
        Identifier::new("Screening programme").unwrap(),
        Identifier::new("AUD").unwrap(),
        Threshold::new(50_000.0).unwrap(),
        None,
        vec![intervention("same", "A"), intervention("same", "B")],
    )
    .is_err());
}

#[test]
fn decision_problem_enforces_currency_and_outcome_schema_constraints() {
    let base = || {
        (
            Identifier::new("screening-001").unwrap(),
            Identifier::new("Screening programme").unwrap(),
            Threshold::new(50_000.0).unwrap(),
            vec![intervention("usual-care", "Usual care")],
        )
    };

    let (id, title, threshold, interventions) = base();
    assert!(DecisionProblem::new(
        id,
        title,
        Identifier::new("AU").unwrap(),
        threshold,
        None,
        interventions,
    )
    .is_err());

    let (id, title, threshold, interventions) = base();
    assert!(DecisionProblem::new(
        id,
        title,
        Identifier::new("AUD").unwrap(),
        threshold,
        Some(vec![]),
        interventions,
    )
    .is_err());

    let (id, title, threshold, interventions) = base();
    let outcome = Identifier::new("QALY").unwrap();
    assert!(DecisionProblem::new(
        id,
        title,
        Identifier::new("AUD").unwrap(),
        threshold,
        Some(vec![outcome.clone(), outcome]),
        interventions,
    )
    .is_err());
}

#[test]
fn intervention_and_decision_problem_serde_are_fail_closed() {
    let valid = r#"{
      "decision_problem_id":"screening-001",
      "title":"Screening programme",
      "analysis_type":"net-benefit-first",
      "currency":"AUD",
      "willingness_to_pay":50000.0,
      "interventions":[{"intervention_id":"usual-care","name":"Usual care","is_reference":true}]
    }"#;
    let decoded: DecisionProblem = serde_json::from_str(valid).unwrap();
    assert_eq!(decoded.interventions()[0].name().as_str(), "Usual care");
    assert!(serde_json::from_str::<DecisionProblem>(&valid.replace("50000.0", "0.0")).is_err());
    assert!(serde_json::from_str::<DecisionProblem>(&valid.replace("\"AUD\"", "\"AU\"")).is_err());
    assert!(serde_json::from_str::<DecisionProblem>(&valid.replace(
        "\"interventions\":[",
        "\"outcome_names\":[],\"interventions\":["
    ))
    .is_err());
    assert!(serde_json::from_str::<DecisionProblem>(&valid.replace(
        "\"interventions\":[",
        "\"outcome_names\":[\"QALY\",\"QALY\"],\"interventions\":["
    ))
    .is_err());
    assert!(serde_json::from_str::<DecisionProblem>(&valid.replace(
        "\"interventions\":[",
        "\"unknown\":true,\"interventions\":["
    ))
    .is_err());
    assert!(
        serde_json::from_str::<Intervention>(r#"{"intervention_id":"","name":"Bad"}"#).is_err()
    );
}

#[test]
fn reproducibility_enforces_fixture_and_stochastic_requirements() {
    let fixture = ReproducibilityMetadata::new(
        None,
        ExecutionMode::Deterministic,
        true,
        provenance(Some("evpi-normative-001")),
    )
    .unwrap();
    assert!(fixture.deterministic_fixture_mode());

    assert!(ReproducibilityMetadata::new(
        None,
        ExecutionMode::Deterministic,
        true,
        provenance(None),
    )
    .is_err());
    assert!(
        ReproducibilityMetadata::new(None, ExecutionMode::Stochastic, false, provenance(None),)
            .is_err()
    );

    let stochastic = ReproducibilityMetadata::new(
        Some(Seed::new(42)),
        ExecutionMode::Stochastic,
        false,
        provenance(None),
    )
    .unwrap();
    assert_eq!(stochastic.seed().unwrap().get(), 42);
}

#[test]
fn reproducibility_serde_cannot_bypass_validation() {
    let missing_fixture = r#"{
      "seed":null,
      "execution_mode":"deterministic",
      "deterministic_fixture_mode":true,
      "provenance":{"voiage_version":"1.0.0","core_version":"rust-v1","method":"evpi","settings":{}}
    }"#;
    assert!(serde_json::from_str::<ReproducibilityMetadata>(missing_fixture).is_err());

    let missing_seed = missing_fixture
        .replace("\"deterministic\"", "\"stochastic\"")
        .replace("true", "false");
    assert!(serde_json::from_str::<ReproducibilityMetadata>(&missing_seed).is_err());
    assert!(serde_json::from_str::<ReproducibilityMetadata>(
        &missing_seed.replace("\"provenance\":", "\"unexpected\":1,\"provenance\":")
    )
    .is_err());
}
