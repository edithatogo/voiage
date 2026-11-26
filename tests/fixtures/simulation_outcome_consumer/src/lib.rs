//! Isolated installed-crate consumer exercising `voiage-domain` and `voiage-numerics`
//! without Kairos, game, or HEOR producer engine dependencies (`O08`).

use voiage_domain::{SimulationOutcomeProfileInput, ValidatedSimulationOutcomeProfile};
use voiage_numerics::{simulation_outcome_evpi, SimulationOutcomeEvpiResult};

/// Validates a JSON `voiage.simulation-outcome.v1` payload and computes EVPI.
///
/// # Errors
///
/// Returns an error string if JSON parsing, profile validation, or EVPI
/// computation fails.
pub fn evaluate_outcome_json_evpi(raw_json: &str) -> Result<SimulationOutcomeEvpiResult, String> {
    let input: SimulationOutcomeProfileInput =
        serde_json::from_str(raw_json).map_err(|err| err.to_string())?;
    let validated =
        ValidatedSimulationOutcomeProfile::validate(&input).map_err(|err| err.to_string())?;
    simulation_outcome_evpi(&validated).map_err(|err| err.to_string())
}

#[cfg(test)]
mod tests {
    use super::evaluate_outcome_json_evpi;
    use serde_json::json;
    use voiage_domain::{
        ObjectiveDirection, SimulationOutcomeProfileInput, ValidatedSimulationOutcomeProfile,
        ValueConvention, WeightSemantics,
    };
    use voiage_numerics::{
        adapt_simulation_outcome, simulation_outcome_evpi, SimulationOutcomeAdapterError,
    };

    const TOLERANCE: f64 = 1.0e-12;

    #[test]
    fn synthetic_health_profile_preserves_evpi_and_parameters() {
        let payload = json!({
            "profile_version": "voiage.simulation-outcome.v1",
            "provenance": {
                "producer_id": "synthetic-heor-consumer",
                "producer_version": "0.1.0",
                "model_revision": "heor-rev-1",
                "valuation_source": "synthetic-qaly-nmb",
                "artifact_digest": "1111111111111111111111111111111111111111111111111111111111111111"
            },
            "strategy_ids": ["standard_care", "intervention"],
            "value_declaration": {
                "unit": "NZD",
                "perspective": "health_system",
                "convention": "net_benefit",
                "objective": "maximize",
                "currency_price_year": "NZD-2026",
                "population_basis": "per_patient",
                "horizon": "5y",
                "discounting_status": "discounted_3.5pct"
            },
            "weight_semantics": "equal",
            "draw_namespace": "synthetic.consumer.health.v1",
            "outcomes": [
                {"draw_namespace": "synthetic.consumer.health.v1", "draw_key": "d2", "strategy_id": "intervention", "value": 6.0, "weight": 1.0},
                {"draw_namespace": "synthetic.consumer.health.v1", "draw_key": "d0", "strategy_id": "standard_care", "value": 10.0, "weight": 1.0},
                {"draw_namespace": "synthetic.consumer.health.v1", "draw_key": "d3", "strategy_id": "intervention", "value": 10.0, "weight": 1.0},
                {"draw_namespace": "synthetic.consumer.health.v1", "draw_key": "d1", "strategy_id": "standard_care", "value": 2.0, "weight": 1.0},
                {"draw_namespace": "synthetic.consumer.health.v1", "draw_key": "d2", "strategy_id": "standard_care", "value": 6.0, "weight": 1.0},
                {"draw_namespace": "synthetic.consumer.health.v1", "draw_key": "d0", "strategy_id": "intervention", "value": 4.0, "weight": 1.0},
                {"draw_namespace": "synthetic.consumer.health.v1", "draw_key": "d3", "strategy_id": "standard_care", "value": 0.0, "weight": 1.0},
                {"draw_namespace": "synthetic.consumer.health.v1", "draw_key": "d1", "strategy_id": "intervention", "value": 8.0, "weight": 1.0}
            ],
            "parameter_samples": [
                {"draw_namespace": "synthetic.consumer.health.v1", "draw_key": "d3", "values": {"risk": 0.4}},
                {"draw_namespace": "synthetic.consumer.health.v1", "draw_key": "d0", "values": {"risk": 0.1}},
                {"draw_namespace": "synthetic.consumer.health.v1", "draw_key": "d2", "values": {"risk": 0.3}},
                {"draw_namespace": "synthetic.consumer.health.v1", "draw_key": "d1", "values": {"risk": 0.2}}
            ]
        });
        let input: SimulationOutcomeProfileInput =
            serde_json::from_value(payload).expect("valid synthetic health input");
        let validated =
            ValidatedSimulationOutcomeProfile::validate(&input).expect("validated health profile");
        let handoff = adapt_simulation_outcome(&validated).expect("handoff");
        let result =
            evaluate_outcome_json_evpi(&serde_json::to_string(&input).expect("serialize"))
                .expect("evpi via consumer helper");

        assert!((result.evpi() - 1.5).abs() <= TOLERANCE);
        assert_eq!(result.weight_semantics(), WeightSemantics::Equal);
        assert_eq!(
            handoff
                .parameter_set()
                .expect("parameters")
                .parameter("risk")
                .expect("risk")
                .as_slice(),
            &[0.1, 0.2, 0.3, 0.4]
        );
    }

    #[test]
    fn synthetic_non_health_probability_profile_preserves_evpi() {
        let payload = json!({
            "profile_version": "voiage.simulation-outcome.v1",
            "provenance": {
                "producer_id": "synthetic-ops-consumer",
                "producer_version": "0.2.0",
                "model_revision": "ops-rev-2",
                "valuation_source": "synthetic-utility",
                "artifact_digest": "2222222222222222222222222222222222222222222222222222222222222222"
            },
            "strategy_ids": ["hold", "reallocate", "hedge"],
            "value_declaration": {
                "unit": "utils",
                "perspective": "portfolio_owner",
                "convention": "utility",
                "objective": "maximize"
            },
            "weight_semantics": "probability",
            "draw_namespace": "synthetic.consumer.ops.v1",
            "outcomes": [
                {"draw_namespace": "synthetic.consumer.ops.v1", "draw_key": "s0", "strategy_id": "hold", "value": 15.0, "weight": 0.2},
                {"draw_namespace": "synthetic.consumer.ops.v1", "draw_key": "s0", "strategy_id": "reallocate", "value": 5.0, "weight": 0.2},
                {"draw_namespace": "synthetic.consumer.ops.v1", "draw_key": "s0", "strategy_id": "hedge", "value": 10.0, "weight": 0.2},
                {"draw_namespace": "synthetic.consumer.ops.v1", "draw_key": "s1", "strategy_id": "hold", "value": 4.0, "weight": 0.5},
                {"draw_namespace": "synthetic.consumer.ops.v1", "draw_key": "s1", "strategy_id": "reallocate", "value": 12.0, "weight": 0.5},
                {"draw_namespace": "synthetic.consumer.ops.v1", "draw_key": "s1", "strategy_id": "hedge", "value": 8.0, "weight": 0.5},
                {"draw_namespace": "synthetic.consumer.ops.v1", "draw_key": "s2", "strategy_id": "hold", "value": 6.0, "weight": 0.3},
                {"draw_namespace": "synthetic.consumer.ops.v1", "draw_key": "s2", "strategy_id": "reallocate", "value": 2.0, "weight": 0.3},
                {"draw_namespace": "synthetic.consumer.ops.v1", "draw_key": "s2", "strategy_id": "hedge", "value": 16.0, "weight": 0.3}
            ]
        });
        let input: SimulationOutcomeProfileInput =
            serde_json::from_value(payload).expect("valid synthetic non-health input");
        let validated =
            ValidatedSimulationOutcomeProfile::validate(&input).expect("validated profile");
        let result = simulation_outcome_evpi(&validated).expect("evpi");

        assert!((result.evpi() - 3.0).abs() <= TOLERANCE);
        assert_eq!(result.weight_semantics(), WeightSemantics::Probability);
    }

    #[test]
    fn synthetic_relative_weights_and_independent_permutation_counterexample() {
        let weighted_payload = json!({
            "profile_version": "voiage.simulation-outcome.v1",
            "provenance": {
                "producer_id": "synthetic-grid-consumer",
                "producer_version": "0.3.0",
                "model_revision": "grid-rev-1",
                "valuation_source": "synthetic-net-value",
                "artifact_digest": "3333333333333333333333333333333333333333333333333333333333333333"
            },
            "strategy_ids": ["keep", "upgrade"],
            "value_declaration": {
                "unit": "USD",
                "perspective": "grid_operator",
                "convention": "net_benefit",
                "objective": "maximize"
            },
            "weight_semantics": "relative_weight",
            "draw_namespace": "synthetic.consumer.grid.v1",
            "outcomes": [
                {"draw_namespace": "synthetic.consumer.grid.v1", "draw_key": "w0", "strategy_id": "keep", "value": 1000.0, "weight": 0.0},
                {"draw_namespace": "synthetic.consumer.grid.v1", "draw_key": "w0", "strategy_id": "upgrade", "value": -500.0, "weight": 0.0},
                {"draw_namespace": "synthetic.consumer.grid.v1", "draw_key": "w1", "strategy_id": "keep", "value": 20.0, "weight": 1.0},
                {"draw_namespace": "synthetic.consumer.grid.v1", "draw_key": "w1", "strategy_id": "upgrade", "value": 10.0, "weight": 1.0},
                {"draw_namespace": "synthetic.consumer.grid.v1", "draw_key": "w2", "strategy_id": "keep", "value": 10.0, "weight": 3.0},
                {"draw_namespace": "synthetic.consumer.grid.v1", "draw_key": "w2", "strategy_id": "upgrade", "value": 30.0, "weight": 3.0},
                {"draw_namespace": "synthetic.consumer.grid.v1", "draw_key": "w3", "strategy_id": "keep", "value": 15.0, "weight": 6.0},
                {"draw_namespace": "synthetic.consumer.grid.v1", "draw_key": "w3", "strategy_id": "upgrade", "value": 5.0, "weight": 6.0}
            ]
        });
        let weighted = ValidatedSimulationOutcomeProfile::validate(
            &serde_json::from_value(weighted_payload).expect("relative input"),
        )
        .expect("validated relative weight profile");
        let weighted_evpi = simulation_outcome_evpi(&weighted).expect("relative evpi");
        assert!((weighted_evpi.evpi() - 6.0).abs() <= TOLERANCE);

        let broken_pairing_payload = json!({
            "profile_version": "voiage.simulation-outcome.v1",
            "provenance": {
                "producer_id": "synthetic-heor-consumer",
                "producer_version": "0.1.0",
                "model_revision": "heor-rev-1",
                "valuation_source": "synthetic-qaly-nmb",
                "artifact_digest": "4444444444444444444444444444444444444444444444444444444444444444"
            },
            "strategy_ids": ["standard_care", "intervention"],
            "value_declaration": {
                "unit": "NZD",
                "perspective": "health_system",
                "convention": "net_benefit",
                "objective": "maximize"
            },
            "weight_semantics": "equal",
            "draw_namespace": "synthetic.consumer.counterexample.v1",
            "outcomes": [
                {"draw_namespace": "synthetic.consumer.counterexample.v1", "draw_key": "d0", "strategy_id": "standard_care", "value": 10.0, "weight": 1.0},
                {"draw_namespace": "synthetic.consumer.counterexample.v1", "draw_key": "d0", "strategy_id": "intervention", "value": 10.0, "weight": 1.0},
                {"draw_namespace": "synthetic.consumer.counterexample.v1", "draw_key": "d1", "strategy_id": "standard_care", "value": 2.0, "weight": 1.0},
                {"draw_namespace": "synthetic.consumer.counterexample.v1", "draw_key": "d1", "strategy_id": "intervention", "value": 8.0, "weight": 1.0},
                {"draw_namespace": "synthetic.consumer.counterexample.v1", "draw_key": "d2", "strategy_id": "standard_care", "value": 6.0, "weight": 1.0},
                {"draw_namespace": "synthetic.consumer.counterexample.v1", "draw_key": "d2", "strategy_id": "intervention", "value": 6.0, "weight": 1.0},
                {"draw_namespace": "synthetic.consumer.counterexample.v1", "draw_key": "d3", "strategy_id": "standard_care", "value": 0.0, "weight": 1.0},
                {"draw_namespace": "synthetic.consumer.counterexample.v1", "draw_key": "d3", "strategy_id": "intervention", "value": 4.0, "weight": 1.0}
            ]
        });
        let counterexample = ValidatedSimulationOutcomeProfile::validate(
            &serde_json::from_value(broken_pairing_payload).expect("counterexample input"),
        )
        .expect("validated counterexample");
        assert!(simulation_outcome_evpi(&counterexample).expect("evpi").evpi().abs() <= TOLERANCE);
    }

    #[test]
    fn loss_minimize_is_rejected_without_implicit_sign_flip() {
        let payload = json!({
            "profile_version": "voiage.simulation-outcome.v1",
            "provenance": {
                "producer_id": "synthetic-loss-consumer",
                "producer_version": "0.1.0",
                "model_revision": "loss-rev-1",
                "valuation_source": "synthetic-loss",
                "artifact_digest": "5555555555555555555555555555555555555555555555555555555555555555"
            },
            "strategy_ids": ["a", "b"],
            "value_declaration": {
                "unit": "USD",
                "perspective": "payer",
                "convention": "loss",
                "objective": "minimize"
            },
            "weight_semantics": "equal",
            "draw_namespace": "synthetic.consumer.loss.v1",
            "outcomes": [
                {"draw_namespace": "synthetic.consumer.loss.v1", "draw_key": "d0", "strategy_id": "a", "value": 4.0, "weight": 1.0},
                {"draw_namespace": "synthetic.consumer.loss.v1", "draw_key": "d0", "strategy_id": "b", "value": 7.0, "weight": 1.0}
            ]
        });
        let validated = ValidatedSimulationOutcomeProfile::validate(
            &serde_json::from_value(payload).expect("loss input"),
        )
        .expect("domain valid loss profile");
        let err = simulation_outcome_evpi(&validated).expect_err("minimize rejected");
        assert_eq!(
            err,
            SimulationOutcomeAdapterError::UnsupportedObjectiveDirection {
                convention: ValueConvention::Loss,
                objective: ObjectiveDirection::Minimize,
            }
        );
    }
}
