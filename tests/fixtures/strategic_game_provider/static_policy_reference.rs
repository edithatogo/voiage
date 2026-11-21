//! Standalone fixture producer for the SI03 static comparison.

use std::collections::BTreeMap;

use paignia_core::{
    ActionEdge, ActionId, CatalogueClaim, ChanceEdge, Game, GameInstanceId, GameSpecification,
    GameSpecificationId, Node, NodeId, PlayerId, ResultAlgorithm, ResultGameClass,
    ResultIdentityRecord, ResultInformationStructure, ResultSelectionRule, ResultSolutionConcept,
    ResultStatus, SolverRunId, GAME_SPEC_SCHEMA_VERSION, RESULT_IDENTITY_SCHEMA_VERSION,
    STATIC_GAME_PROFILE_V1,
};
use paignia_solve::{backward_induction_result, SolverConfig};
use serde_json::json;

const CASES: [(&str, &str, f64, f64); 2] = [
    ("favourable", "clinical_positive", 2.0, 4.0),
    ("adverse", "clinical_negative", -4.0, -8.0),
];
const SPECIFICATION: &str = "urn:ndrg:si03:finite-observation-policy-v1";
const INSTANCE: &str = "urn:ndrg:si03:synthetic-instance";
const RUN: &str = "urn:ndrg:si03:qualified-archive-run";

fn id(value: &str) -> NodeId {
    NodeId(value.into())
}

fn graph() -> GameSpecification {
    let mut nodes = BTreeMap::new();
    let institution = PlayerId("institution".into());
    let public = PlayerId("public".into());
    let mut states = Vec::new();
    for (world, _, actor_value, public_value) in CASES {
        let retain = id(&format!("{world}_maintain"));
        let reimburse = id(&format!("{world}_reimburse"));
        nodes.insert(
            retain.clone(),
            Node::Terminal {
                payoffs: BTreeMap::new(),
            },
        );
        nodes.insert(
            reimburse.clone(),
            Node::Terminal {
                payoffs: BTreeMap::from([
                    (institution.clone(), actor_value),
                    (public.clone(), public_value),
                ]),
            },
        );
        nodes.insert(
            id(world),
            Node::Decision {
                player: institution.clone(),
                edges: vec![
                    ActionEdge {
                        action: ActionId("maintain".into()),
                        target: retain,
                    },
                    ActionEdge {
                        action: ActionId("reimburse".into()),
                        target: reimburse,
                    },
                ],
            },
        );
        states.push(ChanceEdge {
            probability: 0.5,
            target: id(world),
        });
    }
    nodes.insert(id("signal"), Node::Chance { edges: states });
    GameSpecification {
        schema_version: GAME_SPEC_SCHEMA_VERSION.into(),
        id: SPECIFICATION.into(),
        game: Game {
            root: id("signal"),
            nodes,
        },
    }
}

fn main() {
    let game_digest = std::env::args()
        .nth(1)
        .expect("supply the fixture source SHA-256");
    let instance = GameInstanceId(INSTANCE.into());
    let computed = backward_induction_result(
        &graph(),
        STATIC_GAME_PROFILE_V1,
        ResultIdentityRecord {
            schema_version: RESULT_IDENTITY_SCHEMA_VERSION.into(),
            specification_id: GameSpecificationId(SPECIFICATION.into()),
            game_instance_id: instance.clone(),
            solver_run_id: SolverRunId(RUN.into()),
            session_id: None,
            policy_artifact_id: None,
            solver_trace_id: None,
        },
        SolverConfig::default(),
    )
    .expect("execute the qualified native provider");
    let native = &computed.envelope;
    native.validate_against(&graph(), &instance).unwrap();
    assert_eq!(native.status, ResultStatus::Completed);
    assert_eq!(native.catalogue, CatalogueClaim::OneFound);
    assert_eq!(native.game_class, ResultGameClass::FiniteAcyclicExtensive);
    assert_eq!(
        native.information_structure,
        ResultInformationStructure::PerfectInformation
    );
    assert_eq!(
        native.solution_concept,
        ResultSolutionConcept::ToleranceAwareBackwardInductionPolicy
    );
    assert_eq!(
        native.algorithm.algorithm,
        ResultAlgorithm::BackwardInduction
    );
    assert_eq!(
        native.selection_rule,
        ResultSelectionRule::InputOrderPairwiseToleranceThenLexicographic
    );
    assert_eq!(native.solutions.len(), 1);

    let solution = &native.solutions[0];
    assert_eq!(solution.choices[&id("favourable")].0, "reimburse");
    assert_eq!(solution.choices[&id("adverse")].0, "maintain");
    assert_eq!(
        solution.expected_payoffs[&PlayerId("institution".into())].to_bits(),
        1.0_f64.to_bits()
    );
    assert_eq!(
        solution.expected_payoffs[&PlayerId("public".into())].to_bits(),
        2.0_f64.to_bits()
    );
    let choices = CASES.map(|(world, observation, _, _)| {
        json!({
            "world_id": world,
            "observation": observation,
            "native_node_id": world,
            "action_id": solution.choices[&id(world)].0,
        })
    });
    let projection = json!({
        "projection_version": "1.0.0",
        "provider_revision": "bd880dd756d2ba56cc5f479bc078cea31b53c1d1",
        "package_receipt_sha256": "4dedd2b14aad9ffc12e0975202c82091ff678a372bd9312bafff2b97be4e7c01",
        "game_sha256": game_digest,
        "native_player_id": "institution",
        "analysis_id": "si03-static-observation",
        "design_id": "qualified_response",
        "policy_id": "native_policy",
        "native": {
            "schema_version": native.schema_version,
            "semantic_profile": native.semantic_profile,
            "game_class": "finite_acyclic_extensive",
            "information_structure": "perfect_information",
            "solution_concept": "tolerance_aware_backward_induction_policy",
            "algorithm": "paignia/backward-induction",
            "algorithm_version": native.algorithm.version,
            "status": "completed",
            "selection_rule": "input_order_pairwise_tolerance_then_lexicographic",
            "catalogue": "one_found",
            "tolerance": native.tolerance,
            "identities": {
                "specification_id": native.identities.specification_id.0,
                "game_instance_id": native.identities.game_instance_id.0,
                "solver_run_id": native.identities.solver_run_id.0,
                "session_id": null,
                "policy_artifact_id": null,
                "solver_trace_id": null,
            },
        },
        "choices": choices,
    });
    println!("{}", serde_json::to_string_pretty(&projection).unwrap());
}
