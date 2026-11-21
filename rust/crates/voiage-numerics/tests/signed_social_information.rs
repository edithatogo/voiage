//! Independent finite signed-information controls for SI-V01 and SI-V05.

use voiage_domain::SampleMatrix;
use voiage_numerics::{
    evaluate_signed_social_information, SignedInformationDesign, SignedInformationInput,
    SignedInformationSelector, SignedWelfareStage,
};

fn matrix(rows: &[&[f64]]) -> SampleMatrix {
    rows.iter()
        .map(|row| row.to_vec())
        .collect::<Vec<_>>()
        .try_into()
        .unwrap()
}

fn design(policies: &[usize], comparator: usize) -> SignedInformationDesign<'_> {
    SignedInformationDesign {
        policy_indices: policies,
        comparator,
        selected_policy: None,
        selector: SignedInformationSelector::SocialWelfare,
        transfers: &[],
        costs: &[],
    }
}

#[test]
fn perfect_information_retains_the_declared_nonuniform_prior() {
    // Perfect information pays 100 in either world. The best fixed action
    // pays 50 at equal mass and 90 when the first world has nine tenths.
    let payoffs = [
        matrix(&[&[0.0], &[100.0]]),
        matrix(&[&[100.0], &[0.0]]),
        matrix(&[&[100.0], &[100.0]]),
    ];
    let designs = [design(&[0, 1], 0), design(&[0, 1, 2], 0)];
    for (probabilities, fixed_policy, expected) in
        [([0.5, 0.5], 0, 50.0_f64), ([0.9, 0.1], 1, 10.0)]
    {
        let input = SignedInformationInput {
            world_probabilities: &probabilities,
            policy_payoffs: &payoffs,
            welfare_weights: &[1.0],
            role_agents: &[vec![0]],
            welfare_stage: SignedWelfareStage::PostTransfer,
            designs: &designs,
        };
        let output = evaluate_signed_social_information(&input).unwrap();
        assert_eq!(output[0].selected_policy, fixed_policy);
        assert_eq!(output[1].selected_policy, 2);
        assert_eq!(output[1].signed_by_agent, vec![expected]);
        assert_eq!(output[1].signed_by_role, vec![expected]);
        assert_eq!(output[1].signed_welfare.to_bits(), expected.to_bits());
    }
}

#[test]
fn private_harm_and_social_gain_survive_transfers_and_resource_costs() {
    let payoffs = [
        matrix(&[&[0.0, 0.0, 0.0], &[0.0, 0.0, 0.0]]),
        matrix(&[&[2.0, 0.0, 6.0], &[-6.0, 0.0, 2.0]]),
    ];
    let designs = [
        design(&[0], 0),
        design(&[0, 1], 0),
        SignedInformationDesign {
            selected_policy: Some(1),
            selector: SignedInformationSelector::Agent(0),
            transfers: &[(2, 0, 1.0)],
            costs: &[(0, 0.5), (1, 0.25)],
            ..design(&[0, 1], 0)
        },
        SignedInformationDesign {
            selected_policy: Some(0),
            ..design(&[0, 1], 1)
        },
    ];
    let roles = [vec![0], vec![1], vec![2], vec![]];
    let input = SignedInformationInput {
        world_probabilities: &[0.5, 0.5],
        policy_payoffs: &payoffs,
        welfare_weights: &[1.0, 0.0, 1.0],
        role_agents: &roles,
        welfare_stage: SignedWelfareStage::PostTransfer,
        designs: &designs,
    };
    let output = evaluate_signed_social_information(&input).unwrap();
    assert_eq!(output[1].signed_by_agent, vec![-2.0, 0.0, 4.0]);
    assert_eq!(output[1].signed_welfare.to_bits(), 2.0_f64.to_bits());

    let declared = &output[2];
    assert_eq!(declared.selector_values, vec![0.5, -1.5]);
    assert_eq!(declared.selected_policy, 1);
    assert_eq!(declared.ledger.transfer, vec![1.0, 0.0, -1.0]);
    assert_eq!(declared.ledger.cost, vec![0.5, 0.25, 0.0]);
    assert_eq!(declared.signed_by_agent, vec![-1.5, -0.25, 3.0]);
    assert_eq!(declared.signed_by_role, vec![-1.5, -0.25, 3.0, 0.0]);
    assert_eq!(declared.signed_welfare.to_bits(), 1.5_f64.to_bits());
    assert_eq!(output[3].selected_policy, 0);
    assert_eq!(output[3].signed_welfare.to_bits(), (-2.0_f64).to_bits());

    let pre_transfer = evaluate_signed_social_information(&SignedInformationInput {
        welfare_stage: SignedWelfareStage::PreTransfer,
        ..input
    })
    .unwrap();
    assert_eq!(pre_transfer[2].signed_welfare.to_bits(), 2.0_f64.to_bits());
    assert_eq!(pre_transfer[2].signed_by_agent, declared.signed_by_agent);
}

#[test]
fn finite_cancellation_retains_the_small_payoff() {
    let payoffs = [matrix(&[&[4.0e20], &[2.0], &[-4.0e20]])];
    let designs = [design(&[0], 0)];
    let result = evaluate_signed_social_information(&SignedInformationInput {
        world_probabilities: &[0.25, 0.5, 0.25],
        policy_payoffs: &payoffs,
        welfare_weights: &[1.0],
        role_agents: &[],
        welfare_stage: SignedWelfareStage::PostTransfer,
        designs: &designs,
    })
    .unwrap();
    assert_eq!(result[0].ledger.pre_transfer, vec![1.0]);
}

#[test]
fn invalid_world_law_catalog_and_result_fail_distinctly() {
    let payoffs = [
        matrix(&[&[0.0], &[0.0]]),
        matrix(&[&[f64::MAX], &[f64::MAX]]),
    ];
    let designs = [design(&[0], 0)];
    let input = SignedInformationInput {
        world_probabilities: &[0.5, 0.5],
        policy_payoffs: &payoffs,
        welfare_weights: &[1.0],
        role_agents: &[],
        welfare_stage: SignedWelfareStage::PostTransfer,
        designs: &designs,
    };
    let missing_mass = evaluate_signed_social_information(&SignedInformationInput {
        world_probabilities: &[0.5, 0.4],
        ..input
    })
    .unwrap_err();
    assert!(missing_mass.to_string().contains("sum to one"));

    let unadmitted_policy = evaluate_signed_social_information(&SignedInformationInput {
        designs: &[SignedInformationDesign {
            selected_policy: Some(1),
            ..design(&[0], 0)
        }],
        ..input
    })
    .unwrap_err();
    assert!(unadmitted_policy.to_string().contains("catalog"));

    let overflow = evaluate_signed_social_information(&SignedInformationInput {
        welfare_weights: &[2.0],
        designs: &[design(&[1], 0)],
        ..input
    })
    .unwrap_err();
    assert!(overflow.to_string().contains("finite"));
}

#[test]
fn admitted_tables_require_aligned_dimensions_and_declared_accounts() {
    let payoffs = [matrix(&[&[1.0, 2.0]])];
    let designs = [design(&[0], 0)];
    let input = SignedInformationInput {
        world_probabilities: &[1.0],
        policy_payoffs: &payoffs,
        welfare_weights: &[1.0, 1.0],
        role_agents: &[vec![0, 1]],
        welfare_stage: SignedWelfareStage::PostTransfer,
        designs: &designs,
    };
    let wrong_rows = evaluate_signed_social_information(&SignedInformationInput {
        world_probabilities: &[0.5, 0.5],
        ..input
    })
    .unwrap_err();
    assert!(wrong_rows.to_string().contains("rows must match"));
    let wrong_columns = evaluate_signed_social_information(&SignedInformationInput {
        welfare_weights: &[1.0],
        ..input
    })
    .unwrap_err();
    assert!(wrong_columns.to_string().contains("columns must match"));

    for (invalid, message) in [
        (design(&[0], 1), "Comparator"),
        (design(&[0, 0], 0), "distinct and in range"),
        (
            SignedInformationDesign {
                transfers: &[(0, 0, 1.0)],
                ..designs[0]
            },
            "Transfers need distinct declared agents",
        ),
        (
            SignedInformationDesign {
                costs: &[(1, -1.0)],
                ..designs[0]
            },
            "Costs need a declared agent",
        ),
    ] {
        let result = evaluate_signed_social_information(&SignedInformationInput {
            designs: &[invalid],
            ..input
        });
        assert!(result.unwrap_err().to_string().contains(message));
    }
}
