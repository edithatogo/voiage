//! Analytical and generated controls for finite sequential sample information.

use std::num::NonZeroUsize;

use proptest::prelude::*;
use proptest::test_runner::{Config, RngSeed};
use voiage_domain::{BigRational, Identifier, SampleMatrix};
use voiage_numerics::{sequential_evsi, ObservationChannel, SequentialEvsi};

fn id(name: &str) -> Identifier {
    Identifier::new(name).unwrap()
}

fn matrix(rows: &[&[f64]]) -> SampleMatrix {
    rows.iter()
        .map(|row| row.to_vec())
        .collect::<Vec<_>>()
        .try_into()
        .unwrap()
}

fn ratios(numerator: i64, denominator: i64) -> BigRational {
    BigRational::new(numerator.into(), denominator.into())
}

fn channel(name: &str, rows: &[&[f64]]) -> ObservationChannel {
    let outcomes = (0..rows[0].len())
        .map(|index| id(&format!("outcome{index}")))
        .collect();
    ObservationChannel::from_weights(id(name), outcomes, &matrix(rows)).unwrap()
}

fn value(
    prior: &[f64],
    payoffs: &SampleMatrix,
    stages: &[Vec<ObservationChannel>],
) -> (BigRational, BigRational, BigRational) {
    let SequentialEvsi::Complete {
        current_value,
        expected_value,
        evsi,
        ..
    } = sequential_evsi(prior, payoffs, stages, NonZeroUsize::new(256).unwrap()).unwrap()
    else {
        panic!("analytic experiment should fit the branch budget");
    };
    (current_value, expected_value, evsi)
}

#[test]
fn one_stage_matches_analytic_values_before_and_after_garbling() {
    let prior = [1.0, 1.0];
    let payoffs = matrix(&[&[10.0, 0.0], &[0.0, 10.0]]);
    let original = channel("assay", &[&[3.0, 1.0], &[1.0, 3.0]]);
    assert_eq!(original.id(), &id("assay"));
    assert_eq!(original.outcomes(), &[id("outcome0"), id("outcome1")]);
    assert_eq!(
        value(&prior, &payoffs, &[vec![original.clone()]]),
        (ratios(5, 1), ratios(15, 2), ratios(5, 2))
    );

    let noisy = original
        .garble(
            vec![id("first"), id("second")],
            &matrix(&[&[3.0, 1.0], &[1.0, 3.0]]),
        )
        .unwrap();
    assert_eq!(
        noisy.likelihoods(),
        &[
            vec![ratios(5, 8), ratios(3, 8)],
            vec![ratios(3, 8), ratios(5, 8)]
        ]
    );
    assert_eq!(
        value(&prior, &payoffs, &[vec![noisy]]),
        (ratios(5, 1), ratios(25, 4), ratios(5, 4))
    );
    let pooled = original
        .garble(vec![id("pooled")], &matrix(&[&[1.0], &[1.0]]))
        .unwrap();
    assert_eq!(
        value(&prior, &payoffs, &[vec![pooled]]),
        (ratios(5, 1), ratios(5, 1), ratios(0, 1))
    );
}

#[test]
fn a_second_acquisition_action_can_depend_on_the_first_outcome() {
    let payoffs = matrix(&[
        &[1.0, 0.0, 0.0, 0.0],
        &[0.0, 1.0, 0.0, 0.0],
        &[0.0, 0.0, 1.0, 0.0],
        &[0.0, 0.0, 0.0, 1.0],
    ]);
    let grouping = channel(
        "group",
        &[&[1.0, 0.0], &[1.0, 0.0], &[0.0, 1.0], &[0.0, 1.0]],
    );
    let first_group = channel(
        "first_group",
        &[&[1.0, 0.0], &[0.0, 1.0], &[0.0, 1.0], &[0.0, 1.0]],
    );
    let second_group = channel(
        "second_group",
        &[&[0.0, 1.0], &[0.0, 1.0], &[1.0, 0.0], &[0.0, 1.0]],
    );
    let prior = [1.0; 4];
    assert_eq!(
        value(
            &prior,
            &payoffs,
            &[
                vec![grouping.clone()],
                vec![first_group.clone(), second_group.clone()]
            ]
        )
        .2,
        ratios(3, 4)
    );
    for fixed in [first_group.clone(), second_group.clone()] {
        assert_eq!(
            value(&prior, &payoffs, &[vec![grouping.clone()], vec![fixed]]).2,
            ratios(1, 2)
        );
    }
    let pooled = grouping
        .garble(vec![id("pooled")], &matrix(&[&[1.0], &[1.0]]))
        .unwrap();
    assert_eq!(
        value(
            &prior,
            &payoffs,
            &[vec![pooled], vec![first_group, second_group]]
        )
        .2,
        ratios(1, 4)
    );
}

#[test]
fn degenerate_prior_and_identical_channels_do_not_create_information() {
    let payoffs = matrix(&[&[-3.0, -5.0], &[-5.0, -2.0]]);
    let noninformative = channel("identical", &[&[4.0, 1.0, 0.0], &[8.0, 2.0, 0.0]]);
    let revealing = channel("reveal", &[&[1.0, 0.0], &[0.0, 1.0]]);
    assert_eq!(
        value(&[0.0, 1.0], &payoffs, &[vec![revealing]]),
        (ratios(-2, 1), ratios(-2, 1), ratios(0, 1))
    );
    let observed = value(
        &[1.0, 1.0],
        &payoffs,
        &[vec![noninformative.clone()], vec![noninformative]],
    );
    assert_eq!(observed, value(&[1.0, 1.0], &payoffs, &[]));
    assert_eq!(observed, (ratios(-7, 2), ratios(-7, 2), ratios(0, 1)));
}

#[test]
fn validates_weights_identities_shape_and_the_exact_budget_boundary() {
    let a = id("a");
    let b = id("b");
    let payoffs = matrix(&[&[1.0, 0.0], &[0.0, 1.0]]);
    let signal = channel("a", &[&[1.0, 0.0], &[0.0, 1.0]]);
    let stages = [vec![signal.clone()]];
    assert_eq!(
        sequential_evsi(&[1.0, 1.0], &payoffs, &stages, NonZeroUsize::MIN).unwrap(),
        SequentialEvsi::Indeterminate {
            current_value: ratios(1, 2),
            branches_examined: 1
        }
    );
    assert_eq!(
        sequential_evsi(
            &[1.0, 1.0],
            &payoffs,
            &stages,
            NonZeroUsize::new(2).unwrap()
        )
        .unwrap(),
        SequentialEvsi::Complete {
            current_value: ratios(1, 2),
            expected_value: ratios(1, 1),
            evsi: ratios(1, 2),
            branches_examined: 2
        }
    );
    assert!(sequential_evsi(&[1.0], &payoffs, &stages, NonZeroUsize::MIN).is_err());
    for prior in [[-1.0, 1.0], [0.0, 0.0], [f64::INFINITY, 1.0]] {
        assert!(sequential_evsi(&prior, &payoffs, &stages, NonZeroUsize::MIN).is_err());
    }
    let misplaced = channel("single_state", &[&[1.0]]);
    for stage in [
        vec![],
        vec![signal.clone(), signal.clone()],
        vec![misplaced],
    ] {
        assert!(sequential_evsi(&[1.0, 1.0], &payoffs, &[stage], NonZeroUsize::MIN).is_err());
    }
    assert!(
        ObservationChannel::from_weights(a.clone(), vec![a.clone(), a.clone()], &payoffs).is_err()
    );
    assert!(ObservationChannel::from_weights(a.clone(), vec![a.clone()], &payoffs).is_err());
    for invalid in [&[0.0, 0.0][..], &[-1.0, 2.0][..]] {
        assert!(ObservationChannel::from_weights(
            a.clone(),
            vec![a.clone(), b.clone()],
            &matrix(&[invalid, &[1.0, 0.0]])
        )
        .is_err());
    }
    assert!(signal.garble(vec![b], &matrix(&[&[1.0]])).is_err());
}

proptest! {
    #![proptest_config(Config { cases: 64, rng_seed: RngSeed::Fixed(20_261_122), ..Config::default() })]

    #[test]
    fn generated_garbling_preserves_nonnegativity_and_cannot_increase_evsi(
        weights in prop::array::uniform8(1_u16..=128), stakes in prop::array::uniform4(-30_i16..=50),
    ) {
        let weights = weights.map(f64::from);
        let stakes = stakes.map(f64::from);
        let payoffs = matrix(&[&stakes[..2], &stakes[2..]]);
        let prior = [weights[1], weights[7]];
        let original = channel("study", &[&weights[..2], &weights[2..4]]);
        let garbled = original.garble(vec![id("left"), id("right")], &matrix(&[&weights[4..6], &weights[6..]])).unwrap();
        let later = vec![original.clone()];
        let full = value(&prior, &payoffs, &[vec![original], later.clone()]).2;
        let reduced = value(&prior, &payoffs, &[vec![garbled], later]).2;
        prop_assert!(reduced >= ratios(0, 1));
        prop_assert!(reduced <= full);
    }
}
