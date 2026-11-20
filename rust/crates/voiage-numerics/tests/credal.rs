//! Independent analytical controls for exact credal certificates.

use std::collections::BTreeMap;

use proptest::prelude::*;
use proptest::test_runner::{Config, RngSeed};
use voiage_domain::{
    AffineNetBenefit, BigRational, CredalInterval, CredalPoint, CredalPolytope, DominanceWitness,
    Identifier, LinearInequality, SampleVector, StrategyId, Threshold,
};
use voiage_numerics::{verify_e_admissibility, verify_strict_credal_dominance};

fn strategy(name: &str) -> StrategyId {
    StrategyId::new(name).unwrap()
}

fn row(values: &[f64]) -> SampleVector {
    SampleVector::try_from(values.to_vec()).unwrap()
}

fn support(dimension: usize, inequalities: Vec<LinearInequality>) -> CredalPolytope {
    let parameters = (0..dimension)
        .map(|index| Identifier::new(format!("x{index}")).unwrap())
        .collect();
    CredalPolytope::new(
        parameters,
        vec![CredalInterval::new(0.0, 1.0).unwrap(); dimension],
        inequalities,
    )
    .unwrap()
}

fn affine(name: &str, intercept: f64, slopes: &[f64]) -> AffineNetBenefit {
    AffineNetBenefit::new(strategy(name), intercept, row(slopes)).unwrap()
}

fn mixture(support: &CredalPolytope, scale: f64, margin: f64) -> DominanceWitness {
    DominanceWitness::new(
        support,
        strategy("target"),
        BTreeMap::from([(strategy("left"), scale), (strategy("right"), scale)]),
        vec![0.0; 5],
        Threshold::new(margin).unwrap(),
    )
    .unwrap()
}

#[test]
fn a_mixed_alternative_proves_dominance_when_neither_pure_alternative_does() {
    let support = support(
        2,
        vec![LinearInequality::new(row(&[1.0, 1.0]), 1.0).unwrap()],
    );
    let strategies = [
        affine("target", 0.25, &[0.0, 0.0]),
        affine("left", 0.0, &[1.0, 0.0]),
        affine("right", 1.0, &[-1.0, 0.0]),
    ];
    // x and (1 - x) each attain zero. Their even mixture has value 1/2.
    let origin = CredalPoint::from_binary64(&[0.0, 0.0]).unwrap();
    assert!(verify_strict_credal_dominance(
        &support,
        &strategies,
        &origin,
        &mixture(&support, 1.0, 0.25)
    )
    .unwrap());
    assert!(!verify_strict_credal_dominance(
        &support,
        &strategies,
        &origin,
        &mixture(&support, 1.0, 0.5)
    )
    .unwrap());
    assert!(!verify_e_admissibility(&support, &strategies, &strategy("target"), &origin).unwrap());
    let outside = CredalPoint::from_binary64(&[1.0, 1.0]).unwrap();
    assert!(!verify_strict_credal_dominance(
        &support,
        &strategies,
        &outside,
        &mixture(&support, 1.0, 0.25)
    )
    .unwrap());
}

#[test]
fn three_dimensional_halfspace_multipliers_certify_only_the_declared_support() {
    let cap = |upper| vec![LinearInequality::new(row(&[0.0, 0.0, 1.0]), upper).unwrap()];
    let bounded = support(3, cap(0.5));
    let strategies = [
        affine("target", 0.0, &[0.0, 0.0, 1.0]),
        affine("alternative", 0.75, &[0.0, 0.0, 0.0]),
    ];
    let witness = DominanceWitness::new(
        &bounded,
        strategy("target"),
        BTreeMap::from([(strategy("alternative"), 1.0)]),
        vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        Threshold::new(0.25).unwrap(),
    )
    .unwrap();
    let origin = CredalPoint::from_binary64(&[0.0; 3]).unwrap();
    assert!(verify_strict_credal_dominance(&bounded, &strategies, &origin, &witness).unwrap());
    let wider = support(3, cap(0.75));
    assert!(!verify_strict_credal_dominance(&wider, &strategies, &origin, &witness).unwrap());
    let empty = support(3, cap(-0.25));
    assert!(!verify_strict_credal_dominance(&empty, &strategies, &origin, &witness).unwrap());
}

#[test]
fn box_residual_uses_the_correct_endpoint_and_multiplier_order() {
    let box_support = CredalPolytope::new(
        vec![Identifier::new("x").unwrap(), Identifier::new("y").unwrap()],
        vec![
            CredalInterval::new(-1.0, 1.0).unwrap(),
            CredalInterval::new(0.0, 0.5).unwrap(),
        ],
        vec![],
    )
    .unwrap();
    let strategies = [
        affine("target", 0.0, &[0.0, 0.0]),
        affine("alternative", 2.0, &[1.0, -1.0]),
    ];
    let origin = CredalPoint::from_binary64(&[0.0; 2]).unwrap();
    // The minimum of 2 + x - y is 1/2 at x = -1 and y = 1/2.
    for weights in [vec![0.0; 4], vec![1.0, 0.0, 0.0, 1.0]] {
        let certificate = |margin| {
            DominanceWitness::new(
                &box_support,
                strategy("target"),
                BTreeMap::from([(strategy("alternative"), 1.0)]),
                weights.clone(),
                Threshold::new(margin).unwrap(),
            )
            .unwrap()
        };
        assert!(verify_strict_credal_dominance(
            &box_support,
            &strategies,
            &origin,
            &certificate(0.5)
        )
        .unwrap());
        assert!(!verify_strict_credal_dominance(
            &box_support,
            &strategies,
            &origin,
            &certificate(0.75)
        )
        .unwrap());
    }
}

#[test]
fn a_fractional_boundary_tie_verifies_e_admissibility_without_float_rounding() {
    let support = support(
        3,
        vec![
            LinearInequality::new(row(&[3.0, 0.0, 0.0]), 1.0).unwrap(),
            LinearInequality::new(row(&[-3.0, 0.0, 0.0]), -1.0).unwrap(),
        ],
    );
    let strategies = [
        affine("target", -1.0, &[3.0, 0.0, 0.0]),
        affine("alternative", 0.0, &[0.0, 0.0, 0.0]),
    ];
    let zero = BigRational::from_integer(0.into());
    let witness = CredalPoint::new(vec![
        BigRational::new(1.into(), 3.into()),
        zero.clone(),
        zero,
    ])
    .unwrap();
    assert!(verify_e_admissibility(&support, &strategies, &strategy("target"), &witness).unwrap());
    let rounded = CredalPoint::from_binary64(&[1.0 / 3.0, 0.0, 0.0]).unwrap();
    assert!(!verify_e_admissibility(&support, &strategies, &strategy("target"), &rounded).unwrap());
    assert!(verify_e_admissibility(&support, &strategies, &strategy("unknown"), &witness).is_err());
}

proptest! {
    #![proptest_config(Config {
        cases: 64,
        rng_seed: RngSeed::Fixed(20_261_121),
        ..Config::default()
    })]

    #[test]
    fn exact_normalisation_preserves_claims_across_positive_rescaling(scale in 1_u16..=512) {
        let support = support(
            2,
            vec![LinearInequality::new(row(&[1.0, 1.0]), 1.0).unwrap()],
        );
        let origin = CredalPoint::from_binary64(&[0.0; 2]).unwrap();
        let mut strategies = [
            affine("target", 0.25, &[0.0, 0.0]),
            affine("left", 0.0, &[1.0, 0.0]),
            affine("right", 1.0, &[-1.0, 0.0]),
        ];
        let witness = mixture(&support, f64::from(scale), 0.25);
        prop_assert!(verify_strict_credal_dominance(&support, &strategies, &origin, &witness).unwrap());
        strategies[0] = affine("target", 0.5, &[0.0, 0.0]);
        prop_assert!(!verify_strict_credal_dominance(&support, &strategies, &origin, &witness).unwrap());
    }
}
