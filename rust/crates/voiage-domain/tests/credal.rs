//! Admission and analytical point controls for native credal contracts.

use std::collections::BTreeMap;

use voiage_domain::{
    AffineNetBenefit, CredalContractError, CredalInterval, CredalPolytope, DominanceWitness,
    EAdmissibilityResult, Identifier, LinearInequality, SampleVector, StrategyId, Threshold,
};

fn id(value: &str) -> Identifier {
    Identifier::new(value).unwrap()
}

fn strategy(value: &str) -> StrategyId {
    StrategyId::new(value).unwrap()
}

fn interval(lower: f64, upper: f64) -> CredalInterval {
    CredalInterval::new(lower, upper).unwrap()
}

fn row(values: &[f64]) -> SampleVector {
    SampleVector::try_from(values.to_vec()).unwrap()
}

fn two_parameters() -> Vec<Identifier> {
    vec![id("first"), id("second")]
}

#[test]
fn interval_and_halfspace_membership_respect_order_and_boundaries() {
    let polytope = CredalPolytope::new(
        two_parameters(),
        vec![interval(0.0, 1.0), interval(0.0, 2.0)],
        vec![LinearInequality::new(row(&[2.0, 1.0]), 2.0).unwrap()],
    )
    .unwrap();
    assert!(polytope.contains(&[0.75, 0.5]).unwrap());
    assert!(polytope.contains(&[0.0, 2.0]).unwrap());
    assert!(!polytope.contains(&[0.5, 1.5]).unwrap());
    assert!(!polytope.contains(&[-0.25, 1.0]).unwrap());
    assert_eq!(
        polytope.contains(&[0.0]),
        Err(CredalContractError::DimensionMismatch {
            expected: 2,
            actual: 1,
        })
    );
    assert_eq!(
        polytope.contains(&[0.0, f64::NAN]),
        Err(CredalContractError::NonFinite)
    );
}

#[test]
fn pinned_intervals_return_the_direct_signed_net_benefits_with_ties() {
    let polytope = CredalPolytope::new(
        two_parameters(),
        vec![interval(-0.0, 0.0), interval(2.0, 2.0)],
        vec![LinearInequality::new(row(&[0.0, 1.0]), 2.0).unwrap()],
    )
    .unwrap();
    let objectives = [
        AffineNetBenefit::new(strategy("A"), 1.0, row(&[3.0, -2.0])).unwrap(),
        AffineNetBenefit::new(strategy("B"), -4.0, row(&[7.0, 0.5])).unwrap(),
        AffineNetBenefit::new(strategy("C"), 1.0, row(&[3.0, 1.5])).unwrap(),
    ];
    let expected = BTreeMap::from([
        (strategy("A"), -3.0),
        (strategy("B"), -3.0),
        (strategy("C"), 4.0),
    ]);
    assert_eq!(
        polytope.point_expected_net_benefits(&objectives).unwrap(),
        Some(expected)
    );
    let interval_set = CredalPolytope::new(
        two_parameters(),
        vec![interval(0.0, 0.0), interval(1.0, 2.0)],
        vec![],
    )
    .unwrap();
    assert_eq!(
        interval_set
            .point_expected_net_benefits(&objectives)
            .unwrap(),
        None
    );
}

#[test]
fn exact_constraints_reject_an_excluded_fixed_point_after_cancellation() {
    let parameters = vec![id("first"), id("second"), id("third")];
    let inequality = LinearInequality::new(row(&[1e16, 1.0, -1e16]), 0.0).unwrap();
    let box_support = CredalPolytope::new(
        parameters.clone(),
        vec![interval(0.0, 1.0); 3],
        vec![inequality.clone()],
    )
    .unwrap();
    assert_eq!(box_support.contains(&[1.0; 3]), Ok(false));
    assert_eq!(
        CredalPolytope::new(parameters, vec![interval(1.0, 1.0); 3], vec![inequality]),
        Err(CredalContractError::InfeasibleSingleton)
    );
}

#[test]
fn affine_net_benefit_preserves_small_terms_between_large_opposite_terms() {
    let value = AffineNetBenefit::new(strategy("A"), -0.25, row(&[1e16, 1.0, -1e16])).unwrap();
    assert_eq!(value.evaluate(&[1.0; 3]), Ok(0.75));
    let cancelling = AffineNetBenefit::new(
        strategy("B"),
        f64::from_bits(1),
        row(&[f64::MAX, -f64::MAX]),
    )
    .unwrap();
    assert_eq!(cancelling.evaluate(&[2.0, 2.0]), Ok(f64::from_bits(1)));
}

#[test]
fn rejects_inconsistent_contracts_without_inventing_point_results() {
    assert_eq!(
        CredalInterval::new(1.0, 0.0),
        Err(CredalContractError::ReversedInterval)
    );
    assert_eq!(
        CredalInterval::new(0.0, f64::INFINITY),
        Err(CredalContractError::NonFinite)
    );
    assert_eq!(
        CredalPolytope::new(vec![id("x"), id("x")], vec![], vec![]),
        Err(CredalContractError::DuplicateIdentifier)
    );
    let pinned = vec![interval(1.0, 1.0), interval(2.0, 2.0)];
    assert_eq!(
        CredalPolytope::new(
            two_parameters(),
            pinned.clone(),
            vec![LinearInequality::new(row(&[1.0]), 2.0).unwrap()],
        ),
        Err(CredalContractError::DimensionMismatch {
            expected: 2,
            actual: 1,
        })
    );
    assert_eq!(
        CredalPolytope::new(
            two_parameters(),
            pinned.clone(),
            vec![LinearInequality::new(row(&[1.0, 1.0]), 2.0).unwrap()],
        ),
        Err(CredalContractError::InfeasibleSingleton)
    );
    let polytope = CredalPolytope::new(two_parameters(), pinned, vec![]).unwrap();
    let overflow = AffineNetBenefit::new(strategy("A"), 0.0, row(&[0.0, f64::MAX])).unwrap();
    assert_eq!(
        polytope.point_expected_net_benefits(std::slice::from_ref(&overflow)),
        Err(CredalContractError::NonFinite)
    );
    assert_eq!(
        polytope.point_expected_net_benefits(&[overflow.clone(), overflow]),
        Err(CredalContractError::DuplicateIdentifier)
    );
}

#[test]
fn dominance_certificate_data_preserves_order_and_rejects_invalid_multipliers() {
    let support = CredalPolytope::new(
        vec![id("x")],
        vec![interval(0.0, 1.0)],
        vec![LinearInequality::new(row(&[1.0]), 0.75).unwrap()],
    )
    .unwrap();
    let margin = Threshold::new(0.25).unwrap();
    let alternatives = BTreeMap::from([(strategy("B"), 0.25), (strategy("C"), 0.75)]);
    let witness = DominanceWitness::new(
        &support,
        strategy("A"),
        alternatives.clone(),
        vec![0.5, 0.0, 0.0],
        margin,
    )
    .unwrap();
    assert_eq!(witness.strategy(), &strategy("A"));
    assert_eq!(
        witness.certificate(),
        (&alternatives, &[0.5, 0.0, 0.0][..], margin)
    );
    assert!(matches!(
        EAdmissibilityResult::Dominated(witness),
        EAdmissibilityResult::Dominated(value) if value.strategy() == &strategy("A")
    ));
    assert_eq!(
        DominanceWitness::new(
            &support,
            strategy("A"),
            alternatives.clone(),
            vec![0.0],
            margin
        ),
        Err(CredalContractError::DimensionMismatch {
            expected: 3,
            actual: 1
        })
    );
    for invalid in [f64::NAN, -1.0] {
        assert_eq!(
            DominanceWitness::new(
                &support,
                strategy("A"),
                alternatives.clone(),
                vec![invalid, 0.0, 0.0],
                margin,
            ),
            Err(CredalContractError::InvalidMultipliers)
        );
    }
    assert_eq!(
        DominanceWitness::new(&support, strategy("B"), alternatives, vec![0.0; 3], margin,),
        Err(CredalContractError::InvalidMultipliers)
    );
}
