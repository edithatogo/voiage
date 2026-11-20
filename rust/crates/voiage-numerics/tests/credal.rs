//! Independent analytical controls for exact credal certificates.

use std::collections::BTreeMap;
use std::num::NonZeroUsize;

use proptest::prelude::*;
use proptest::test_runner::{Config, RngSeed};
use voiage_domain::{
    AffineNetBenefit, BigRational, CredalInterval, CredalPoint, CredalPolytope, DominanceWitness,
    EAdmissibilityResult, Identifier, LinearInequality, SampleVector, StrategyId, Threshold,
};
use voiage_numerics::{
    solve_credal, verify_e_admissibility, verify_strict_credal_dominance, CredalAnalysis,
    CredalSolveResult,
};

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

fn ratio(numerator: i64, denominator: i64) -> BigRational {
    BigRational::new(numerator.into(), denominator.into())
}

fn solve(
    support: &CredalPolytope,
    strategies: &[AffineNetBenefit],
    budget: usize,
) -> CredalAnalysis {
    let CredalSolveResult::Feasible(result) =
        solve_credal(support, strategies, NonZeroUsize::new(budget).unwrap()).unwrap()
    else {
        panic!("expected a feasible polytope with certified ranges");
    };
    result
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

#[test]
fn generated_witnesses_distinguish_maximin_maximax_interval_and_mixed_dominance() {
    let support = support(
        2,
        vec![LinearInequality::new(row(&[1.0, 1.0]), 1.0).unwrap()],
    );
    let strategies = [
        affine("target", 0.5, &[0.0, 0.0]),
        affine("right", 1.0, &[-1.0, 0.0]),
        affine("left", 0.0, &[2.0, 0.0]),
        affine("inferior", -1.0, &[0.0, 0.0]),
    ];
    let result = solve(&support, &strategies, 1_000);
    assert_eq!(result.gamma_maximin, [strategy("target")]);
    assert_eq!(result.gamma_maximax, [strategy("left")]);
    assert_eq!(
        result.interval_undominated,
        [strategy("left"), strategy("right"), strategy("target")]
    );
    let endpoints = |name| {
        let range = &result.ranges[&strategy(name)];
        (&range.minimum.value, &range.maximum.value)
    };
    assert_eq!(endpoints("left"), (&ratio(0, 1), &ratio(2, 1)));
    assert_eq!(endpoints("right"), (&ratio(0, 1), &ratio(1, 1)));
    assert_eq!(endpoints("target"), (&ratio(1, 2), &ratio(1, 2)));
    for objective in &strategies {
        let range = &result.ranges[objective.objective().0];
        for extremum in [&range.minimum, &range.maximum] {
            assert!(support.contains_exact(&extremum.point).unwrap());
            assert_eq!(
                objective.evaluate_exact(&extremum.point).unwrap(),
                extremum.value
            );
        }
    }
    let [EAdmissibilityResult::Dominated(inferior), left, right, EAdmissibilityResult::Dominated(target)] =
        result.e_admissibility.as_slice()
    else {
        panic!("expected a mixed dominance certificate and two admissible alternatives");
    };
    for (entry, name) in [(left, "left"), (right, "right")] {
        let EAdmissibilityResult::Admissible {
            strategy: id,
            point,
        } = entry
        else {
            panic!("missing admissible strategy {name}");
        };
        assert_eq!(*id, strategy(name));
        assert!(verify_e_admissibility(&support, &strategies, id, point).unwrap());
    }
    assert_eq!(inferior.strategy(), &strategy("inferior"));
    assert_eq!(target.strategy(), &strategy("target"));
    let (weights, _, margin) = target.certificate();
    assert_eq!(
        weights,
        &BTreeMap::from([
            (strategy("left"), ratio(1, 3)),
            (strategy("right"), ratio(2, 3))
        ])
    );
    assert_eq!(*margin, ratio(1, 6));
    assert!(verify_strict_credal_dominance(
        &support,
        &strategies,
        &result.ranges[&strategy("target")].minimum.point,
        target
    )
    .unwrap());
    let mut reordered = strategies.clone();
    reordered.reverse();
    assert_eq!(result, solve(&support, &reordered, 1_000));
}

#[test]
fn e_admissibility_can_require_a_nonvertex_fractional_tie() {
    let support = support(
        3,
        vec![LinearInequality::new(row(&[0.0, 1.0, 1.0]), 1.0).unwrap()],
    );
    let strategies = [
        affine("target", 0.0, &[0.0; 3]),
        affine("up", -1.0, &[3.0, 0.0, 0.0]),
        affine("down", 1.0, &[-3.0, 0.0, 0.0]),
    ];
    let result = solve(&support, &strategies, 1_000);
    let EAdmissibilityResult::Admissible {
        strategy: id,
        point,
    } = &result.e_admissibility[1]
    else {
        panic!("the target is optimal at the exact shared boundary");
    };
    assert_eq!(*id, strategy("target"));
    assert_eq!(point.coordinates()[0], ratio(1, 3));
    assert!(verify_e_admissibility(&support, &strategies, id, point).unwrap());
    assert_eq!(result.gamma_maximin, [strategy("target")]);
    assert_eq!(result.gamma_maximax, [strategy("up")]);
}

#[test]
fn generated_dual_uses_binding_halfspace_and_preserves_weak_interval_ties() {
    let support = support(
        3,
        vec![
            LinearInequality::new(row(&[0.0, 0.0, 3.0]), 1.0).unwrap(),
            LinearInequality::new(row(&[0.0, 0.0, -3.0]), -1.0).unwrap(),
        ],
    );
    let strategies = [
        affine("a", 0.0, &[0.0, 0.0, 1.0]),
        affine("b", 0.5, &[0.0; 3]),
        affine("c", 0.0, &[0.0, 0.0, 1.5]),
    ];
    let result = solve(&support, &strategies, 1_000);
    assert_eq!(result.gamma_maximin, [strategy("b"), strategy("c")]);
    assert_eq!(result.gamma_maximax, result.gamma_maximin);
    assert_eq!(result.interval_undominated, result.gamma_maximin);
    assert_eq!(result.ranges[&strategy("a")].maximum.value, ratio(1, 3));
    let EAdmissibilityResult::Dominated(witness) = &result.e_admissibility[0] else {
        panic!("an affine strategy is strictly dominated on the constrained plane");
    };
    let (_, support_weights, margin) = witness.certificate();
    assert_eq!(*margin, ratio(1, 6));
    assert!(support_weights[..2]
        .iter()
        .any(|weight| weight > &ratio(0, 1)));
    assert!(matches!(
        &result.e_admissibility[1],
        EAdmissibilityResult::Admissible { .. }
    ));
    assert!(matches!(
        &result.e_admissibility[2],
        EAdmissibilityResult::Admissible { .. }
    ));
}

#[test]
fn budget_exhaustion_does_not_report_unproven_ranges_or_exclusions() {
    let support = support(2, vec![]);
    let strategies = [
        affine("left", 0.0, &[1.0, 0.0]),
        affine("right", 1.0, &[-1.0, 0.0]),
        affine("target", 0.25, &[0.0, 0.0]),
    ];
    assert_eq!(
        solve_credal(&support, &strategies, NonZeroUsize::MIN).unwrap(),
        CredalSolveResult::Indeterminate { bases_examined: 1 }
    );
    let partial = solve(&support, &strategies, 6);
    assert_eq!(partial.bases_examined, 6);
    assert_eq!(partial.gamma_maximin, [strategy("target")]);
    assert!(matches!(
        &partial.e_admissibility[0],
        EAdmissibilityResult::Admissible { .. }
    ));
    assert!(matches!(
        &partial.e_admissibility[1],
        EAdmissibilityResult::Admissible { .. }
    ));
    assert_eq!(
        partial.e_admissibility[2],
        EAdmissibilityResult::Indeterminate {
            strategy: strategy("target")
        }
    );
}

#[test]
fn pinned_and_empty_support_are_distinguished_and_inputs_are_validated() {
    let pinned = CredalPolytope::new(
        vec![Identifier::new("x").unwrap(), Identifier::new("y").unwrap()],
        vec![
            CredalInterval::new(0.25, 0.25).unwrap(),
            CredalInterval::new(-0.0, 0.0).unwrap(),
        ],
        vec![],
    )
    .unwrap();
    let strategies = [affine("a", 0.0, &[1.0, 0.0]), affine("b", 0.5, &[0.0; 2])];
    let result = solve(&pinned, &strategies, 1);
    assert_eq!(result.bases_examined, 0);
    assert_eq!(
        result.ranges[&strategy("a")].minimum,
        result.ranges[&strategy("a")].maximum
    );
    assert_eq!(result.gamma_maximin, [strategy("b")]);
    let EAdmissibilityResult::Dominated(witness) = &result.e_admissibility[0] else {
        panic!("a strict comparison at the pinned point must have a certificate");
    };
    assert_eq!(*witness.certificate().2, ratio(1, 4));
    let empty = support(
        2,
        vec![LinearInequality::new(row(&[1.0, 1.0]), -1.0).unwrap()],
    );
    assert_eq!(
        solve_credal(&empty, &strategies, NonZeroUsize::new(10).unwrap()).unwrap(),
        CredalSolveResult::Infeasible { bases_examined: 10 }
    );
    assert!(solve_credal(&pinned, &[], NonZeroUsize::MIN).is_err());
    assert!(solve_credal(
        &pinned,
        &[strategies[0].clone(), strategies[0].clone()],
        NonZeroUsize::MIN
    )
    .is_err());
    assert!(solve_credal(&pinned, &[affine("wrong", 0.0, &[1.0])], NonZeroUsize::MIN).is_err());
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

    #[test]
    fn generated_crossing_lines_certify_the_analytic_optimal_mixture(
        left in 1_u16..=32, right in 1_u16..=32,
    ) {
        let support = support(2, vec![]);
        let strategies = [
            affine("left", 0.0, &[f64::from(left), 0.0]),
            affine("right", f64::from(right), &[-f64::from(right), 0.0]),
            affine("target", 0.0, &[0.0; 2]),
        ];
        let result = solve(&support, &strategies, 256);
        let EAdmissibilityResult::Dominated(witness) = &result.e_admissibility[2] else {
            return Err(TestCaseError::fail("no dominance certificate for crossing lines"));
        };
        let (left, right) = (i64::from(left), i64::from(right));
        let (weights, support_weights, margin) = witness.certificate();
        prop_assert_eq!(&weights[&strategy("left")], &ratio(right, left + right));
        prop_assert_eq!(&weights[&strategy("right")], &ratio(left, left + right));
        prop_assert_eq!(margin, &ratio(left * right, left + right));
        prop_assert_eq!(support_weights, &vec![ratio(0, 1); 4]);
    }
}
