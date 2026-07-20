//! Property contracts for dominance classification and frontier invariants.

use proptest::prelude::*;
use voiage_domain::SampleVector;
use voiage_numerics::dominance;

fn vector(values: Vec<f64>) -> SampleVector {
    values.try_into().expect("generated vector is valid")
}

proptest! {
    #[test]
    fn classifications_are_disjoint_and_exhaustive(
        pairs in prop::collection::vec((-1_000_i16..=1_000, -1_000_i16..=1_000), 2..=32),
    ) {
        let costs = pairs.iter().map(|pair| f64::from(pair.0)).collect::<Vec<_>>();
        let effects = pairs.iter().map(|pair| f64::from(pair.1)).collect::<Vec<_>>();
        let result = dominance(&vector(costs), &vector(effects)).expect("dominance result");
        let count = result.frontier_indices.len()
            + result.strongly_dominated_indices.len()
            + result.extended_dominated_indices.len();

        prop_assert_eq!(count, pairs.len());
        for index in 0..pairs.len() {
            let memberships = usize::from(result.frontier_indices.contains(&index))
                + usize::from(result.strongly_dominated_indices.contains(&index))
                + usize::from(result.extended_dominated_indices.contains(&index));
            prop_assert_eq!(memberships, 1);
        }
        prop_assert!(result.icers.windows(2).all(|pair| pair[0] < pair[1]));
    }

    #[test]
    fn translating_all_costs_preserves_classification_and_icers(
        pairs in prop::collection::vec((-100_i16..=100, -100_i16..=100), 2..=16),
        translation in -100_i16..=100,
    ) {
        let costs = pairs.iter().map(|pair| f64::from(pair.0)).collect::<Vec<_>>();
        let effects = pairs.iter().map(|pair| f64::from(pair.1)).collect::<Vec<_>>();
        let translated = costs
            .iter()
            .map(|cost| cost + f64::from(translation))
            .collect::<Vec<_>>();
        let baseline = dominance(&vector(costs), &vector(effects.clone())).expect("baseline");
        let shifted = dominance(&vector(translated), &vector(effects)).expect("shifted");

        prop_assert_eq!(baseline.frontier_indices, shifted.frontier_indices);
        prop_assert_eq!(baseline.strongly_dominated_indices, shifted.strongly_dominated_indices);
        prop_assert_eq!(baseline.extended_dominated_indices, shifted.extended_dominated_indices);
        prop_assert_eq!(baseline.icers, shifted.icers);
    }
}
