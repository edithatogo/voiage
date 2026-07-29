//! Contract tests for typed dominance and ICER C ABI results.

#![allow(unsafe_code)]

use voiage_ffi::{
    voiage_v1_dominance_result, VoiageDominanceResultV1, VoiageStatusV1,
    VOIAGE_ABI_DOMINANCE_RESULT,
};

fn empty_result() -> VoiageDominanceResultV1 {
    VoiageDominanceResultV1 {
        struct_size: 0,
        struct_version: 0,
        strategy_count: 0,
        frontier_count: 0,
        strongly_dominated_count: 0,
        extended_dominated_count: 0,
        transition_count: 0,
    }
}

#[test]
fn dominance_result_writes_classifications_frontier_and_icers() {
    let costs = [100.0, 120.0, 150.0];
    let effects = [1.0, 0.9, 2.0];
    let mut statuses = [-1_i32; 3];
    let mut frontier = [u64::MAX; 3];
    let mut incremental_costs = [f64::NAN; 2];
    let mut incremental_effects = [f64::NAN; 2];
    let mut icers = [f64::NAN; 2];
    let mut result = empty_result();

    let status = unsafe {
        voiage_v1_dominance_result(
            costs.as_ptr(),
            effects.as_ptr(),
            3,
            statuses.as_mut_ptr(),
            frontier.as_mut_ptr(),
            3,
            incremental_costs.as_mut_ptr(),
            incremental_effects.as_mut_ptr(),
            icers.as_mut_ptr(),
            2,
            &raw mut result,
        )
    };

    assert_eq!(status, VoiageStatusV1::Ok);
    assert_eq!(statuses, [0, 1, 0]);
    assert_eq!(&frontier[..2], &[0, 2]);
    assert!((incremental_costs[0] - 50.0).abs() < f64::EPSILON);
    assert!((incremental_effects[0] - 1.0).abs() < f64::EPSILON);
    assert!((icers[0] - 50.0).abs() < f64::EPSILON);
    assert_eq!(result.struct_size, 48);
    assert_eq!(result.struct_version, 1);
    assert_eq!(result.strategy_count, 3);
    assert_eq!(result.frontier_count, 2);
    assert_eq!(result.strongly_dominated_count, 1);
    assert_eq!(result.extended_dominated_count, 0);
    assert_eq!(result.transition_count, 1);
}

#[test]
fn dominance_result_rejects_short_capacity_without_partial_writes() {
    let costs = [100.0, 120.0, 150.0];
    let effects = [1.0, 0.9, 2.0];
    let mut statuses = [-1_i32; 3];
    let mut frontier = [u64::MAX; 3];
    let mut incremental_costs = [101.0; 2];
    let mut incremental_effects = [102.0; 2];
    let mut icers = [103.0; 2];
    let mut result = empty_result();

    let status = unsafe {
        voiage_v1_dominance_result(
            costs.as_ptr(),
            effects.as_ptr(),
            3,
            statuses.as_mut_ptr(),
            frontier.as_mut_ptr(),
            2,
            incremental_costs.as_mut_ptr(),
            incremental_effects.as_mut_ptr(),
            icers.as_mut_ptr(),
            2,
            &raw mut result,
        )
    };

    assert_eq!(status, VoiageStatusV1::BufferTooSmall);
    assert_eq!(statuses, [-1; 3]);
    assert_eq!(frontier, [u64::MAX; 3]);
    assert!(incremental_costs
        .iter()
        .all(|value| value.to_bits() == 101.0_f64.to_bits()));
    assert!(incremental_effects
        .iter()
        .all(|value| value.to_bits() == 102.0_f64.to_bits()));
    assert!(icers
        .iter()
        .all(|value| value.to_bits() == 103.0_f64.to_bits()));
    assert_eq!(result, empty_result());
}

#[test]
fn dominance_capability_bit_is_nonzero() {
    assert_ne!(VOIAGE_ABI_DOMINANCE_RESULT, 0);
}
