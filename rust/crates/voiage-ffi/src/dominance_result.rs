//! Typed dominance, frontier, and ICER C ABI result.

use std::panic::{self, AssertUnwindSafe};

use voiage_domain::SampleVector;
use voiage_numerics::{dominance, DominanceKernelResult, DominanceStatus};

use crate::VoiageStatusV1;

const DOMINANCE_RESULT_STRUCT_SIZE: u32 = 48;

/// Fixed-width summary for dominance and cost-effectiveness frontier results.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VoiageDominanceResultV1 {
    /// Byte size of this structure for forward-compatible callers.
    pub struct_size: u32,
    /// Version of this result structure.
    pub struct_version: u32,
    /// Number of input strategies and classification entries.
    pub strategy_count: u64,
    /// Number of valid entries written to `out_frontier_indices`.
    pub frontier_count: u64,
    /// Number of strongly dominated strategies.
    pub strongly_dominated_count: u64,
    /// Number of extended-dominated strategies.
    pub extended_dominated_count: u64,
    /// Number of valid entries in each incremental transition array.
    pub transition_count: u64,
}

/// Computes dominance classifications, the frontier, and incremental ICERs.
///
/// Status codes are `0` for frontier, `1` for strongly dominated, and `2` for
/// extended dominated. `strategy_capacity` must be at least `strategy_count`
/// for both status and frontier-index arrays. `transition_capacity` must be at
/// least `strategy_count - 1` for all three incremental arrays. No output is
/// written on failure.
///
/// # Safety
///
/// Inputs must each point to `strategy_count` readable, aligned `f64` values.
/// Every output pointer must be non-null, correctly aligned, and writable for
/// its declared capacity. All input and output regions must be non-overlapping.
/// No pointer is retained.
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn voiage_v1_dominance_result(
    costs: *const f64,
    effects: *const f64,
    strategy_count: u64,
    out_status: *mut i32,
    out_frontier_indices: *mut u64,
    strategy_capacity: u64,
    out_incremental_costs: *mut f64,
    out_incremental_effects: *mut f64,
    out_icers: *mut f64,
    transition_capacity: u64,
    out_result: *mut VoiageDominanceResultV1,
) -> VoiageStatusV1 {
    if !pointers_valid(
        costs,
        effects,
        out_status,
        out_frontier_indices,
        out_incremental_costs,
        out_incremental_effects,
        out_icers,
        out_result,
    ) {
        return VoiageStatusV1::InvalidArgument;
    }
    let Ok(count) = usize::try_from(strategy_count) else {
        return VoiageStatusV1::InvalidArgument;
    };
    if count < 2 || count > isize::MAX as usize / std::mem::size_of::<f64>() {
        return VoiageStatusV1::InvalidArgument;
    }
    if strategy_capacity < strategy_count || transition_capacity < strategy_count.saturating_sub(1)
    {
        return VoiageStatusV1::BufferTooSmall;
    }

    let computed = panic::catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: pointers and addressable lengths were validated above; the
        // caller guarantees readable, non-overlapping input regions.
        let cost_values = unsafe { std::slice::from_raw_parts(costs, count) };
        // SAFETY: same contract as cost_values.
        let effect_values = unsafe { std::slice::from_raw_parts(effects, count) };
        let costs = SampleVector::try_from(cost_values.to_vec())
            .map_err(|_| VoiageStatusV1::InvalidArgument)?;
        let effects = SampleVector::try_from(effect_values.to_vec())
            .map_err(|_| VoiageStatusV1::InvalidArgument)?;
        dominance(&costs, &effects).map_err(|_| VoiageStatusV1::NumericalFailure)
    }));
    match computed {
        Ok(Ok(result)) => {
            write_result(
                &result,
                out_status,
                out_frontier_indices,
                out_incremental_costs,
                out_incremental_effects,
                out_icers,
                out_result,
            );
            VoiageStatusV1::Ok
        }
        Ok(Err(status)) => status,
        Err(_) => VoiageStatusV1::Panic,
    }
}

#[allow(clippy::too_many_arguments)]
fn pointers_valid(
    costs: *const f64,
    effects: *const f64,
    out_status: *mut i32,
    out_frontier_indices: *mut u64,
    out_incremental_costs: *mut f64,
    out_incremental_effects: *mut f64,
    out_icers: *mut f64,
    out_result: *mut VoiageDominanceResultV1,
) -> bool {
    !costs.is_null()
        && !effects.is_null()
        && !out_status.is_null()
        && !out_frontier_indices.is_null()
        && !out_incremental_costs.is_null()
        && !out_incremental_effects.is_null()
        && !out_icers.is_null()
        && !out_result.is_null()
        && (costs as usize) % std::mem::align_of::<f64>() == 0
        && (effects as usize) % std::mem::align_of::<f64>() == 0
        && (out_status as usize) % std::mem::align_of::<i32>() == 0
        && (out_frontier_indices as usize) % std::mem::align_of::<u64>() == 0
        && (out_incremental_costs as usize) % std::mem::align_of::<f64>() == 0
        && (out_incremental_effects as usize) % std::mem::align_of::<f64>() == 0
        && (out_icers as usize) % std::mem::align_of::<f64>() == 0
        && (out_result as usize) % std::mem::align_of::<VoiageDominanceResultV1>() == 0
}

#[allow(clippy::too_many_arguments)]
fn write_result(
    result: &DominanceKernelResult,
    out_status: *mut i32,
    out_frontier_indices: *mut u64,
    out_incremental_costs: *mut f64,
    out_incremental_effects: *mut f64,
    out_icers: *mut f64,
    out_result: *mut VoiageDominanceResultV1,
) {
    let statuses = result
        .status
        .iter()
        .map(|status| match status {
            DominanceStatus::Frontier => 0_i32,
            DominanceStatus::StronglyDominated => 1,
            DominanceStatus::ExtendedDominated => 2,
        })
        .collect::<Vec<_>>();
    let frontier = result
        .frontier_indices
        .iter()
        .map(|&index| index as u64)
        .collect::<Vec<_>>();
    let summary = VoiageDominanceResultV1 {
        struct_size: DOMINANCE_RESULT_STRUCT_SIZE,
        struct_version: 1,
        strategy_count: statuses.len() as u64,
        frontier_count: frontier.len() as u64,
        strongly_dominated_count: result.strongly_dominated_indices.len() as u64,
        extended_dominated_count: result.extended_dominated_indices.len() as u64,
        transition_count: result.icers.len() as u64,
    };

    // SAFETY: pointer validity, non-overlap, and worst-case capacities are
    // established by the public contract before computation.
    unsafe {
        std::ptr::copy_nonoverlapping(statuses.as_ptr(), out_status, statuses.len());
        std::ptr::copy_nonoverlapping(frontier.as_ptr(), out_frontier_indices, frontier.len());
        std::ptr::copy_nonoverlapping(
            result.incremental_costs.as_ptr(),
            out_incremental_costs,
            result.incremental_costs.len(),
        );
        std::ptr::copy_nonoverlapping(
            result.incremental_effects.as_ptr(),
            out_incremental_effects,
            result.incremental_effects.len(),
        );
        std::ptr::copy_nonoverlapping(result.icers.as_ptr(), out_icers, result.icers.len());
        out_result.write(summary);
    }
}
