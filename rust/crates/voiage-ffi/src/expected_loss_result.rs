//! Typed expected-loss result and caller-owned per-strategy arrays.

use std::panic::{self, AssertUnwindSafe};

use voiage_numerics::{expected_loss, ExpectedLossKernelResult};

use crate::{checked_dimensions, validated_matrix, VoiageStatusV1};

const EXPECTED_LOSS_RESULT_STRUCT_SIZE: u32 = 64;

/// Fixed-width expected-loss summary with explicit assurance availability.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VoiageExpectedLossResultV1 {
    /// Byte size of this structure for forward-compatible callers.
    pub struct_size: u32,
    /// Version of this result structure.
    pub struct_version: u32,
    /// Lowest-index strategy with greatest expected net benefit.
    pub optimal_strategy_index: u64,
    /// Number of uncertainty samples.
    pub sample_count: u64,
    /// Number of strategies.
    pub strategy_count: u64,
    /// Expected opportunity loss of the current-information optimum.
    pub minimum_expected_opportunity_loss: f64,
    /// One when variance and Monte Carlo error are available, otherwise zero.
    pub has_assurance: u32,
    /// Reserved; callers must ignore and producers set to zero.
    pub reserved: u32,
    /// Unbiased opportunity-loss variance for the selected strategy.
    pub opportunity_loss_variance: f64,
    /// Monte Carlo standard error of expected loss.
    pub monte_carlo_standard_error: f64,
}

/// Computes expected loss into caller-owned strategy arrays and a typed result.
///
/// `array_capacity` is the number of `f64` elements available in both output
/// arrays and must be at least `columns`. No output is written on failure.
///
/// # Safety
///
/// `values` must point to `rows * columns` readable `f64` values.
/// `out_expected_net_benefit` and `out_expected_opportunity_loss` must each
/// point to `array_capacity` writable, aligned `f64` values. `out_result` must
/// be writable and aligned for one [`VoiageExpectedLossResultV1`]. These four
/// regions must not overlap and no pointer is retained.
#[no_mangle]
pub unsafe extern "C" fn voiage_v1_expected_loss_result(
    values: *const f64,
    rows: u64,
    columns: u64,
    out_expected_net_benefit: *mut f64,
    out_expected_opportunity_loss: *mut f64,
    array_capacity: u64,
    out_result: *mut VoiageExpectedLossResultV1,
) -> VoiageStatusV1 {
    if values.is_null()
        || out_expected_net_benefit.is_null()
        || out_expected_opportunity_loss.is_null()
        || out_result.is_null()
        || (values as usize) % std::mem::align_of::<f64>() != 0
        || (out_expected_net_benefit as usize) % std::mem::align_of::<f64>() != 0
        || (out_expected_opportunity_loss as usize) % std::mem::align_of::<f64>() != 0
        || (out_result as usize) % std::mem::align_of::<VoiageExpectedLossResultV1>() != 0
    {
        return VoiageStatusV1::InvalidArgument;
    }
    let Ok((row_count, column_count, length)) = checked_dimensions(rows, columns) else {
        return VoiageStatusV1::InvalidArgument;
    };
    if array_capacity < columns {
        return VoiageStatusV1::BufferTooSmall;
    }

    let computed = panic::catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: checked_dimensions bounds the readable slice to isize::MAX
        // bytes and the caller guarantees that region.
        let slice = unsafe { std::slice::from_raw_parts(values, length) };
        let matrix = validated_matrix(slice, row_count, column_count)?;
        expected_loss(&matrix).map_err(|_| VoiageStatusV1::NumericalFailure)
    }));
    match computed {
        Ok(Ok(result)) => {
            write_result(
                &result,
                out_expected_net_benefit,
                out_expected_opportunity_loss,
                out_result,
            );
            VoiageStatusV1::Ok
        }
        Ok(Err(status)) => status,
        Err(_) => VoiageStatusV1::Panic,
    }
}

fn write_result(
    result: &ExpectedLossKernelResult,
    out_expected_net_benefit: *mut f64,
    out_expected_opportunity_loss: *mut f64,
    out_result: *mut VoiageExpectedLossResultV1,
) {
    let assurance = result
        .opportunity_loss_variance
        .zip(result.monte_carlo_standard_error);
    let envelope = VoiageExpectedLossResultV1 {
        struct_size: EXPECTED_LOSS_RESULT_STRUCT_SIZE,
        struct_version: 1,
        optimal_strategy_index: result.optimal_strategy_index as u64,
        sample_count: result.sample_count as u64,
        strategy_count: result.strategy_count as u64,
        minimum_expected_opportunity_loss: result.minimum_expected_opportunity_loss,
        has_assurance: u32::from(assurance.is_some()),
        reserved: 0,
        opportunity_loss_variance: assurance.map_or(0.0, |values| values.0),
        monte_carlo_standard_error: assurance.map_or(0.0, |values| values.1),
    };

    // SAFETY: pointers, alignment, non-overlap, and capacity are established
    // by the public caller contract and validated before computation.
    unsafe {
        std::ptr::copy_nonoverlapping(
            result.expected_net_benefit_by_strategy.as_ptr(),
            out_expected_net_benefit,
            result.strategy_count,
        );
        std::ptr::copy_nonoverlapping(
            result.expected_opportunity_loss_by_strategy.as_ptr(),
            out_expected_opportunity_loss,
            result.strategy_count,
        );
        out_result.write(envelope);
    }
}
