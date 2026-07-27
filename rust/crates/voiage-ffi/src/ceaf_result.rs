//! Threshold-aligned CEAF C ABI result.

use std::panic::{self, AssertUnwindSafe};

use voiage_domain::{SampleCube, SampleVector};
use voiage_numerics::{ceaf, CeafKernelResult};

use crate::VoiageStatusV1;

const CEAF_RESULT_STRUCT_SIZE: u32 = 32;

/// Fixed-width summary for threshold-aligned CEAF outputs.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VoiageCeafResultV1 {
    /// Byte size of this structure for forward-compatible callers.
    pub struct_size: u32,
    /// Version of this result structure.
    pub struct_version: u32,
    /// Number of uncertainty samples.
    pub sample_count: u64,
    /// Number of strategies.
    pub strategy_count: u64,
    /// Number of threshold-aligned entries in every output array.
    pub threshold_count: u64,
}

/// Computes CEAF outputs into caller-owned threshold-aligned arrays.
///
/// Net benefit is row-major `[sample][strategy][threshold]`.
/// `threshold_capacity` must be at least `threshold_count` for every output
/// array. No output is written on failure.
///
/// # Safety
///
/// Inputs and outputs must be non-null, correctly aligned, readable or
/// writable for their declared lengths, and mutually non-overlapping. No
/// pointer is retained.
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn voiage_v1_ceaf_result(
    values: *const f64,
    sample_count: u64,
    strategy_count: u64,
    threshold_count: u64,
    thresholds: *const f64,
    confidence_level: f64,
    out_optimal_strategy_indices: *mut u64,
    out_acceptability_probabilities: *mut f64,
    out_probability_lower: *mut f64,
    out_probability_upper: *mut f64,
    out_expected_net_benefit: *mut f64,
    out_has_assurance: *mut u32,
    out_probability_variance: *mut f64,
    out_probability_standard_error: *mut f64,
    threshold_capacity: u64,
    out_result: *mut VoiageCeafResultV1,
) -> VoiageStatusV1 {
    if !pointers_valid(
        values,
        thresholds,
        out_optimal_strategy_indices,
        out_acceptability_probabilities,
        out_probability_lower,
        out_probability_upper,
        out_expected_net_benefit,
        out_has_assurance,
        out_probability_variance,
        out_probability_standard_error,
        out_result,
    ) {
        return VoiageStatusV1::InvalidArgument;
    }
    let Ok((samples, strategies, threshold_total, value_count)) =
        checked_dimensions(sample_count, strategy_count, threshold_count)
    else {
        return VoiageStatusV1::InvalidArgument;
    };
    if threshold_capacity < threshold_count {
        return VoiageStatusV1::BufferTooSmall;
    }

    let computed = panic::catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: pointer alignment and addressable lengths were validated;
        // the caller guarantees readable, non-overlapping inputs.
        let flat_values = unsafe { std::slice::from_raw_parts(values, value_count) };
        // SAFETY: threshold_total is addressable and caller-provided.
        let threshold_values = unsafe { std::slice::from_raw_parts(thresholds, threshold_total) };
        let mut offset = 0;
        let cube_values = (0..samples)
            .map(|_| {
                (0..strategies)
                    .map(|_| {
                        let row = flat_values[offset..offset + threshold_total].to_vec();
                        offset += threshold_total;
                        row
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let cube =
            SampleCube::try_from(cube_values).map_err(|_| VoiageStatusV1::InvalidArgument)?;
        let thresholds = SampleVector::try_from(threshold_values.to_vec())
            .map_err(|_| VoiageStatusV1::InvalidArgument)?;
        ceaf(&cube, &thresholds, confidence_level).map_err(|_| VoiageStatusV1::InvalidArgument)
    }));
    match computed {
        Ok(Ok(result)) => {
            write_result(
                &result,
                samples,
                strategies,
                out_optimal_strategy_indices,
                out_acceptability_probabilities,
                out_probability_lower,
                out_probability_upper,
                out_expected_net_benefit,
                out_has_assurance,
                out_probability_variance,
                out_probability_standard_error,
                out_result,
            );
            VoiageStatusV1::Ok
        }
        Ok(Err(status)) => status,
        Err(_) => VoiageStatusV1::Panic,
    }
}

fn checked_dimensions(
    sample_count: u64,
    strategy_count: u64,
    threshold_count: u64,
) -> Result<(usize, usize, usize, usize), VoiageStatusV1> {
    if sample_count == 0 || strategy_count == 0 || threshold_count == 0 {
        return Err(VoiageStatusV1::InvalidArgument);
    }
    let value_count = sample_count
        .checked_mul(strategy_count)
        .and_then(|count| count.checked_mul(threshold_count))
        .and_then(|count| usize::try_from(count).ok())
        .ok_or(VoiageStatusV1::InvalidArgument)?;
    if value_count > isize::MAX as usize / std::mem::size_of::<f64>() {
        return Err(VoiageStatusV1::InvalidArgument);
    }
    Ok((
        usize::try_from(sample_count).map_err(|_| VoiageStatusV1::InvalidArgument)?,
        usize::try_from(strategy_count).map_err(|_| VoiageStatusV1::InvalidArgument)?,
        usize::try_from(threshold_count).map_err(|_| VoiageStatusV1::InvalidArgument)?,
        value_count,
    ))
}

#[allow(clippy::too_many_arguments)]
fn pointers_valid(
    values: *const f64,
    thresholds: *const f64,
    out_optimal_strategy_indices: *mut u64,
    out_acceptability_probabilities: *mut f64,
    out_probability_lower: *mut f64,
    out_probability_upper: *mut f64,
    out_expected_net_benefit: *mut f64,
    out_has_assurance: *mut u32,
    out_probability_variance: *mut f64,
    out_probability_standard_error: *mut f64,
    out_result: *mut VoiageCeafResultV1,
) -> bool {
    !values.is_null()
        && !thresholds.is_null()
        && !out_optimal_strategy_indices.is_null()
        && !out_acceptability_probabilities.is_null()
        && !out_probability_lower.is_null()
        && !out_probability_upper.is_null()
        && !out_expected_net_benefit.is_null()
        && !out_has_assurance.is_null()
        && !out_probability_variance.is_null()
        && !out_probability_standard_error.is_null()
        && !out_result.is_null()
        && (values as usize) % std::mem::align_of::<f64>() == 0
        && (thresholds as usize) % std::mem::align_of::<f64>() == 0
        && (out_optimal_strategy_indices as usize) % std::mem::align_of::<u64>() == 0
        && (out_acceptability_probabilities as usize) % std::mem::align_of::<f64>() == 0
        && (out_probability_lower as usize) % std::mem::align_of::<f64>() == 0
        && (out_probability_upper as usize) % std::mem::align_of::<f64>() == 0
        && (out_expected_net_benefit as usize) % std::mem::align_of::<f64>() == 0
        && (out_has_assurance as usize) % std::mem::align_of::<u32>() == 0
        && (out_probability_variance as usize) % std::mem::align_of::<f64>() == 0
        && (out_probability_standard_error as usize) % std::mem::align_of::<f64>() == 0
        && (out_result as usize) % std::mem::align_of::<VoiageCeafResultV1>() == 0
}

#[allow(clippy::too_many_arguments)]
fn write_result(
    result: &CeafKernelResult,
    sample_count: usize,
    strategy_count: usize,
    out_optimal_strategy_indices: *mut u64,
    out_acceptability_probabilities: *mut f64,
    out_probability_lower: *mut f64,
    out_probability_upper: *mut f64,
    out_expected_net_benefit: *mut f64,
    out_has_assurance: *mut u32,
    out_probability_variance: *mut f64,
    out_probability_standard_error: *mut f64,
    out_result: *mut VoiageCeafResultV1,
) {
    let optimal = result
        .optimal_strategy_indices
        .iter()
        .map(|&index| index as u64)
        .collect::<Vec<_>>();
    let has_assurance = result
        .probability_variance
        .iter()
        .zip(&result.probability_standard_error)
        .map(|(variance, error)| u32::from(variance.is_some() && error.is_some()))
        .collect::<Vec<_>>();
    let variance = result
        .probability_variance
        .iter()
        .map(|value| value.unwrap_or(0.0))
        .collect::<Vec<_>>();
    let standard_error = result
        .probability_standard_error
        .iter()
        .map(|value| value.unwrap_or(0.0))
        .collect::<Vec<_>>();
    let summary = VoiageCeafResultV1 {
        struct_size: CEAF_RESULT_STRUCT_SIZE,
        struct_version: 1,
        sample_count: sample_count as u64,
        strategy_count: strategy_count as u64,
        threshold_count: optimal.len() as u64,
    };
    let count = optimal.len();

    // SAFETY: pointer validity, non-overlap, and capacities are established by
    // the public contract before computation.
    unsafe {
        std::ptr::copy_nonoverlapping(optimal.as_ptr(), out_optimal_strategy_indices, count);
        std::ptr::copy_nonoverlapping(
            result.acceptability_probabilities.as_ptr(),
            out_acceptability_probabilities,
            count,
        );
        std::ptr::copy_nonoverlapping(
            result.probability_lower.as_ptr(),
            out_probability_lower,
            count,
        );
        std::ptr::copy_nonoverlapping(
            result.probability_upper.as_ptr(),
            out_probability_upper,
            count,
        );
        std::ptr::copy_nonoverlapping(
            result.expected_net_benefit.as_ptr(),
            out_expected_net_benefit,
            count,
        );
        std::ptr::copy_nonoverlapping(has_assurance.as_ptr(), out_has_assurance, count);
        std::ptr::copy_nonoverlapping(variance.as_ptr(), out_probability_variance, count);
        std::ptr::copy_nonoverlapping(
            standard_error.as_ptr(),
            out_probability_standard_error,
            count,
        );
        out_result.write(summary);
    }
}
