//! Stable full-sample linear-regression EVPPI C ABI result.

use std::panic::{self, AssertUnwindSafe};

use voiage_diagnostics::ErrorCategory;
use voiage_numerics::{evppi_with_assurance, EvppiKernelResult};

use crate::{checked_dimensions, validated_matrix, VoiageStatusV1};

const EVPPI_REGRESSION_RESULT_STRUCT_SIZE: u32 = 48;

/// The fit is deterministic, but estimator assurance remains incomplete.
pub const VOIAGE_EVPPI_ASSURANCE_INCOMPLETE: u32 = 0;

/// Fixed-width result for the stable full-sample linear EVPPI estimator.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VoiageEvppiRegressionResultV1 {
    /// Byte size of this structure for forward-compatible callers.
    pub struct_size: u32,
    /// Version of this result structure.
    pub struct_version: u32,
    /// Non-negative regression-based EVPPI estimate.
    pub value: f64,
    /// Number of aligned PSA samples.
    pub sample_count: u64,
    /// Number of strategies.
    pub strategy_count: u64,
    /// Number of parameters in the linear design.
    pub parameter_count: u64,
    /// Zero: a single fit does not establish bias, variance, or convergence.
    pub assurance_state: u32,
    /// Reserved; callers must ignore and producers set to zero.
    pub reserved: u32,
}

/// Computes the stable full-sample linear-regression EVPPI result.
///
/// Net benefit is row-major `[sample][strategy]`; parameter samples are
/// row-major `[sample][parameter]`. No output is written on failure.
///
/// # Safety
///
/// Inputs and output must be non-null, correctly aligned, readable or writable
/// for their declared lengths, and mutually non-overlapping. No pointer is
/// retained.
#[no_mangle]
pub unsafe extern "C" fn voiage_v1_evppi_regression_result(
    net_benefit: *const f64,
    sample_count: u64,
    strategy_count: u64,
    parameter_samples: *const f64,
    parameter_sample_count: u64,
    parameter_count: u64,
    out_result: *mut VoiageEvppiRegressionResultV1,
) -> VoiageStatusV1 {
    if net_benefit.is_null()
        || parameter_samples.is_null()
        || out_result.is_null()
        || (net_benefit as usize) % std::mem::align_of::<f64>() != 0
        || (parameter_samples as usize) % std::mem::align_of::<f64>() != 0
        || (out_result as usize) % std::mem::align_of::<VoiageEvppiRegressionResultV1>() != 0
    {
        return VoiageStatusV1::InvalidArgument;
    }
    let Ok((samples, strategies, net_benefit_length)) =
        checked_dimensions(sample_count, strategy_count)
    else {
        return VoiageStatusV1::InvalidArgument;
    };
    let Ok((parameter_rows, parameters, parameter_length)) =
        checked_dimensions(parameter_sample_count, parameter_count)
    else {
        return VoiageStatusV1::InvalidArgument;
    };

    let computed = panic::catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: dimensions bound both slices to addressable lengths and the
        // caller guarantees readable, non-overlapping input regions.
        let net_benefit_values =
            unsafe { std::slice::from_raw_parts(net_benefit, net_benefit_length) };
        // SAFETY: validated as above for the parameter matrix.
        let parameter_values =
            unsafe { std::slice::from_raw_parts(parameter_samples, parameter_length) };
        let net_benefit = validated_matrix(net_benefit_values, samples, strategies)?;
        let parameters = validated_matrix(parameter_values, parameter_rows, parameters)?;
        evppi_with_assurance(&net_benefit, &parameters).map_err(|error| map_numerical_error(&error))
    }));

    match computed {
        Ok(Ok(result)) => {
            write_result(&result, out_result);
            VoiageStatusV1::Ok
        }
        Ok(Err(status)) => status,
        Err(_) => VoiageStatusV1::Panic,
    }
}

fn map_numerical_error(error: &voiage_numerics::NumericalInputError) -> VoiageStatusV1 {
    match error.category() {
        ErrorCategory::DimensionMismatch => VoiageStatusV1::DimensionMismatch,
        ErrorCategory::Numerical => VoiageStatusV1::NumericalFailure,
        _ => VoiageStatusV1::InvalidArgument,
    }
}

fn write_result(result: &EvppiKernelResult, out: *mut VoiageEvppiRegressionResultV1) {
    let envelope = VoiageEvppiRegressionResultV1 {
        struct_size: EVPPI_REGRESSION_RESULT_STRUCT_SIZE,
        struct_version: 1,
        value: result.value,
        sample_count: result.sample_count as u64,
        strategy_count: result.strategy_count as u64,
        parameter_count: result.parameter_count as u64,
        assurance_state: VOIAGE_EVPPI_ASSURANCE_INCOMPLETE,
        reserved: 0,
    };
    // SAFETY: output nullness and alignment were validated before computation.
    unsafe { out.write(envelope) };
}
