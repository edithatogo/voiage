//! Promoted Rust-native EVSI approximation C ABI results.

use std::panic::{self, AssertUnwindSafe};

use voiage_diagnostics::ErrorCategory;
use voiage_numerics::{
    evsi_efficient_linear, evsi_moment_based, EvsiApproximationResult, NumericalInputError,
};

use crate::{checked_dimensions, validated_matrix, VoiageStatusV1};

const EVSI_APPROXIMATION_RESULT_STRUCT_SIZE: u32 = 88;

/// Stable code for the promoted linear-regression EVSI estimator.
pub const VOIAGE_EVSI_ESTIMATOR_REGRESSION: u32 = 1;
/// Stable code for the promoted moment-matching EVSI estimator.
pub const VOIAGE_EVSI_ESTIMATOR_MOMENT_MATCHING: u32 = 2;
/// A single deterministic fit does not provide replicate assurance.
pub const VOIAGE_EVSI_ASSURANCE_INCOMPLETE: u32 = 0;

/// Fixed-width result shared by promoted deterministic EVSI estimators.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VoiageEvsiApproximationResultV1 {
    /// Byte size of this structure for forward-compatible callers.
    pub struct_size: u32,
    /// Version of this result structure.
    pub struct_version: u32,
    /// Non-negative expected value of sample information.
    pub evsi: f64,
    /// Expected value under current information.
    pub expected_current_value: f64,
    /// Expected value after the estimator-specific approximation.
    pub expected_sample_value: f64,
    /// Expected value under perfect information.
    pub expected_perfect_information: f64,
    /// Fraction of information represented by the trial-size proxy.
    pub information_fraction: f64,
    /// Number of aligned uncertainty samples.
    pub sample_count: u64,
    /// Number of strategies.
    pub strategy_count: u64,
    /// Number of parameter columns.
    pub parameter_count: u64,
    /// Requested trial sample size.
    pub trial_sample_size: u64,
    /// One of the stable `VOIAGE_V1_EVSI_ESTIMATOR_*` codes.
    pub estimator_kind: u32,
    /// Zero: replicate assurance is not established by one fit.
    pub assurance_state: u32,
}

/// Computes the promoted linear-regression EVSI approximation.
///
/// # Safety
///
/// Inputs and output must be non-null, correctly aligned, readable or writable
/// for their declared lengths, and mutually non-overlapping. No pointer is
/// retained, and output is not written on failure.
#[no_mangle]
pub unsafe extern "C" fn voiage_v1_evsi_regression_result(
    net_benefit: *const f64,
    sample_count: u64,
    strategy_count: u64,
    parameter_samples: *const f64,
    parameter_sample_count: u64,
    parameter_count: u64,
    trial_sample_size: u64,
    out_result: *mut VoiageEvsiApproximationResultV1,
) -> VoiageStatusV1 {
    compute(
        net_benefit,
        sample_count,
        strategy_count,
        parameter_samples,
        parameter_sample_count,
        parameter_count,
        trial_sample_size,
        out_result,
        Estimator::Regression,
    )
}

/// Computes the promoted centered moment-matching EVSI approximation.
///
/// # Safety
///
/// Inputs and output must be non-null, correctly aligned, readable or writable
/// for their declared lengths, and mutually non-overlapping. No pointer is
/// retained, and output is not written on failure.
#[no_mangle]
pub unsafe extern "C" fn voiage_v1_evsi_moment_matching_result(
    net_benefit: *const f64,
    sample_count: u64,
    strategy_count: u64,
    parameter_samples: *const f64,
    parameter_sample_count: u64,
    parameter_count: u64,
    trial_sample_size: u64,
    out_result: *mut VoiageEvsiApproximationResultV1,
) -> VoiageStatusV1 {
    compute(
        net_benefit,
        sample_count,
        strategy_count,
        parameter_samples,
        parameter_sample_count,
        parameter_count,
        trial_sample_size,
        out_result,
        Estimator::MomentMatching,
    )
}

#[derive(Clone, Copy)]
enum Estimator {
    Regression,
    MomentMatching,
}

#[allow(clippy::too_many_arguments)]
fn compute(
    net_benefit: *const f64,
    sample_count: u64,
    strategy_count: u64,
    parameter_samples: *const f64,
    parameter_sample_count: u64,
    parameter_count: u64,
    trial_sample_size: u64,
    out_result: *mut VoiageEvsiApproximationResultV1,
    estimator: Estimator,
) -> VoiageStatusV1 {
    if net_benefit.is_null()
        || parameter_samples.is_null()
        || out_result.is_null()
        || (net_benefit as usize) % std::mem::align_of::<f64>() != 0
        || (parameter_samples as usize) % std::mem::align_of::<f64>() != 0
        || (out_result as usize) % std::mem::align_of::<VoiageEvsiApproximationResultV1>() != 0
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
    let Ok(trial_size) = usize::try_from(trial_sample_size) else {
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
        match estimator {
            Estimator::Regression => evsi_efficient_linear(&net_benefit, &parameters, trial_size),
            Estimator::MomentMatching => evsi_moment_based(&net_benefit, &parameters, trial_size),
        }
        .map_err(|error| map_numerical_error(&error))
    }));

    match computed {
        Ok(Ok(result)) => {
            write_result(
                &result,
                trial_sample_size,
                match estimator {
                    Estimator::Regression => VOIAGE_EVSI_ESTIMATOR_REGRESSION,
                    Estimator::MomentMatching => VOIAGE_EVSI_ESTIMATOR_MOMENT_MATCHING,
                },
                out_result,
            );
            VoiageStatusV1::Ok
        }
        Ok(Err(status)) => status,
        Err(_) => VoiageStatusV1::Panic,
    }
}

fn map_numerical_error(error: &NumericalInputError) -> VoiageStatusV1 {
    match error.category() {
        ErrorCategory::DimensionMismatch => VoiageStatusV1::DimensionMismatch,
        ErrorCategory::Numerical => VoiageStatusV1::NumericalFailure,
        _ => VoiageStatusV1::InvalidArgument,
    }
}

fn write_result(
    result: &EvsiApproximationResult,
    trial_sample_size: u64,
    estimator_kind: u32,
    out: *mut VoiageEvsiApproximationResultV1,
) {
    let envelope = VoiageEvsiApproximationResultV1 {
        struct_size: EVSI_APPROXIMATION_RESULT_STRUCT_SIZE,
        struct_version: 1,
        evsi: result.evsi,
        expected_current_value: result.expected_current_value,
        expected_sample_value: result.expected_sample_value,
        expected_perfect_information: result.expected_perfect_information,
        information_fraction: result.information_fraction,
        sample_count: result.sample_count as u64,
        strategy_count: result.strategy_count as u64,
        parameter_count: result.parameter_count as u64,
        trial_sample_size,
        estimator_kind,
        assurance_state: VOIAGE_EVSI_ASSURANCE_INCOMPLETE,
    };
    // SAFETY: output nullness and alignment were validated before computation.
    unsafe { out.write(envelope) };
}
