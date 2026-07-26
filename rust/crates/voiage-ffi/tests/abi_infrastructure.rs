//! Contract tests for the portable v1 C ABI.

#![allow(unsafe_code)]

use std::mem::{align_of, offset_of, size_of};

use voiage_ffi::{
    voiage_v1_abi_version, voiage_v1_capabilities, voiage_v1_evpi, voiage_v1_evpi_i32,
    voiage_v1_evpi_result, VoiageAbiCapabilitiesV1, VoiageAbiVersionV1, VoiageEvpiResultV1,
    VoiageExpectedLossResultV1, VoiageStatusV1, VOIAGE_ABI_CAPABILITY_DOCUMENT,
    VOIAGE_ABI_CAPABILITY_QUERY, VOIAGE_ABI_ENBS, VOIAGE_ABI_EVPI_RESULT,
    VOIAGE_ABI_EXPECTED_LOSS_RESULT, VOIAGE_ABI_VERSION_NEGOTIATION, VOIAGE_V1_ABI_MAJOR,
    VOIAGE_V1_ABI_MINOR,
};

const LAYOUT_BASELINE: &str = include_str!("../../../../specs/abi/v1/layouts.txt");

#[test]
fn version_query_returns_a_fixed_width_self_describing_structure() {
    let version = voiage_v1_abi_version();

    assert_eq!(size_of::<VoiageAbiVersionV1>(), 12);
    assert_eq!(align_of::<VoiageAbiVersionV1>(), align_of::<u32>());
    assert_eq!(version.struct_size, 12);
    assert_eq!(version.abi_major, VOIAGE_V1_ABI_MAJOR);
    assert_eq!(version.abi_minor, VOIAGE_V1_ABI_MINOR);
}

#[test]
fn capability_query_advertises_typed_evpi_results() {
    let capabilities = voiage_v1_capabilities();

    assert_eq!(size_of::<VoiageAbiCapabilitiesV1>(), 16);
    assert_eq!(capabilities.struct_size, 16);
    assert_eq!(capabilities.struct_version, 1);
    assert_eq!(
        capabilities.capability_bits,
        VOIAGE_ABI_VERSION_NEGOTIATION
            | VOIAGE_ABI_CAPABILITY_QUERY
            | voiage_ffi::VOIAGE_ABI_EVPI
            | VOIAGE_ABI_EVPI_RESULT
            | VOIAGE_ABI_CAPABILITY_DOCUMENT
            | VOIAGE_ABI_EXPECTED_LOSS_RESULT
            | VOIAGE_ABI_ENBS
    );
    assert_eq!(capabilities.capability_bits & !0b111_1111, 0);
}

#[test]
fn public_queries_have_the_exact_namespaced_function_signatures() {
    let version_query: extern "C" fn() -> VoiageAbiVersionV1 = voiage_v1_abi_version;
    let capability_query: extern "C" fn() -> VoiageAbiCapabilitiesV1 = voiage_v1_capabilities;
    let typed_evpi: unsafe extern "C" fn(
        *const f64,
        u64,
        u64,
        *mut VoiageEvpiResultV1,
    ) -> VoiageStatusV1 = voiage_v1_evpi_result;

    let _ = (version_query, capability_query, typed_evpi);
}

#[test]
fn evpi_abi_executes_the_rust_kernel_and_validates_shape() {
    let values = [10.0, 1.0, 2.0, 8.0];
    let mut result = 0.0;
    let status = unsafe { voiage_v1_evpi(values.as_ptr(), 2, 2, &raw mut result) };
    assert_eq!(status, VoiageStatusV1::Ok);
    assert!((result - 3.0).abs() < f64::EPSILON);

    let status = unsafe { voiage_v1_evpi(std::ptr::null(), 2, 2, &raw mut result) };
    assert_eq!(status, VoiageStatusV1::InvalidArgument);

    let storage = [0_u8; 40];
    #[allow(clippy::cast_ptr_alignment)]
    let misaligned = unsafe { storage.as_ptr().add(1).cast::<f64>() };
    let status = unsafe { voiage_v1_evpi(misaligned, 2, 2, &raw mut result) };
    assert_eq!(status, VoiageStatusV1::InvalidArgument);

    let aligned = [0.0];
    let status = unsafe { voiage_v1_evpi(aligned.as_ptr(), u64::MAX, 2, &raw mut result) };
    assert_eq!(status, VoiageStatusV1::InvalidArgument);
}

#[test]
fn evpi_i32_abi_adapter_reuses_the_rust_kernel() {
    let values = [10.0, 1.0, 2.0, 8.0];
    let mut result = 0.0;
    let status = unsafe { voiage_v1_evpi_i32(values.as_ptr(), 2, 2, &raw mut result) };
    assert_eq!(status, VoiageStatusV1::Ok);
    assert!((result - 3.0).abs() < f64::EPSILON);
}

#[test]
fn typed_evpi_result_exposes_dimensions_and_assurance() {
    let values = [10.0, 1.0, 2.0, 8.0];
    let mut result = VoiageEvpiResultV1 {
        struct_size: 0,
        struct_version: 0,
        value: 0.0,
        sample_count: 0,
        strategy_count: 0,
        has_assurance: 0,
        reserved: 1,
        opportunity_loss_variance: 0.0,
        monte_carlo_standard_error: 0.0,
    };
    let status = unsafe { voiage_v1_evpi_result(values.as_ptr(), 2, 2, &raw mut result) };

    assert_eq!(status, VoiageStatusV1::Ok);
    assert_eq!(result.struct_size, 56);
    assert_eq!(result.struct_version, 1);
    assert!((result.value - 3.0).abs() < f64::EPSILON);
    assert_eq!(result.sample_count, 2);
    assert_eq!(result.strategy_count, 2);
    assert_eq!(result.has_assurance, 1);
    assert_eq!(result.reserved, 0);
    assert!((result.opportunity_loss_variance - 18.0).abs() < f64::EPSILON);
    assert!((result.monte_carlo_standard_error - 3.0).abs() < f64::EPSILON);

    let status = unsafe { voiage_v1_evpi_result(values.as_ptr(), 2, 2, std::ptr::null_mut()) };
    assert_eq!(status, VoiageStatusV1::InvalidArgument);
}

#[test]
fn committed_layout_baseline_matches_rust_types_exactly() {
    let expected = LAYOUT_BASELINE
        .lines()
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .collect::<Vec<_>>()
        .join("\n");
    let actual = format!(
        concat!(
            "VoiageAbiVersionV1 {} {}\n",
            "VoiageAbiVersionV1.struct_size {}\n",
            "VoiageAbiVersionV1.abi_major {}\n",
            "VoiageAbiVersionV1.abi_minor {}\n",
            "VoiageAbiCapabilitiesV1 {} {}\n",
            "VoiageAbiCapabilitiesV1.struct_size {}\n",
            "VoiageAbiCapabilitiesV1.struct_version {}\n",
            "VoiageAbiCapabilitiesV1.capability_bits {}\n",
            "VoiageEvpiResultV1 {} {}\n",
            "VoiageEvpiResultV1.struct_size {}\n",
            "VoiageEvpiResultV1.struct_version {}\n",
            "VoiageEvpiResultV1.value {}\n",
            "VoiageEvpiResultV1.sample_count {}\n",
            "VoiageEvpiResultV1.strategy_count {}\n",
            "VoiageEvpiResultV1.has_assurance {}\n",
            "VoiageEvpiResultV1.reserved {}\n",
            "VoiageEvpiResultV1.opportunity_loss_variance {}\n",
            "VoiageEvpiResultV1.monte_carlo_standard_error {}\n",
            "VoiageExpectedLossResultV1 {} {}\n",
            "VoiageExpectedLossResultV1.struct_size {}\n",
            "VoiageExpectedLossResultV1.struct_version {}\n",
            "VoiageExpectedLossResultV1.optimal_strategy_index {}\n",
            "VoiageExpectedLossResultV1.sample_count {}\n",
            "VoiageExpectedLossResultV1.strategy_count {}\n",
            "VoiageExpectedLossResultV1.minimum_expected_opportunity_loss {}\n",
            "VoiageExpectedLossResultV1.has_assurance {}\n",
            "VoiageExpectedLossResultV1.reserved {}\n",
            "VoiageExpectedLossResultV1.opportunity_loss_variance {}\n",
            "VoiageExpectedLossResultV1.monte_carlo_standard_error {}\n",
            "VoiageHandleV1 {} {}\n",
            "voiage_v1_status {} {}",
        ),
        size_of::<VoiageAbiVersionV1>(),
        align_of::<VoiageAbiVersionV1>(),
        offset_of!(VoiageAbiVersionV1, struct_size),
        offset_of!(VoiageAbiVersionV1, abi_major),
        offset_of!(VoiageAbiVersionV1, abi_minor),
        size_of::<VoiageAbiCapabilitiesV1>(),
        align_of::<VoiageAbiCapabilitiesV1>(),
        offset_of!(VoiageAbiCapabilitiesV1, struct_size),
        offset_of!(VoiageAbiCapabilitiesV1, struct_version),
        offset_of!(VoiageAbiCapabilitiesV1, capability_bits),
        size_of::<VoiageEvpiResultV1>(),
        align_of::<VoiageEvpiResultV1>(),
        offset_of!(VoiageEvpiResultV1, struct_size),
        offset_of!(VoiageEvpiResultV1, struct_version),
        offset_of!(VoiageEvpiResultV1, value),
        offset_of!(VoiageEvpiResultV1, sample_count),
        offset_of!(VoiageEvpiResultV1, strategy_count),
        offset_of!(VoiageEvpiResultV1, has_assurance),
        offset_of!(VoiageEvpiResultV1, reserved),
        offset_of!(VoiageEvpiResultV1, opportunity_loss_variance),
        offset_of!(VoiageEvpiResultV1, monte_carlo_standard_error),
        size_of::<VoiageExpectedLossResultV1>(),
        align_of::<VoiageExpectedLossResultV1>(),
        offset_of!(VoiageExpectedLossResultV1, struct_size),
        offset_of!(VoiageExpectedLossResultV1, struct_version),
        offset_of!(VoiageExpectedLossResultV1, optimal_strategy_index),
        offset_of!(VoiageExpectedLossResultV1, sample_count),
        offset_of!(VoiageExpectedLossResultV1, strategy_count),
        offset_of!(
            VoiageExpectedLossResultV1,
            minimum_expected_opportunity_loss
        ),
        offset_of!(VoiageExpectedLossResultV1, has_assurance),
        offset_of!(VoiageExpectedLossResultV1, reserved),
        offset_of!(VoiageExpectedLossResultV1, opportunity_loss_variance),
        offset_of!(VoiageExpectedLossResultV1, monte_carlo_standard_error),
        size_of::<u64>(),
        align_of::<u64>(),
        size_of::<VoiageStatusV1>(),
        align_of::<VoiageStatusV1>(),
    );

    assert_eq!(actual, expected);
}
