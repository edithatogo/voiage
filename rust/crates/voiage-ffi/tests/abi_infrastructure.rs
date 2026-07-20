//! Contract tests for the portable, infrastructure-only v1 C ABI.

use std::mem::{align_of, offset_of, size_of};

use voiage_ffi::{
    voiage_v1_abi_version, voiage_v1_capabilities, VoiageAbiCapabilitiesV1, VoiageAbiVersionV1,
    VoiageStatusV1, VOIAGE_ABI_CAPABILITY_QUERY, VOIAGE_ABI_VERSION_NEGOTIATION,
    VOIAGE_V1_ABI_MAJOR, VOIAGE_V1_ABI_MINOR,
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
fn capability_query_advertises_infrastructure_only() {
    let capabilities = voiage_v1_capabilities();

    assert_eq!(size_of::<VoiageAbiCapabilitiesV1>(), 16);
    assert_eq!(capabilities.struct_size, 16);
    assert_eq!(capabilities.struct_version, 1);
    assert_eq!(
        capabilities.capability_bits,
        VOIAGE_ABI_VERSION_NEGOTIATION | VOIAGE_ABI_CAPABILITY_QUERY
    );
    assert_eq!(capabilities.capability_bits & !0b11, 0);
}

#[test]
fn public_queries_have_the_exact_namespaced_function_signatures() {
    let version_query: extern "C" fn() -> VoiageAbiVersionV1 = voiage_v1_abi_version;
    let capability_query: extern "C" fn() -> VoiageAbiCapabilitiesV1 = voiage_v1_capabilities;

    let _ = (version_query, capability_query);
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
        size_of::<u64>(),
        align_of::<u64>(),
        size_of::<VoiageStatusV1>(),
        align_of::<VoiageStatusV1>(),
    );

    assert_eq!(actual, expected);
}
