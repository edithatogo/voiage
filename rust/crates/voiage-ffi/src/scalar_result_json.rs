//! Caller-owned scalar result-envelope JSON transports.

use std::panic;

use voiage_serialization::{
    normalize_enbs_result_json, normalize_evppi_result_json, normalize_evsi_result_json,
};

use crate::{json_transport::normalize_and_copy, VoiageStatusV1};

macro_rules! result_json_export {
    ($name:ident, $normalizer:ident, $description:literal) => {
        #[doc = $description]
        ///
        /// # Safety
        ///
        /// `input` must point to `input_length` readable bytes. `required` must
        /// be non-null, aligned, and writable for one `u64`. A non-zero
        /// `capacity` requires that many writable bytes at `buffer`. Regions
        /// must not overlap, and no pointer is retained.
        #[no_mangle]
        pub unsafe extern "C" fn $name(
            input: *const u8,
            input_length: u64,
            buffer: *mut u8,
            capacity: u64,
            required: *mut u64,
        ) -> VoiageStatusV1 {
            panic::catch_unwind(|| {
                // SAFETY: the public wrapper transfers its caller contract to
                // the shared checked transport.
                unsafe {
                    normalize_and_copy(input, input_length, buffer, capacity, required, $normalizer)
                }
            })
            .unwrap_or(VoiageStatusV1::Panic)
        }
    };
}

result_json_export!(
    voiage_v1_evppi_result_json,
    normalize_evppi_result_json,
    "Validates and normalizes an EVPPI v1 result into caller-owned JSON."
);
result_json_export!(
    voiage_v1_evsi_result_json,
    normalize_evsi_result_json,
    "Validates and normalizes an EVSI v1 result into caller-owned JSON."
);
result_json_export!(
    voiage_v1_enbs_result_json,
    normalize_enbs_result_json,
    "Validates and normalizes an ENBS v1 result into caller-owned JSON."
);
