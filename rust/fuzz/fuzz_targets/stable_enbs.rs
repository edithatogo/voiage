#![no_main]

use libfuzzer_sys::fuzz_target;
use voiage_numerics::enbs;

fuzz_target!(|data: &[u8]| {
    if data.len() < 32 {
        return;
    }

    let evsi_per_person = f64::from_le_bytes(data[0..8].try_into().unwrap());
    let sample_size = f64::from_le_bytes(data[8..16].try_into().unwrap());
    let cost_fixed = f64::from_le_bytes(data[16..24].try_into().unwrap());
    let cost_per_sample = f64::from_le_bytes(data[24..32].try_into().unwrap());

    // Only non-negative finite inputs are valid domain for enbs calculation
    if evsi_per_person.is_finite()
        && evsi_per_person >= 0.0
        && sample_size.is_finite()
        && sample_size >= 0.0
        && cost_fixed.is_finite()
        && cost_fixed >= 0.0
        && cost_per_sample.is_finite()
        && cost_per_sample >= 0.0
    {
        if let Ok(result) = enbs(evsi_per_person, sample_size, cost_fixed, cost_per_sample) {
            assert!(result.is_finite());
            let total_cost = cost_fixed + cost_per_sample * sample_size;
            let expected_enbs = evsi_per_person - total_cost;
            assert!((result - expected_enbs).abs() < 1e-9);
        }
    }
});
