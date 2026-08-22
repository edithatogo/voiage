#![no_main]

use libfuzzer_sys::fuzz_target;
use voiage_numerics::enbs;

fuzz_target!(|data: &[u8]| {
    if data.len() < 16 {
        return;
    }

    let evsi_result = f64::from_le_bytes(data[0..8].try_into().unwrap());
    let research_cost = f64::from_le_bytes(data[8..16].try_into().unwrap());

    if evsi_result.is_finite() && evsi_result >= 0.0 && research_cost.is_finite() && research_cost >= 0.0 {
        if let Ok(result) = enbs(evsi_result, research_cost) {
            assert!(result.is_finite());
            let expected_enbs = evsi_result - research_cost;
            assert!((result - expected_enbs).abs() < 1e-9);
        }
    }
});
