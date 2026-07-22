#![no_main]

use libfuzzer_sys::fuzz_target;
use voiage_domain::SampleMatrix;
use voiage_numerics::evpi;

fuzz_target!(|data: &[u8]| {
    if data.len() < 2 {
        return;
    }

    let sample_count = usize::from(data[0] % 16) + 1;
    let strategy_count = usize::from(data[1] % 8) + 1;
    let payload = &data[2..];
    if payload.is_empty() {
        return;
    }

    let mut rows = Vec::with_capacity(sample_count);
    for sample in 0..sample_count {
        let mut row = Vec::with_capacity(strategy_count);
        for strategy in 0..strategy_count {
            let offset = (sample * strategy_count + strategy) * 2;
            let high = payload[offset % payload.len()];
            let low = payload[(offset + 1) % payload.len()];
            row.push(f64::from(i16::from_be_bytes([high, low])) / 16.0);
        }
        rows.push(row);
    }

    let matrix: SampleMatrix = rows.try_into().expect("bounded rectangular matrix");
    let result = evpi(&matrix).expect("validated finite matrix");
    assert!(result.is_finite());
    assert!(result >= 0.0);
});
