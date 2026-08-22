#![no_main]

use libfuzzer_sys::fuzz_target;
use voiage_domain::SampleMatrix;

fuzz_target!(|data: &[u8]| {
    if data.len() < 4 {
        return;
    }

    let rows_count = usize::from(data[0] % 32);
    let cols_count = usize::from(data[1] % 16);
    let payload = &data[2..];

    let mut rows = Vec::with_capacity(rows_count);
    let mut offset = 0;
    for _ in 0..rows_count {
        let mut row = Vec::with_capacity(cols_count);
        for _ in 0..cols_count {
            if offset + 8 <= payload.len() {
                let val = f64::from_le_bytes(payload[offset..offset + 8].try_into().unwrap());
                row.push(val);
                offset += 8;
            } else {
                row.push(0.0);
            }
        }
        rows.push(row);
    }

    // Attempting construction: must succeed if and only if rectangular, non-empty, and all finite
    let matrix_result: Result<SampleMatrix, _> = rows.clone().try_into();
    if let Ok(matrix) = matrix_result {
        let shape = matrix.shape();
        assert!(shape[0] > 0);
        assert!(shape[1] > 0);
        assert_eq!(shape[0], rows.len());
        assert_eq!(shape[1], rows[0].len());
    }
});
