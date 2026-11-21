// SPDX-License-Identifier: Apache-2.0

use std::env;
use std::fs;
use std::hint::black_box;
use std::time::Instant;

use voiage_domain::SampleMatrix;
use voiage_numerics::evpi;

fn main() {
    let args: Vec<_> = env::args().collect();
    assert_eq!(
        args.len(),
        6,
        "usage: rust_evpi FILE ROWS COLS WARMUPS REPETITIONS"
    );
    let rows: usize = args[2].parse().expect("rows");
    let cols: usize = args[3].parse().expect("columns");
    let warmups: usize = args[4].parse().expect("warmups");
    let repetitions: usize = args[5].parse().expect("repetitions");

    let setup = Instant::now();
    let bytes = fs::read(&args[1]).expect("read little-endian float64 fixture");
    assert_eq!(bytes.len(), rows * cols * 8);
    let values: Vec<Vec<f64>> = bytes
        .chunks_exact(cols * 8)
        .map(|row| {
            row.chunks_exact(8)
                .map(|cell| f64::from_le_bytes(cell.try_into().expect("8 bytes per cell")))
                .collect()
        })
        .collect();
    let matrix = SampleMatrix::try_from(values).expect("valid fixture");
    let setup_ns = setup.elapsed().as_nanos();

    let started = Instant::now();
    let value = black_box(evpi(black_box(&matrix)).expect("canonical EVPI"));
    let cold_ns = started.elapsed().as_nanos();
    for _ in 0..warmups {
        black_box(evpi(black_box(&matrix)).expect("warmup EVPI"));
    }
    let warm_ns: Vec<_> = (0..repetitions)
        .map(|_| {
            let started = Instant::now();
            black_box(evpi(black_box(&matrix)).expect("warm EVPI"));
            started.elapsed().as_nanos()
        })
        .collect();
    println!(
        "{{\"value\":{value:.17e},\"setup_ns\":{setup_ns},\"cold_ns\":{cold_ns},\"warm_ns\":{warm_ns:?}}}"
    );
}
