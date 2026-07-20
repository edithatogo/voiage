//! Thread-safety and determinism coverage for the native EVSI kernels.

use std::thread;

use voiage_domain::SampleMatrix;
use voiage_numerics::{
    evsi_efficient_linear, evsi_moment_based, evsi_stochastic, EvsiApproximationResult,
    EvsiKernelResult,
};

fn matrix(rows: &[&[f64]]) -> SampleMatrix {
    rows.iter()
        .map(|row| row.to_vec())
        .collect::<Vec<_>>()
        .try_into()
        .expect("valid sample matrix")
}

#[derive(Clone, Debug, PartialEq)]
struct EvsiResults {
    seeded: EvsiKernelResult,
    efficient_linear: EvsiApproximationResult,
    moment_based: EvsiApproximationResult,
}

fn compute_results(net_benefit: &SampleMatrix, parameters: &SampleMatrix) -> EvsiResults {
    EvsiResults {
        seeded: evsi_stochastic(net_benefit, 3, 7, 42).expect("valid seeded-bootstrap inputs"),
        efficient_linear: evsi_efficient_linear(net_benefit, parameters, 3)
            .expect("valid efficient-linear inputs"),
        moment_based: evsi_moment_based(net_benefit, parameters, 3)
            .expect("valid moment-based inputs"),
    }
}

#[test]
fn native_evsi_kernels_are_thread_safe_and_deterministic() {
    let net_benefit = matrix(&[
        &[10.0, 4.0],
        &[8.0, 6.0],
        &[6.0, 8.0],
        &[4.0, 10.0],
        &[7.0, 5.0],
        &[5.0, 7.0],
    ]);
    let parameters = matrix(&[&[-1.0], &[0.0], &[1.0], &[2.0], &[3.0], &[4.0]]);
    let expected = compute_results(&net_benefit, &parameters);

    thread::scope(|scope| {
        let workers = (0..8)
            .map(|_| scope.spawn(|| compute_results(&net_benefit, &parameters)))
            .collect::<Vec<_>>();

        for worker in workers {
            let actual = worker.join().expect("native EVSI worker must not panic");
            assert_eq!(actual, expected);
        }
    });
}
