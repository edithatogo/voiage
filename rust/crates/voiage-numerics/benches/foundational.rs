//! Reproducible smoke benchmarks for the deterministic numerical kernels.

use std::hint::black_box;
use std::time::Instant;

use voiage_domain::{SampleCube, SampleMatrix, SampleVector};
use voiage_numerics::{
    ceaf, dominance, enbs, evpi, evppi, evsi_efficient_linear, evsi_moment_based, evsi_stochastic,
    expected_loss, net_benefit, WtpMode,
};

#[allow(clippy::too_many_lines)]
fn main() {
    benchmark("net_benefit", || {
        let costs = (0..32_768).map(f64::from).collect::<Vec<_>>();
        let effects = (0..32_768)
            .map(|value| f64::from(value) / 100.0)
            .collect::<Vec<_>>();
        black_box(
            net_benefit(&costs, &effects, &[50_000.0], WtpMode::Scalar)
                .expect("net-benefit benchmark"),
        );
    });
    benchmark("evpi", || {
        let rows = (0..4096)
            .map(|sample| {
                (0..8)
                    .map(|strategy| f64::from((sample + strategy * 17) % 101))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let values: SampleMatrix = rows.try_into().expect("benchmark matrix");
        black_box(evpi(&values).expect("EVPI benchmark"));
    });
    benchmark("enbs", || {
        black_box(enbs(12.5, 5.0).expect("ENBS benchmark"));
    });
    benchmark("expected_loss", || {
        let rows = (0..4096)
            .map(|sample| {
                (0..8)
                    .map(|strategy| f64::from((sample + strategy * 17) % 101))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let values: SampleMatrix = rows.try_into().expect("benchmark matrix");
        black_box(expected_loss(&values).expect("expected-loss benchmark"));
    });
    benchmark("evppi", || {
        let net_benefit: SampleMatrix = (0..2048)
            .map(|sample| {
                (0..4)
                    .map(|strategy| f64::from((sample + strategy * 17) % 101))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
            .try_into()
            .expect("EVPPI net-benefit matrix");
        let parameters: SampleMatrix = (0..2048)
            .map(|sample| vec![f64::from(sample % 101), f64::from((sample * 3) % 97)])
            .collect::<Vec<_>>()
            .try_into()
            .expect("EVPPI parameter matrix");
        black_box(evppi(&net_benefit, &parameters).expect("EVPPI benchmark"));
    });
    benchmark("evsi", || {
        let values: SampleMatrix = (0..2048)
            .map(|sample| {
                (0..4)
                    .map(|strategy| f64::from((sample + strategy * 17) % 101))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
            .try_into()
            .expect("EVSI net-benefit matrix");
        black_box(evsi_stochastic(&values, 32, 64, 42).expect("EVSI benchmark"));
    });
    benchmark("evsi_efficient_linear", || {
        let values: SampleMatrix = (0..512)
            .map(|sample| {
                (0..4)
                    .map(|strategy| f64::from((sample + strategy * 17) % 101))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
            .try_into()
            .expect("efficient-linear EVSI net-benefit matrix");
        let parameters: SampleMatrix = (0..512)
            .map(|sample| vec![f64::from(sample % 101), f64::from((sample * 3) % 97)])
            .collect::<Vec<_>>()
            .try_into()
            .expect("efficient-linear EVSI parameter matrix");
        black_box(
            evsi_efficient_linear(&values, &parameters, 32)
                .expect("efficient-linear EVSI benchmark"),
        );
    });
    benchmark("evsi_moment_based", || {
        let values: SampleMatrix = (0..512)
            .map(|sample| {
                (0..4)
                    .map(|strategy| f64::from((sample + strategy * 17) % 101))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
            .try_into()
            .expect("moment-based EVSI net-benefit matrix");
        let parameters: SampleMatrix = (0..512)
            .map(|sample| vec![f64::from(sample % 101), f64::from((sample * 3) % 97)])
            .collect::<Vec<_>>()
            .try_into()
            .expect("moment-based EVSI parameter matrix");
        black_box(
            evsi_moment_based(&values, &parameters, 32).expect("moment-based EVSI benchmark"),
        );
    });
    benchmark("dominance", || {
        let costs: SampleVector = (0..128)
            .map(f64::from)
            .collect::<Vec<_>>()
            .try_into()
            .expect("benchmark costs");
        let effects: SampleVector = (0..128)
            .map(|index| f64::from((index * 3) % 97))
            .collect::<Vec<_>>()
            .try_into()
            .expect("benchmark effects");
        black_box(dominance(&costs, &effects).expect("dominance benchmark"));
    });
    benchmark("ceaf", || {
        let values: SampleCube = (0..2048)
            .map(|sample| {
                (0..4)
                    .map(|strategy| {
                        (0..8)
                            .map(|threshold| {
                                f64::from((sample + strategy * 11 + threshold * 7) % 101)
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
            .try_into()
            .expect("benchmark cube");
        let thresholds: SampleVector = (0..8)
            .map(f64::from)
            .collect::<Vec<_>>()
            .try_into()
            .expect("benchmark thresholds");
        black_box(ceaf(&values, &thresholds, 0.95).expect("CEAF benchmark"));
    });
}

fn benchmark(name: &str, operation: impl FnOnce()) {
    let started = Instant::now();
    operation();
    println!(
        "benchmark={name},elapsed_ns={}",
        started.elapsed().as_nanos()
    );
}
