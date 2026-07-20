//! Differential coverage for the versioned seeded-bootstrap EVSI contract.

use voiage_domain::SampleMatrix;
use voiage_numerics::evsi_stochastic;

const MIX64: u64 = 0x9E37_79B9_7F4A_7C15;
const STAR_MULTIPLIER: u64 = 0x2545_F491_4F6C_DD1D;

fn matrix(rows: &[&[f64]]) -> SampleMatrix {
    rows.iter()
        .map(|row| row.to_vec())
        .collect::<Vec<_>>()
        .try_into()
        .expect("valid sample matrix")
}

fn reference_next_index(state: &mut u64, sample_count: usize) -> usize {
    if *state == 0 {
        *state = MIX64;
    }
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    let output = (*state).wrapping_mul(STAR_MULTIPLIER);
    usize::try_from(output % sample_count as u64).expect("bounded sample index")
}

#[allow(clippy::cast_precision_loss)]
fn reference_seeded_bootstrap(
    net_benefit: &SampleMatrix,
    trial_sample_size: usize,
    resample_count: usize,
    seed: u64,
) -> (f64, f64, f64, f64) {
    let [sample_count, strategy_count] = net_benefit.shape();
    let draw_count = trial_sample_size.min(sample_count);

    let expected_current_value = (0..strategy_count)
        .map(|strategy| {
            (0..sample_count)
                .map(|row| net_benefit.row(row).unwrap()[strategy])
                .sum::<f64>()
                / sample_count as f64
        })
        .fold(f64::NEG_INFINITY, f64::max);
    let expected_perfect_information = (0..sample_count)
        .map(|row| {
            net_benefit
                .row(row)
                .unwrap()
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max)
        })
        .sum::<f64>()
        / sample_count as f64;

    let expected_sample_value = (0..resample_count)
        .map(|resample| {
            let mut state = seed ^ ((resample as u64 + 1).wrapping_mul(MIX64));
            let mut sums = vec![0.0; strategy_count];
            for _ in 0..draw_count {
                let row = net_benefit
                    .row(reference_next_index(&mut state, sample_count))
                    .unwrap();
                for (strategy, value) in row.iter().copied().enumerate() {
                    sums[strategy] += value;
                }
            }
            sums.into_iter()
                .map(|sum| sum / draw_count as f64)
                .fold(f64::NEG_INFINITY, f64::max)
        })
        .sum::<f64>()
        / resample_count as f64;

    let evsi = (expected_sample_value - expected_current_value).max(0.0);
    (
        expected_current_value,
        expected_sample_value,
        expected_perfect_information,
        evsi,
    )
}

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 1.0e-12,
        "{actual} != {expected}"
    );
}

#[test]
fn seeded_bootstrap_matches_reference_for_multiple_seeds_and_loop_sizes() {
    let net_benefit = matrix(&[&[10.0, 4.0], &[8.0, 6.0], &[6.0, 8.0], &[4.0, 10.0]]);

    for seed in [0, 42, u64::MAX] {
        for (trial_sample_size, resample_count) in [(1, 1), (2, 4), (4, 4), (6, 1)] {
            let actual = evsi_stochastic(&net_benefit, trial_sample_size, resample_count, seed)
                .expect("valid seeded-bootstrap inputs");
            let expected =
                reference_seeded_bootstrap(&net_benefit, trial_sample_size, resample_count, seed);

            assert_eq!(actual.estimator, "seeded_bootstrap");
            assert_eq!(actual.contract_version, 1);
            assert_eq!(actual.draw_count, trial_sample_size.min(4));
            assert_eq!(actual.resample_count, resample_count);
            assert_close(actual.expected_current_value, expected.0);
            assert_close(actual.expected_sample_value, expected.1);
            assert_close(actual.expected_perfect_information, expected.2);
            assert_close(actual.evsi, expected.3);
        }
    }
}

#[test]
fn seeded_bootstrap_is_reproducible_for_extreme_and_zero_seeds() {
    let net_benefit = matrix(&[&[10.0, 4.0], &[8.0, 6.0], &[6.0, 8.0], &[4.0, 10.0]]);

    for seed in [0, 42, u64::MAX] {
        let first =
            evsi_stochastic(&net_benefit, 3, 7, seed).expect("valid seeded-bootstrap inputs");
        let second =
            evsi_stochastic(&net_benefit, 3, 7, seed).expect("valid seeded-bootstrap inputs");
        assert_eq!(first, second);
    }
}

#[test]
fn seeded_bootstrap_returns_zero_for_one_strategy() {
    let net_benefit = matrix(&[&[10.0], &[8.0], &[6.0], &[4.0]]);

    for seed in [0, 42, u64::MAX] {
        let result = evsi_stochastic(&net_benefit, 2, 4, seed).expect("valid one-strategy inputs");
        assert_close(result.evsi, 0.0);
        assert_close(result.expected_sample_value, result.expected_current_value);
    }
}
