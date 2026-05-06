use std::hint::black_box;
use std::time::Instant;

use serde::Deserialize;
use voiage_core::evpi;

fn baseline_workload() -> Vec<Vec<f64>> {
    vec![vec![10.0, 1.0], vec![2.0, 8.0]]
}

fn run_scalar_baseline() -> f64 {
    evpi(&baseline_workload()).expect("baseline workload should be valid")
}

#[cfg(unix)]
fn current_rss_bytes() -> Option<u64> {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
    let result = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
    if result != 0 {
        return None;
    }
    let usage = unsafe { usage.assume_init() };
    let rss = u64::try_from(usage.ru_maxrss).ok()?;
    #[cfg(target_os = "linux")]
    {
        rss.checked_mul(1024)
    }
    #[cfg(target_os = "macos")]
    {
        Some(rss)
    }
    #[cfg(all(unix, not(any(target_os = "linux", target_os = "macos"))))]
    {
        Some(rss)
    }
}

#[cfg(not(unix))]
fn current_rss_bytes() -> Option<u64> {
    None
}

#[derive(Debug, Deserialize)]
struct MemoryThroughputBaselineArtifact {
    benchmark_name: String,
    metric_type: String,
    workload: MemoryThroughputWorkload,
    samples: Vec<MemoryThroughputSample>,
    summary: MemoryThroughputSummary,
    expected: MemoryThroughputExpected,
    metadata: MemoryThroughputMetadata,
}

#[derive(Debug, Deserialize)]
struct MemoryThroughputWorkload {
    seed: u64,
    repeats: u64,
    warmup_runs: u64,
    net_benefits: Vec<Vec<f64>>,
}

#[derive(Debug, Deserialize)]
struct MemoryThroughputSample {
    phase: String,
    iteration: u64,
    latency_ns: u64,
    throughput_ops_per_sec: f64,
    rss_before_bytes: Option<u64>,
    rss_after_bytes: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct MemoryThroughputSummary {
    evpi: f64,
    cold_start_latency_ns: u64,
    warm_start_latency_ns: u64,
    mean_latency_ns: u64,
    peak_rss_bytes: Option<u64>,
    throughput_ops_per_sec: f64,
}

#[derive(Debug, Deserialize)]
struct MemoryThroughputExpected {
    evpi: f64,
    comparison_rule: String,
    regression_policy: String,
}

#[derive(Debug, Deserialize)]
struct MemoryThroughputMetadata {
    phase: String,
    notes: Vec<String>,
}

#[derive(Debug)]
struct MemoryThroughputSnapshot {
    phase: String,
    iteration: u64,
    latency_ns: u64,
    throughput_ops_per_sec: f64,
    rss_before_bytes: Option<u64>,
    rss_after_bytes: Option<u64>,
}

fn collect_live_snapshots() -> Vec<MemoryThroughputSnapshot> {
    let mut snapshots = Vec::new();
    for (iteration, phase) in ["cold", "warm", "warm"].into_iter().enumerate() {
        let rss_before = current_rss_bytes();
        let start = Instant::now();
        let mut total = 0.0;
        for _ in 0..1_000 {
            total += black_box(run_scalar_baseline());
        }
        let elapsed = start.elapsed();
        let rss_after = current_rss_bytes();
        let latency_ns = (elapsed.as_nanos() / 1_000) as u64;
        let throughput_ops_per_sec = 1_000.0 / elapsed.as_secs_f64();
        assert!(total > 0.0);
        snapshots.push(MemoryThroughputSnapshot {
            phase: phase.to_string(),
            iteration: iteration as u64,
            latency_ns,
            throughput_ops_per_sec,
            rss_before_bytes: rss_before,
            rss_after_bytes: rss_after,
        });
    }
    snapshots
}

fn load_artifact() -> MemoryThroughputBaselineArtifact {
    serde_json::from_str(include_str!("memory_throughput_baseline.json"))
        .expect("baseline artifact should be valid JSON")
}

#[test]
fn memory_throughput_baseline_artifact_matches_contract() {
    let artifact = load_artifact();

    assert_eq!(artifact.benchmark_name, "memory_throughput_baseline");
    assert_eq!(artifact.metric_type, "memory_throughput");
    assert_eq!(artifact.workload.seed, 42);
    assert_eq!(artifact.workload.repeats, 1_000);
    assert_eq!(artifact.workload.warmup_runs, 1);
    assert_eq!(artifact.workload.net_benefits, baseline_workload());
    assert_eq!(artifact.expected.evpi, 3.0);
    assert_eq!(artifact.expected.comparison_rule, "shape-and-range");
    assert_eq!(artifact.expected.regression_policy, "ci-contract-only");
    assert_eq!(
        artifact.metadata.phase,
        "phase-2-memory-throughput-measurement"
    );
    assert!(!artifact.metadata.notes.is_empty());

    assert_eq!(artifact.samples.len(), 3);
    assert_eq!(artifact.samples[0].phase, "cold");
    assert_eq!(artifact.samples[0].iteration, 0);
    assert_eq!(artifact.samples[1].phase, "warm");
    assert_eq!(artifact.samples[1].iteration, 1);
    assert!(artifact.samples[0].rss_before_bytes.is_some());
    assert!(artifact.samples[0].rss_after_bytes.is_some());
    assert!(artifact.samples.iter().all(|sample| sample.latency_ns > 0));
    assert!(artifact
        .samples
        .iter()
        .all(|sample| sample.throughput_ops_per_sec > 0.0));
    assert!(artifact.summary.evpi > 0.0);
    assert!(artifact.summary.cold_start_latency_ns > 0);
    assert!(artifact.summary.warm_start_latency_ns > 0);
    assert!(artifact.summary.mean_latency_ns > 0);
    assert!(artifact.summary.peak_rss_bytes.is_some());
    assert!(artifact.summary.throughput_ops_per_sec > 0.0);
}

#[test]
fn memory_throughput_live_samples_are_deterministic_and_positive() {
    let snapshots = collect_live_snapshots();
    assert_eq!(snapshots.len(), 3);

    for snapshot in &snapshots {
        assert!(!snapshot.phase.is_empty());
        assert!(snapshot.iteration < 3);
        assert!(snapshot.latency_ns > 0);
        assert!(snapshot.throughput_ops_per_sec > 0.0);
        #[cfg(unix)]
        {
            assert!(snapshot.rss_before_bytes.is_some());
            assert!(snapshot.rss_after_bytes.is_some());
        }
        #[cfg(not(unix))]
        {
            assert!(snapshot.rss_before_bytes.is_none());
            assert!(snapshot.rss_after_bytes.is_none());
        }
    }

    let cold = &snapshots[0];
    let warm = &snapshots[1];
    assert_eq!(cold.phase, "cold");
    assert_eq!(warm.phase, "warm");
    assert!(cold.rss_before_bytes.is_some());
    assert!(cold.rss_after_bytes.is_some());
    assert!(warm.rss_before_bytes.is_some());
    assert!(warm.rss_after_bytes.is_some());
    assert!(cold.latency_ns > 0);
    assert!(warm.latency_ns > 0);
}
