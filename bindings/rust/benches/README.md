# Rust Scalar CPU Baseline

This directory holds the lightweight local benchmark scaffold for the Rust
core performance track.

The current baseline is intentionally scalar and deterministic:

- workload: a fixed two-strategy EVPI matrix
- expected result: `3.0`
- metric shape: wall-clock timing around repeated scalar execution
- artifact: `scalar_cpu_baseline.json`
- regression rule: exact workload/value match in CI, timing threshold deferred

Recommended local verification:

```bash
cargo test --benches scalar_cpu_baseline -- --nocapture
```

The benchmark source is kept simple on purpose so the performance track can
record a baseline before introducing Criterion, CI thresholds, or parallel
variants. The next steps are to add machine-readable timing artifacts and then
promote regression gates once the baseline stabilizes.
