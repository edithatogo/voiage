# Track Implementation Plan: Rust Core Performance And Profiling

## Phase 1: Scalar CPU Baseline Profiling Contract [checkpoint: ]

- [x] Task: Define the Rust profiling metrics and workload matrix.
  - [x] scalar CPU wall time
  - [x] cold-start vs warm-start behavior
  - [x] representative VOI workloads with fixed seeds and inputs
- [x] Task: Add baseline profiling artifacts.
  - [x] Choose a machine-readable artifact format.
  - [x] Define the seed and workload metadata.
- [x] Task: Add local scalar CPU benchmark scaffolding.
  - [x] Provide a repeatable EVPI workload in `benches/`.
  - [x] Document the local benchmark entrypoint and expected result.
- [x] Task: Define the CI regression gate for the scalar baseline.
  - [x] Set the comparison rule against the recorded baseline.
  - [x] Define the failure threshold and reporting format.
  - [x] CI runs `cargo test --locked --benches scalar_cpu_baseline -- --nocapture`.
  - [x] The committed JSON artifact must match the benchmark name, metric type, seed, repeat count, workload matrix, expected EVPI, and regression policy.
  - [x] Timing is logged and observed locally, but threshold enforcement is deferred until a stable historical baseline exists.
- [x] Task: Conductor - Automated Review And Checkpoint 'Baseline Profiling Contract' (Protocol in workflow.md)

## Phase 2: Memory And Throughput Measurement [checkpoint: ]

- [x] Task: Add memory profiling coverage.
  - [x] Measure allocations and RSS/heap behavior.
  - [x] Record representative peak-memory cases.
- [x] Task: Add throughput profiling coverage.
  - [x] Measure analyses per second on the baseline workloads.
  - [x] Record scaling behavior for larger input sets.
- [x] Task: Capture cold-start versus warm-start memory behavior.
  - [x] Capture memory deltas between cold and warm runs.
  - [x] Keep the artifact schema aligned with Phase 1.
- [x] Task: Conductor - Automated Review And Checkpoint 'Memory And Throughput Measurement' (Protocol in workflow.md)

## Phase 3: Rayon, SIMD, And Parallel CPU Feasibility [checkpoint: ]

- [x] Task: Define the parallel CPU feasibility policy.
  - [x] Use the scalar baseline and memory/throughput artifacts to identify the hot paths that justify Rayon or an equivalent multithreading layer.
  - [x] Capture the concurrency model, safety constraints, and determinism rules for any parallel variant.
  - [x] State which workloads are eligible for parallel execution and which remain scalar-only by contract.
- [x] Task: Capture SIMD feasibility against the baseline workload.
  - [x] Compare scalar versus SIMD-enabled variants on the same deterministic workload.
  - [x] Record only the cases where SIMD is a measurable and repeatable win.
  - [x] Treat SIMD as an internal optimization step, not a new public contract.
- [x] Task: Conductor - Automated Review And Checkpoint 'Rayon, SIMD, And Parallel CPU Feasibility' (Protocol in workflow.md)

## Phase 4: GPU, TPU, And Custom-Circuit Feasibility Only [checkpoint: ]

- [x] Task: Evaluate accelerator feasibility only after the CPU evidence is strong.
  - [x] Determine which VOI workloads are plausibly GPU-bound from the scalar, memory, and throughput baseline data.
  - [x] Only prototype accelerator kernels if the measured CPU-path evidence justifies a separate follow-on track.
- [x] Task: State the accelerator non-goals and escalation criteria.
  - [x] Document the conditions under which a GPU, TPU, or custom-circuit approach would make sense.
  - [x] State that accelerator work is out of scope when the control flow is branchy, the workload is small, or the ROI is not benchmark-backed.
  - [x] Treat these efforts as feasibility-only until a later track is explicitly opened.
- [x] Task: Conductor - Automated Review And Checkpoint 'GPU, TPU, And Custom-Circuit Feasibility Only' (Protocol in workflow.md)

## Phase 5: Benchmark CI And Regression Gates [checkpoint: ]

- [x] Task: Wire Rust performance checks into CI.
  - [x] Add benchmark or profiling jobs.
  - [x] Publish the benchmark artifacts from CI.
  - [x] Compare each run to the recorded scalar baseline contract.
  - [x] Emit a machine-readable Rust profiling artifact from the same CI job.
  - [x] Capture wall time and maximum RSS in the artifact without enforcing a brittle threshold.
  - [x] Keep timing thresholds deferred until a stable historical baseline exists.
- [x] Task: Sync the performance docs and guidance.
  - [x] Document the profiling workflow in the developer guide.
  - [x] Document how to read the performance artifacts.
  - [x] Document the criteria for promoting SIMD or accelerator work into a follow-on track.
  - [x] Covered by `docs/developer_guide/profiling.rst` and the scalar baseline scaffold in `bindings/rust/benches/`.
- [x] Task: Conductor - Automated Review And Checkpoint 'Benchmark CI And Regression Gates' (Protocol in workflow.md)
