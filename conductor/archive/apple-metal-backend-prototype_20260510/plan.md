# Track Implementation Plan: Apple Metal Backend Prototype

## Phase 1: Define The Backend Seam [checkpoint: 2026-05-10T16:28:35Z]

- [x] Task: Write tests that describe the prototype backend contract and CPU
  fallback behavior.
- [x] Task: Define the internal Metal adapter boundary for the committed Apple
  workloads.
- [x] Task: Conductor - Automated Review And Checkpoint 'Phase 1: Define The
  Backend Seam' (Protocol in workflow.md)

## Phase 2: Implement The Metal Prototype [checkpoint: 2026-05-10T22:40:00Z]

- [x] Task: Add the Metal-backed execution path for the committed scalar EVPI
  and memory/throughput workloads.
- [x] Task: Keep the CPU fallback authoritative and preserve the public
  result envelope.
- [x] Task: Add regression tests for device selection, result shape, and
  fallback behavior.
- [x] Task: Conductor - Automated Review And Checkpoint 'Phase 2: Implement
  The Metal Prototype' (Protocol in workflow.md)

## Phase 3: Benchmark And Hand Off [checkpoint: 2026-05-11T17:14:00Z]

- [x] Task: Run the Apple Silicon benchmark comparison against the CPU baseline.
  - [x] Run both committed workloads (scalar EVPI and memory/throughput)
    on Apple Silicon with CPU fallback and Metal-backed execution.
  - [x] Run at least two independent runs per workload and capture both payloads.
  - [x] Persist a comparison bundle with host/runtime metadata from
    `benchmark_mps_vs_cpu` (`runtime`, `review`, `payload_version`,
    `workload`).
- [x] Task: Document the reproducible runtime and packaging requirements for the
  prototype.
  - [x] Record minimum host requirements and feature flags used for Metal
    backend execution.
  - [x] Record dependency and toolchain inputs required to run the
    comparison.
  - [x] Record reproducibility metadata used for artifact generation.
- [x] Task: Define explicit Phase-3 completion criteria and handoff gate.
  - [x] Confirm a CPU contract envelope match with/without Metal path.
  - [x] Confirm benchmark runs are reproducible from deterministic entry
    scripts.
  - [x] Confirm handoff bundle includes evidence, environment notes, and
    next-step optimization questions for the integrated GPU track.
- [x] Task: Hand off the benchmark evidence to the Apple integrated GPU
  optimization track and close loop on what to optimize first.
  - [x] Send the evidence bundle and reproducibility notes to
    `apple-metal-integrated-gpu-optimization_20260511`.
  - [x] Confirm the target workload ordering for the next optimization phase
    from the benchmark outcomes.
- [x] Task: Conductor - Automated Review And Checkpoint 'Phase 3: Benchmark
  And Hand Off' (Protocol in workflow.md)
