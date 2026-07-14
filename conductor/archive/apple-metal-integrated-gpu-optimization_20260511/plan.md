# Track Implementation Plan: Apple Metal Integrated GPU Optimization

## Phase 1: Baseline The Apple Workloads [checkpoint: ]

- [x] Task: Identify the representative VOI workloads that are plausible
  candidates for Apple integrated GPU acceleration.
- [x] Task: Record the scalar CPU baseline for those workloads.
- [x] Task: Conductor - Automated Review And Checkpoint 'Phase 1: Baseline The
  Apple Workloads' (Protocol in workflow.md)

## Phase 2: Design The Metal Path [checkpoint: ]

- [x] Task: Define the Metal-backed execution or adapter strategy.
- [x] Task: Keep the CPU fallback path contract-preserving.
- [x] Task: Conductor - Automated Review And Checkpoint 'Phase 2: Design The
  Metal Path' (Protocol in workflow.md)

## Phase 3: Benchmark And Validate [checkpoint: ]

- [x] Task: Compare the Apple integrated GPU path against the CPU baseline.
  - [x] CPU-reference comparison packet is reproducible on every environment and
    confirms workload and runtime fingerprints.
  - [x] Document device-backed comparison requirement as `apple_metal` availability
    depends on macOS + MPS runtime.
  - [x] Record CPU-reference comparison artifacts now and document the exact
    follow-up requirement when Apple Silicon hardware is unavailable.
  - [x] Record a reviewable deferred item for one future `apple_metal` payload
    capture when Apple Silicon hardware is available.
- [x] Task: Document the reproducible deployment and packaging requirements.
- [x] Task: Conductor - Automated Review And Checkpoint 'Phase 3: Benchmark
  And Validate' (Protocol in workflow.md), with explicit note that full
  device-backed speedup evidence is deferred until Apple Silicon availability.
