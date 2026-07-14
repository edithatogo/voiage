# Track Implementation Plan: Discrete GPU Acceleration

## Phase 1: Establish Evidence Gates [checkpoint: ]

- [x] Task: Reuse the Apple integrated GPU evidence as the gate for discrete
  GPU work.
  - [x] Track remains in feasibility mode while `apple_metal_vs_cpu` status is
    `cpu_reference_only`.
  - [x] Wait for one confirmed Apple Silicon comparison before expansion.
- [x] Task: Wait for the shared HPC acceleration abstraction contract to be
  approved in `hpc-acceleration-abstraction-contract_20260511`.
- [x] Task: Identify the workloads that are likely to scale better on a
  discrete GPU and capture the ranked shortlist in track working notes.
- [x] Task: Conductor - Automated Review And Checkpoint 'Phase 1: Establish
  Evidence Gates' (Protocol in workflow.md)

## Phase 2: Evaluate Backend Candidates [checkpoint: ]

- [x] Task: Define the discrete GPU backend candidates and their deployment
  assumptions.
- [x] Task: Keep the CPU contract and public result shape stable.
- [x] Task: Conductor - Automated Review And Checkpoint 'Phase 2: Evaluate
  Backend Candidates' (Protocol in workflow.md)

## Phase 3: Benchmark And Decide [checkpoint: ]

- [x] Task: Benchmark the discrete GPU path against CPU and integrated GPU
  baselines.
- [x] Task: Decide whether discrete GPU implementation work should continue
  or stay a feasibility track.
  - [x] Current decision: stay in feasibility hold until Apple hardware-backed
    speedup evidence is available and reusable under the shared abstraction.
- [x] Task: Conductor - Automated Review And Checkpoint 'Phase 3: Benchmark
  And Decide' (Protocol in workflow.md)
