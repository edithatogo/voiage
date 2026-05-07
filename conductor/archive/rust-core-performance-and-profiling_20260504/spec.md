# Track Specification: Rust Core Performance And Profiling

## Overview

This track establishes the profiling and performance-validation layer for a Rust-based core. It is intentionally separate from the numerics implementation so the performance contract can be measured and improved independently.

## Goals

1. Define a scalar CPU baseline for the Rust core.
2. Establish memory and throughput profiling after the scalar baseline is stable.
3. Only then explore Rayon/SIMD parallelization where measurements show a real gain.
4. Treat GPU, TPU, and custom-circuit work as feasibility-only unless benchmark evidence justifies a new track.
5. Produce benchmark artifacts and regression gates that CI can enforce.

## Functional Requirements

1. Provide a scalar CPU profiling baseline for the Rust core.
2. Measure wall time, memory use, and throughput for representative VOI workloads.
3. Define how Rayon or equivalent thread-parallel execution is enabled only after the scalar baseline exists.
4. Evaluate whether GPU, TPU, or custom-circuit backends are worthwhile for the hot kernels.
5. Produce machine-readable benchmark artifacts for regression testing.
6. Define CI regression gates that compare new runs against the recorded baselines.

## Acceptance Criteria

1. The Rust core has a repeatable scalar CPU baseline in place.
2. Memory and throughput profiling are captured after the baseline is stable.
3. Rayon/SIMD work is documented as a measured optimization step, not the first profiling target.
4. The repo documents GPU, TPU, and custom-circuit exploration as feasibility-only unless measurements justify escalation.
5. Benchmark artifacts are machine-readable and suitable for CI regression comparison.
6. Performance regression checks can be run in CI or locally.

## Out Of Scope

1. Hardware-specific optimization beyond feasibility proof and benchmark-backed promotion.
2. Premature custom-circuit work without a clear stable kernel, benchmark signal, and ROI.

## Execution Notes

- Start with scalar CPU measurement, not optimization.
- Treat SIMD, GPU, TPU, and custom-circuit work as later-stage options that require evidence from the baseline and memory/throughput phases.
- Keep the performance contract separate from the algorithm-porting contract.
