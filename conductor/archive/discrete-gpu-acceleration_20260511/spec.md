# Track Specification: Discrete GPU Acceleration

## Overview

This track extends the accelerator roadmap beyond Apple integrated GPUs. It
assesses the best-fit discrete GPU backends and the workload families that are
large and regular enough to justify them.

## Functional Requirements

1. Use the integrated GPU results as the evidence gate for moving to discrete
   GPUs.
2. Identify which workloads are plausibly discrete-GPU-bound.
3. Define the backend candidates and their contract-preserving requirements.
4. Benchmark the discrete GPU path against the CPU and integrated GPU baselines.

## Non-Functional Requirements

1. Preserve portability and reproducibility.
2. Avoid a backend-specific public API.
3. Keep the CPU path as the reference contract.

## Acceptance Criteria

1. The discrete GPU candidates are ranked by fit.
2. Benchmark evidence shows whether discrete GPU work is justified.
3. The contract stays stable across CPU, integrated GPU, and discrete GPU
   paths.

## Out of Scope

1. TPU and ASIC work.
2. Changing the public result envelope.
3. Treating the discrete GPU layer as mandatory for all users.
