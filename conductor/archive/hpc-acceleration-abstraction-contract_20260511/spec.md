# Track Specification: HPC Acceleration Abstraction Contract

## Overview

The HPC roadmap currently moves from Apple Metal toward broader GPU, TPU, and ASIC
consideration. This track makes the hardware plan safe and coherent by defining a
single abstraction contract for future accelerator paths.

The goal is to avoid fragmented backend implementations and ensure every
accelerator track evaluates against the same compute contract, benchmark payload,
and fallback behavior.

## Functional Requirements

1. Evaluate a shared acceleration strategy that covers:
   - Apple Metal (integrated GPU),
   - CUDA/ROCm-style discrete GPU,
   - TPU,
   - and ASIC/custom circuits as a feasibility boundary.
2. Compare candidate stacks that can cover multiple targets (for example
   NumPy-compatible IR paths and XLA-style graphs) against project constraints.
3. Define a target matrix for each hardware class including supported kernel shapes,
   transfer overhead tolerance, and compile/runtime expectations.
4. Require deterministic contract-preservation checkpoints before any new backend leaves
   experimental scope.
5. Require CPU fallback as the authoritative reference for all new HPC paths.

## Non-Functional Requirements

1. Keep the public contract unchanged.
2. Avoid framework lock-in by capturing interface and data-shape expectations at the track level.
3. Prefer portable and reproducible evidence over single-vendor optimization.

## Acceptance Criteria

1. A single acceleration abstraction decision is recorded for next-phase GPU/TPU/FPGA/ASIC work.
2. The decision is explicit about when and why to keep current NumPy/JAX/Metal paths.
3. Discrete GPU, TPU, and ASIC track contracts reference the same benchmark and fallback rules as Apple Metal.
4. The roadmap is updated so standardization is a prerequisite, not an afterthought.

## Out of Scope

1. Immediate implementation of discrete GPU/TPU/FPGA/ASIC kernels.
2. Changing the existing public APIs solely to fit backend availability.
3. In-repo credentialed registry publication actions.
