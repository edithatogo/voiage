# Track Specification: HPC Native Acceleration Roadmap

## Overview

This umbrella track defines the path from the current HPC-friendly posture to
an evidence-backed HPC-native claim. It does not add accelerator kernels
directly. Instead, it sequences the staged work so Apple integrated GPU
optimization happens first, followed by broader GPU, TPU, and ASIC feasibility
work.

This track is intentionally downstream of the release and registry submission
program. The repository should finish the release-to-registry pathway first so
the HPC work builds on a stable, fully documented deployment baseline.

Note: the TPU and ASIC feasibility track names below are historical. The
active repository state now treats TPU, FPGA, and ASIC as hardware-dependent
follow-up work with explicit placeholder or archived decision records instead
of open feasibility lanes.

The child tracks are:

1. `apple-metal-integrated-gpu-optimization_20260511`
2. `discrete-gpu-acceleration_20260511`
3. `tpu-acceleration-feasibility_20260511`
4. `asic-acceleration-feasibility_20260511`

## Launch Order

The accelerator work should be approached in this order:

1. Apple Metal integrated GPU optimization
2. Discrete GPU acceleration
3. TPU feasibility
4. ASIC acceleration feasibility

## Functional Requirements

1. Define the acceleration roadmap in a way that keeps CPU fallback behavior
   authoritative.
2. Place Apple Metal / integrated GPU work before the broader accelerator
   tracks.
3. Keep the GPU, TPU, and ASIC work in scope without claiming native status
   prematurely.
4. Update the registry and roadmap docs so the staged progression is explicit.

## Non-Functional Requirements

1. Prefer evidence-backed acceleration over aspirational hardware claims.
2. Keep child-track scopes non-overlapping.
3. Preserve contract stability across every staged accelerator path.

## Acceptance Criteria

1. The roadmap names the integrated GPU, GPU, TPU, and ASIC sequence.
2. Each child track has a distinct purpose and launch order.
3. The roadmap and contract docs agree on the HPC-native path.

## Out of Scope

1. Implementing the accelerator kernels themselves.
2. Claiming HPC-native status before evidence exists.
3. Changing the public result envelope for backend-specific reasons.
