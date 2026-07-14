# Track Specification: Apple Metal Backend Prototype

## Overview

This track creates a real Metal-backed internal execution prototype for the
committed Apple workloads. Its purpose is to make the Apple integrated GPU
optimization track benchmarkable against a live device-backed path instead of
only a contract-level design.

## Functional Requirements

1. Implement a Metal-backed execution or adapter seam for the committed Apple
   workloads.
2. Keep the CPU fallback path authoritative and contract-preserving.
3. Support the committed scalar EVPI and memory/throughput baseline workloads
   as the first device-backed targets.
4. Provide benchmark hooks so the Apple track can compare device-backed and
   CPU-backed results on Apple Silicon.
5. Document the Apple-specific runtime and packaging requirements needed to
   reproduce the prototype.

## Non-Functional Requirements

1. Preserve the current public API and result envelope.
2. Keep the prototype deterministic and reproducible.
3. Avoid expanding the scope to broader GPU, TPU, or ASIC work.
4. Keep the backend internal rather than introducing a new user-facing API.

## Acceptance Criteria

1. A real Metal-backed path exists for the committed Apple workloads.
2. The CPU fallback continues to pass the same contract checks.
3. Benchmark comparison against the CPU baseline is executable.
4. The benchmark artifact includes reproducible CPU-vs-Metal results for both
   committed workloads, including platform/runtime metadata.
5. A handoff bundle exists for the next optimization track with:
   - workload-level speed/latency and shape evidence,
   - build/packaging requirements,
   - known limitations and open questions.
6. The Apple integrated GPU optimization track can consume the prototype as
   its benchmark target.

## Out of Scope

1. Discrete GPU backends beyond Apple integrated GPUs.
2. TPU or ASIC acceleration.
3. Public API changes to expose the backend directly.
4. Broad optimization work beyond the prototype workloads.
