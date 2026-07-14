# Track Specification: Apple Metal Integrated GPU Optimization

## Overview

This track is the first accelerator step on the path to HPC-native readiness.
It focuses on Apple Silicon integrated GPUs and Metal-backed execution. The
goal is to validate that the project can use integrated GPUs to accelerate
representative workloads without changing the public contract.

## Functional Requirements

1. Identify the workloads that are worth moving onto Apple integrated GPUs.
2. Define the Metal-backed execution path or adapter strategy that preserves
   the current contract shape.
3. Benchmark Apple integrated GPU performance against the scalar CPU baseline.
4. Keep a CPU fallback path available and authoritative.
5. Record the deployment and packaging requirements for Apple platform users.

## Non-Functional Requirements

1. Prefer contract preservation over backend novelty.
2. Keep the implementation deterministic and reproducible.
3. Avoid public API changes just to support Metal.

## Acceptance Criteria

1. Representative workloads are identified for Apple integrated GPU use.
2. Benchmark evidence exists for the CPU-reference path, and the track records a
   concrete deferred follow-up for a device-backed comparison when Apple Silicon
   is available.
3. The CPU fallback remains correct and documented.
4. The track documents a clear path that can justify later discrete GPU work.

## Out of Scope

1. TPU or ASIC work.
2. Discrete GPU backends beyond Apple integrated GPUs.
3. Changing the public result envelope.
