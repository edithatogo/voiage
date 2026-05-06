# Track Specification: Rust EVSI Stochastic Kernel

## Overview

This track promotes the EVSI sample-information boundary from a Rust-owned
summary contract into a Rust-owned stochastic kernel. The current Rust core
already owns the deterministic EVSI summary envelope; this follow-on track
implements the actual sample-information computation so EVSI can be executed,
benchmarked, and validated in the Rust core rather than only summarized there.

The track is intentionally contract-first and parallel-friendly. It should keep
the EVSI result shape stable, preserve diagnostics and reporting envelopes, and
keep the Python façade thin while the Rust kernel becomes the source of truth
for stochastic EVSI behavior.

## Goals

1. Implement the Rust EVSI stochastic kernel behind the existing summary
   contract, without changing the public EVSI result shape.
2. Preserve the current EVSI envelope, diagnostics, reporting, and maturity
   metadata rules already owned by the Rust core.
3. Validate the Rust kernel against deterministic fixtures and the current
   Python reference behavior.
4. Keep the implementation batch-oriented and parallelization-friendly so it
   can later benefit from Rayon or SIMD without changing the public contract.
5. Document the kernel boundary, approximation policy, and benchmarking
   expectations once the Rust implementation lands.

## Functional Requirements

1. Implement the core EVSI stochastic kernel in Rust.
2. Preserve the current EVSI summary envelope and reporting payload structure
   already exposed by the Rust core.
3. Support the existing deterministic inputs used by the Python EVSI surface,
   including proposed trial design, PSA samples, and model-generated
   net-benefit values.
4. Add fixture-backed parity tests for the Rust EVSI kernel against the Python
   reference behavior.
5. Keep the implementation deterministic under fixed seeds and fixture inputs.
6. Keep the work split into independent slices so the kernel, parity tests, and
   benchmarking can progress in parallel.

## Non-Functional Requirements

1. The Rust implementation must not change the EVSI public result shape.
2. Diagnostics and CHEERS-style reporting must remain populated.
3. The implementation should be suitable for later Rayon/SIMD promotion
   without changing the contract.
4. The track must remain compatible with the core Rust domain model and shared
   fixture system.
5. Approximation or surrogate choices should be explicit and documented rather
   than hidden behind the summary contract.

## Acceptance Criteria

1. A Rust EVSI stochastic kernel exists and is reachable from the Rust core.
2. Fixture-backed parity tests cover the kernel path and validate the current
   result envelope.
3. The Rust output matches the expected EVSI behavior within tolerance on the
   committed fixtures.
4. The contract boundary between the EVSI summary envelope and the stochastic
   kernel is documented clearly, including which pieces were already present
   before this track and which pieces are newly implemented here.
5. The kernel is benchmarkable independently of the Python implementation.

## Out Of Scope

1. Frontier-adjacent VOI methods beyond EVSI.
2. GPU, TPU, or custom-circuit implementation work.
3. Binding packaging changes that are not required by the Rust kernel.
4. New result-shape experiments that would break the stable EVSI contract.
