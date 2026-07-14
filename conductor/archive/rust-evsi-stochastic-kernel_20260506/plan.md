# Track Implementation Plan: Rust EVSI Stochastic Kernel

The Rust EVSI summary envelope is already owned by the Rust core. This track
is kernel-only: it exists to port the stochastic EVSI kernel underneath the
existing contract and keep the summary boundary explicit.

## Phase 1: Kernel Contract Boundary And Fixture Harness [checkpoint: ]

- [x] Task: Define the Rust EVSI kernel input and output contract.
  - [x] Reuse the existing Rust EVSI summary envelope and keep the public
    shape stable while the kernel is added underneath it.
  - [x] Identify the kernel inputs that come from PSA samples, trial design,
    and net-benefit generation.
  - [x] Keep the boundary explicit between summary contract and stochastic
    evaluation, including the pieces already owned by the Rust core.
- [x] Task: Define the shared fixture path for EVSI parity.
  - [x] Add deterministic fixture cases that cover representative EVSI
    workloads.
  - [x] Verify the fixture loader can exercise the Rust kernel directly.
  - [x] Record the expected diagnostics and reporting payloads for parity.
- [x] Task: Conductor - Automated Review And Checkpoint 'Kernel Contract Boundary And Fixture Harness' (Protocol in workflow.md)

## Phase 2: Rust Two-Loop EVSI Kernel [checkpoint: ]

- [x] Task: Port the stochastic two-loop EVSI kernel into Rust.
  - [x] Preserve the current EVSI semantics under fixed seeds and fixtures.
  - [x] Keep the implementation deterministic for the committed test cases.
  - [x] Reuse the Rust domain model and reporting envelopes.
- [x] Task: Add regression tests for the Rust EVSI kernel.
  - [x] Compare the Rust kernel against the committed fixture behavior.
  - [x] Verify tolerance, diagnostics, and result-shape stability.
  - [x] Cover empty, degenerate, and small-sample edge cases.
- [x] Task: Conductor - Automated Review And Checkpoint 'Rust Two-Loop EVSI Kernel' (Protocol in workflow.md)

## Phase 3: Approximation Policy And Optional Kernel Variants [checkpoint: ]

- [x] Task: Decide which approximation variants belong in Rust core.
  - [x] Evaluate whether regression-based, efficient, or moment-based EVSI
    variants should be ported or remain façade-side.
  - [x] Keep any approximation policy explicit in the contract and docs.
  - [x] Preserve compatibility with the current Python EVSI surface.
  - [x] Decision: only the seeded bootstrap kernel is promoted here; other
    approximation variants remain façade-side for now.
- [x] Task: Add parity tests for any approximation variants that land.
  - [x] Verify the committed behavior against fixtures or the Python
    reference.
  - [x] Confirm diagnostics and reporting stay stable across variants.
- [x] Task: Conductor - Automated Review And Checkpoint 'Approximation Policy And Optional Kernel Variants' (Protocol in workflow.md)

## Phase 4: Benchmark Baseline And Handoff [checkpoint: ]

- [x] Task: Record a baseline for the Rust EVSI kernel.
  - [x] Capture representative small, medium, and larger EVSI workloads.
  - [x] Capture machine-readable baseline artifacts for the kernel path.
- [x] Task: Document the EVSI kernel handoff contract.
  - [x] State that the Rust core already owns the EVSI summary envelope,
    diagnostics, reporting, and maturity metadata.
  - [x] State that the new work in this track is the stochastic kernel and the
    parity/benchmark evidence that proves it.
  - [x] State what remains in bindings versus the Rust core after the kernel
    lands.
  - [x] State how future EVSI optimization work should be promoted into core
    once the stochastic kernel is established.
- [x] Task: Conductor - Automated Review And Checkpoint 'Benchmark Baseline And Handoff' (Protocol in workflow.md)
