# Track Implementation Plan: Rust Core Numerics Engine

## Phase 1: Contract Boundary And Fixture Harness [checkpoint: ]

- [x] Task: Map the Rust-facing type groups to the stable core contract
  artifacts.
  - [x] `ValueArray`
  - [x] `ParameterSet`
  - [x] `TrialDesign`
  - [x] diagnostics and reporting envelopes
  - [x] result envelopes
- [x] Task: Define the shared fixture-loading and result-assertion path for the
  Rust core.
  - [x] Verify schema-first envelopes.
  - [x] Verify deterministic numeric tolerances.
  - [x] Verify empty and degenerate cases.
- [x] Task: Conductor - Automated Review And Checkpoint 'Contract Boundary And
  Fixture Harness' (Protocol in workflow.md)

## Phase 2: Deterministic Scalar Methods [checkpoint: ]

- [x] Task: Port the scalar-first deterministic method family in parallel.
  - [x] EVPI
  - [x] ENBS
- [x] Task: Add fixture-backed regression tests for each scalar method.
  - [x] Verify expected values and tolerances.
  - [x] Verify diagnostics and reporting envelopes.
- [x] Task: Conductor - Automated Review And Checkpoint 'Deterministic Scalar
  Methods' (Protocol in workflow.md)

## Phase 3: Partial Information And Summary Methods [checkpoint: ]

- [x] Task: Port the deterministic partial-information and summary families in
  parallel.
  - [x] EVPPI
    - The shared Rust domain model already carries `EvppiSummary`, and the
      Rust core now exposes a deterministic envelope-backed summary contract.
  - [x] CEAF
  - [x] dominance
  - [x] heterogeneity
- [x] Task: Add result-shape and metadata checks for each family.
  - [x] Verify schema-first result envelopes for EVPPI.
  - [x] Verify reporting payload stability for CEAF, dominance, and
    heterogeneity.
- [x] Task: Conductor - Automated Review And Checkpoint 'Partial Information
  And Summary Methods' (Protocol in workflow.md)

## Phase 4: EVSI And Frontier-Adjacent Kernels [checkpoint: ]

- [x] Task: Port the sample-information boundary into the Rust core model.
  - [x] EVSI
    - The Rust domain model and `bindings/rust` already expose the EVSI
      summary contract, and the current core boundary is now explicit: the
      summary/result envelope is owned by Rust core while the stochastic
      kernel remains a later follow-on decision.
    - The track now treats the core boundary itself as the implementable
      contract for this phase, rather than silently implying a deferred kernel
      implementation.
- [x] Task: Add fixture-backed parity tests for each kernel family.
  - [x] Verify numerics within tolerance for any Rust kernels that land.
  - [x] Verify method maturity metadata and handoff expectations for any
    kernel promoted from the frontier track.
- [x] Task: Keep frontier-adjacent methods in the frontier track unless a
  later boundary decision promotes them into Rust core.
  - [x] Value of Perspective already has a Python/frontier surface.
  - [x] Other deterministic frontier-adjacent kernels are tracked outside the
    Rust numerics engine.
- [x] Task: Conductor - Automated Review And Checkpoint 'EVSI And
  Frontier-Adjacent Kernels' (Protocol in workflow.md)

## Phase 5: Benchmark Baseline And Handoff [checkpoint: ]

- [x] Task: Record the scalar benchmark baseline for the Rust numerics engine.
  - [x] Capture representative scalar workloads.
  - [x] Capture the machine-readable scalar baseline artifact.
- [x] Task: Expand the baselines beyond the scalar CPU contract.
  - [x] Capture latency, memory, and throughput baselines.
- [x] Task: Document the engine handoff contract.
  - [x] State that EVPI, ENBS, EVPPI, and the EVSI summary contract already
    belong to the Rust core surface, while the EVSI stochastic kernel remains
    deferred unless a later track promotes it.
  - [x] State what remains in bindings versus the core, including which
    frontier-adjacent methods stay outside the numerics engine.
  - [x] State how future methods should be added and what evidence is required
    before a stochastic kernel is handed off into core.
- [x] Task: Conductor - Automated Review And Checkpoint 'Benchmark Baseline
  And Handoff' (Protocol in workflow.md)
