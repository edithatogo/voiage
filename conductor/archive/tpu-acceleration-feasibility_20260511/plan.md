# Track Implementation Plan: TPU Acceleration Feasibility

## Phase 1: Define TPU Viability Criteria [checkpoint: ]

- [x] Task: Define the workload characteristics that make TPU use plausible.
- [x] Task: Require the existing GPU evidence before TPU work starts.
  - [x] Wait for explicit GPU/discrete acceleration decision from upstream tracks.
  - [x] Current status: hold because upstream tracks are in hardware-evidence prerequisite.
- [x] Task: Require the approved `hpc-acceleration-abstraction-contract_20260511`
  decision before feasibility moves from assessment to implementation.
- [x] Task: Conductor - Automated Review And Checkpoint 'Phase 1: Define TPU
  Viability Criteria' (Protocol in workflow.md)

## Phase 2: Assess Contract Preservation [checkpoint: ]

- [x] Task: Determine whether a TPU path can preserve the current contract
  shape.
  - [x] Decision: only if contract-preserving comparison packets are available.
- [x] Task: Keep CPU fallback behavior authoritative.
- [x] Task: Conductor - Automated Review And Checkpoint 'Phase 2: Assess
  Contract Preservation' (Protocol in workflow.md)

## Phase 3: Decide Feasibility [checkpoint: ]

- [x] Task: Record whether TPU is justified as a follow-on implementation
  track.
  - [x] Current decision: keep TPU in feasibility hold until clear contract-safe
    gains are confirmed.
  - [x] Recorded in `handoff/feasibility_decision.json`.
- [x] Task: Update the roadmap if TPU moves from feasibility to implementation.
  - [x] Current decision recorded in the accelerator working notes; roadmap remains
    in evidence-gated stage.
- [x] Task: Conductor - Automated Review And Checkpoint 'Phase 3: Decide
  Feasibility' (Protocol in workflow.md)
