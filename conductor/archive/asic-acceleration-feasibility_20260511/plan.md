# Track Implementation Plan: ASIC Acceleration Feasibility

## Phase 1: Define ASIC Viability Criteria [checkpoint: ]

- [x] Task: Define the workload characteristics that make ASIC work plausible.
- [x] Task: Require TPU- or GPU-level evidence before ASIC evaluation starts.
  - [x] Wait for the GPU acceleration decision and TPU feasibility outcome from
    upstream tracks.
  - [x] Current status remains blocked by unresolved upstream hardware evidence.
- [x] Task: Require the approved `hpc-acceleration-abstraction-contract_20260511`
  contract before moving from feasibility to implementation planning.
- [x] Task: Conductor - Automated Review And Checkpoint 'Phase 1: Define ASIC
  Viability Criteria' (Protocol in workflow.md)

## Phase 2: Assess Contract Risk [checkpoint: ]

- [x] Task: Determine whether an ASIC path can preserve the current contract
  shape.
- [x] Task: Keep CPU fallback behavior authoritative.
- [x] Task: Conductor - Automated Review And Checkpoint 'Phase 2: Assess
  Contract Risk' (Protocol in workflow.md)

## Phase 3: Decide Feasibility [checkpoint: ]

- [x] Task: Record whether ASIC work is justified as a follow-on track.
  - [x] Current decision: retain in feasibility-only mode pending sustained upstream gains.
  - [x] Recorded in `handoff/feasibility_decision.json`.
- [x] Task: Update the roadmap if ASIC moves beyond feasibility.
  - [x] Roadmap remains at feasibility stage with explicit no-implementation hold.
- [x] Task: Conductor - Automated Review And Checkpoint 'Phase 3: Decide
  Feasibility' (Protocol in workflow.md)
