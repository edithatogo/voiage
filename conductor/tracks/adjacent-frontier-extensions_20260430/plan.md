# Track Implementation Plan: Adjacent Frontier Extensions

## Phase 1: Causal, Transportability, And External-Validity VOI [checkpoint: ]

- [ ] Task: Define the causal-identification, transportability, and
  external-validity VOI contract scope.
  - [ ] Model source-to-target population shifts.
  - [ ] Model transport weights and validity penalties.
  - [ ] Distinguish internal validity, external validity, and
    transportability assumptions.
- [ ] Task: Define the expected result payloads and diagnostics for the causal
  family.
- [ ] Task: Conductor - Automated Review and Checkpoint 'Causal and
  Transportability VOI' (Protocol in workflow.md)

## Phase 2: Data-Quality, Measurement, Privacy, And Linkage VOI [checkpoint: ]

- [ ] Task: Define the data-quality, measurement-error, data-acquisition,
  privacy, and linkage VOI contract scope.
  - [ ] Model operational acquisition costs.
  - [ ] Model privacy-constrained information value.
  - [ ] Distinguish source quality, linkage quality, and missingness or
    measurement-error value.
- [ ] Task: Define the expected result payloads and diagnostics for the data
  and privacy family.
- [ ] Task: Conductor - Automated Review and Checkpoint 'Data and Privacy VOI'
  (Protocol in workflow.md)

## Phase 3: Computational And Model-Refinement VOI [checkpoint: ]

- [ ] Task: Define the computational VOI and value of model refinement contract
  scope.
  - [ ] Model compute budget explicitly.
  - [ ] Model approximation error explicitly.
  - [ ] Distinguish refinement value from implementation cost.
- [ ] Task: Define the expected result payloads and diagnostics for the
  computational family.
- [ ] Task: Conductor - Automated Review and Checkpoint 'Computational VOI'
  (Protocol in workflow.md)

## Phase 4: Expert-Elicitation And Evidence-Synthesis Design VOI [checkpoint: ]

- [ ] Task: Define the expert-elicitation VOI and evidence-synthesis design VOI
  contract scope.
  - [ ] Distinguish elicitation design value from downstream decision value.
  - [ ] Model evidence-synthesis design choices explicitly.
- [ ] Task: Define the expected result payloads and diagnostics for the expert
  and synthesis family.
- [ ] Task: Conductor - Automated Review and Checkpoint 'Expert and Synthesis
  VOI' (Protocol in workflow.md)

## Phase 5: Shared Maturity And Follow-On Handoff [checkpoint: ]

- [ ] Task: Define the maturity labels, diagnostics, and reporting metadata
  shared by the adjacent families.
- [ ] Task: Define how these families reuse the CHEERS-VOI reporting envelope.
- [ ] Task: Define what counts as experimental versus fixture-backed in this
  group.
- [ ] Task: Split out any follow-on implementation tracks where the semantics
  warrant it.
- [ ] Task: Define the next-step fixture and schema requirements for each
  resulting family.
- [ ] Task: Document the handoff path back into the main frontier track.
- [ ] Task: Conductor - Automated Review and Checkpoint 'Adjacent Frontier
  Handoff' (Protocol in workflow.md)

## Execution Notes

- This track is intentionally a contract scoping track, not a runtime
  implementation track.
- The goal is to stop these families from remaining an undifferentiated bullet
  in the roadmap.
