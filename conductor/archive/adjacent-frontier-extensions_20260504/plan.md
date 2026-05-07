# Track Implementation Plan: Adjacent Frontier Extensions

## Phase 1: Causal, Transportability, And External-Validity VOI [checkpoint: ]

- [x] Task: Define the causal-identification, transportability, and
  external-validity VOI contract scope.
  - [x] Model source-to-target population shifts.
  - [x] Model transport weights and validity penalties.
  - [x] Distinguish internal validity, external validity, and
    transportability assumptions.
  - [x] Add the causal-transportability contract scaffold under
    `specs/frontier/causal-transportability/v1/`.
- [x] Task: Define the expected result payloads and diagnostics for the causal
  family.
- [x] Task: Conductor - Automated Review and Checkpoint 'Causal and
  Transportability VOI' (Protocol in workflow.md)

## Phase 2: Data-Quality, Measurement, Privacy, And Linkage VOI [checkpoint: ]

- [x] Task: Define the data-quality, measurement-error, data-acquisition,
  privacy, and linkage VOI contract scope.
  - [x] Model operational acquisition costs.
  - [x] Model privacy-constrained information value.
  - [x] Distinguish source quality, linkage quality, and missingness or
    measurement-error value.
  - [x] Add the data-quality contract scaffold under
    `specs/frontier/data-quality/v1/`.
- [x] Task: Define the expected result payloads and diagnostics for the data
  and privacy family.
- [x] Task: Conductor - Automated Review and Checkpoint 'Data and Privacy VOI'
  (Protocol in workflow.md)

## Phase 3: Computational And Model-Refinement VOI [checkpoint: ]

- [x] Task: Define the computational VOI and value of model refinement contract
  scope.
  - [x] Model compute budget explicitly.
  - [x] Model approximation error explicitly.
  - [x] Distinguish refinement value from implementation cost.
  - [x] Add the computational contract scaffold under
    `specs/frontier/computational/v1/`.
- [x] Task: Define the expected result payloads and diagnostics for the
  computational family.
- [x] Task: Conductor - Automated Review and Checkpoint 'Computational VOI'
  (Protocol in workflow.md)

## Phase 4: Expert-Elicitation And Evidence-Synthesis Design VOI [checkpoint: ]

- [x] Task: Define the expert-elicitation VOI and evidence-synthesis design VOI
  contract scope.
  - [x] Distinguish elicitation design value from downstream decision value.
  - [x] Model evidence-synthesis design choices explicitly.
  - [x] Add the expert-synthesis contract scaffold under
    `specs/frontier/expert-synthesis/v1/`.
- [x] Task: Define the expected result payloads and diagnostics for the expert
  and synthesis family.
- [x] Task: Conductor - Automated Review and Checkpoint 'Expert and Synthesis
  VOI' (Protocol in workflow.md)

## Phase 5: Shared Maturity And Follow-On Handoff [checkpoint: ]

- [x] Task: Define the maturity labels, diagnostics, and reporting metadata
  shared by the adjacent families.
- [x] Task: Define how these families reuse the CHEERS-VOI reporting envelope.
- [x] Task: Define what counts as experimental versus fixture-backed in this
  group.
- [x] Task: Split out any follow-on implementation tracks where the semantics
  warrant it.
  - [x] No additional follow-on implementation tracks are required yet.
- [x] Task: Define the next-step fixture and schema requirements for each
  resulting family.
- [x] Task: Document the handoff path back into the main frontier track.
- [x] Task: Conductor - Automated Review and Checkpoint 'Adjacent Frontier
  Handoff' (Protocol in workflow.md)

## Execution Notes

- This track is intentionally a contract scoping track, not a runtime
  implementation track.
- The goal is to stop these families from remaining an undifferentiated bullet
  in the roadmap.
