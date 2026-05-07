# Track Implementation Plan: Dynamic Real-Options VOI

## Phase 1: Define The Contract Scope [checkpoint: ]

- [x] Task: Write the dynamic real-options VOI contract document.
  - [x] Record the decision timing, evidence arrival, and exercise rules.
  - [x] Record how delay and irreversibility affect value.
  - [x] Record the maturity and diagnostics fields required of results.
- [x] Task: Define the output payloads for option value and policy-path
  comparison.
- [x] Task: Conductor - Automated Review and Checkpoint 'Dynamic Real-Options
  Contract Scope' (Protocol in workflow.md)

## Phase 2: Add Schemas And Examples [checkpoint: ]

- [x] Task: Author the versioned schema layout for dynamic real-options inputs
  and outputs.
- [x] Task: Split the schema into explicit staged-evidence, exercise-rule, and
  policy-path result shapes.
- [x] Task: Add deterministic example payloads for staged-evidence scenarios.
- [x] Task: Add deterministic example payloads for irreversible-policy
  scenarios.
- [x] Task: Define fixture/provenance metadata for the examples.
- [x] Task: Conductor - Automated Review and Checkpoint 'Dynamic Real-Options
  Schemas' (Protocol in workflow.md)

## Phase 3: Add Deterministic Fixtures And Validation [checkpoint: ]

- [x] Task: Add normative fixtures for representative dynamic real-options
  scenarios.
- [x] Task: Add fixtures that exercise timing sensitivity and lock-in
  comparisons.
- [x] Task: Add deterministic validator coverage for the contract artifacts.
- [x] Task: Document the route to runtime implementation once the contract is
  locked.
- [x] Task: Conductor - Automated Review and Checkpoint 'Dynamic Real-Options
  Fixtures' (Protocol in workflow.md)

## Execution Notes

- Keep the contract separate from the sequential VOI implementation so staged
  timing semantics remain explicit.
- Treat this track as a frontier contract track first and a runtime track only
  after the semantics are stable.
