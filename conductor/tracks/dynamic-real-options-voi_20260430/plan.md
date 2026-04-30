# Track Implementation Plan: Dynamic Real-Options VOI

## Phase 1: Define The Contract Scope [checkpoint: ]

- [ ] Task: Write the dynamic real-options VOI contract document.
  - [ ] Record the decision timing, evidence arrival, and exercise rules.
  - [ ] Record how delay and irreversibility affect value.
  - [ ] Record the maturity and diagnostics fields required of results.
- [ ] Task: Define the output payloads for option value and policy-path
  comparison.
- [ ] Task: Conductor - Automated Review and Checkpoint 'Dynamic Real-Options
  Contract Scope' (Protocol in workflow.md)

## Phase 2: Add Schemas And Examples [checkpoint: ]

- [ ] Task: Author the versioned schema layout for dynamic real-options inputs
  and outputs.
- [ ] Task: Split the schema into explicit staged-evidence, exercise-rule, and
  policy-path result shapes.
- [ ] Task: Add deterministic example payloads for staged-evidence scenarios.
- [ ] Task: Add deterministic example payloads for irreversible-policy
  scenarios.
- [ ] Task: Define fixture/provenance metadata for the examples.
- [ ] Task: Conductor - Automated Review and Checkpoint 'Dynamic Real-Options
  Schemas' (Protocol in workflow.md)

## Phase 3: Add Deterministic Fixtures And Validation [checkpoint: ]

- [ ] Task: Add normative fixtures for representative dynamic real-options
  scenarios.
- [ ] Task: Add fixtures that exercise timing sensitivity and lock-in
  comparisons.
- [ ] Task: Add deterministic validator coverage for the contract artifacts.
- [ ] Task: Document the route to runtime implementation once the contract is
  locked.
- [ ] Task: Conductor - Automated Review and Checkpoint 'Dynamic Real-Options
  Fixtures' (Protocol in workflow.md)

## Execution Notes

- Keep the contract separate from the sequential VOI implementation so staged
  timing semantics remain explicit.
- Treat this track as a frontier contract track first and a runtime track only
  after the semantics are stable.
