# Track Implementation Plan: Roadmap Completion Program

## Phase 1: Define the Remaining Workstreams [checkpoint: ]

- [x] Task: Map the remaining roadmap work into three child tracks
    - [x] Define the contract-focused child track boundary
    - [x] Define the frontier-method follow-through boundary
    - [x] Define the CLI cleanup boundary
- [x] Task: Record the dependency order between the child tracks
    - [x] Prefer core contract work before any new binding expansion
    - [x] Keep CLI cleanup isolated from the contract work
    - [x] Keep frontier work explicit about what is still experimental
- [x] Task: Define the exit criteria for each child track
    - [x] State what completion looks like for the contract track
    - [x] State what completion looks like for the frontier track
    - [x] State what completion looks like for the CLI cleanup track
- [x] Task: Conductor - Automated Review and Checkpoint 'Define The Remaining Workstreams' (Protocol in workflow.md)

## Phase 2: Launch Child Track Scaffolds [checkpoint: ]

- [x] Task: Draft the child-track brief for `core-api-spec-and-polyglot-contracts`
    - [x] Capture the remaining core API / polyglot contract scope
    - [x] Capture the expected fixtures and contract artifacts
- [x] Task: Draft the child-track brief for `frontier-method-followthrough`
    - [x] Capture the remaining phase 7 follow-through scope
    - [x] Separate stable follow-through from experimental expansion
- [x] Task: Draft the child-track brief for `remove-cli-sequential-step-stub`
    - [x] Capture the CLI stub replacement or unsupported-path decision
    - [x] Keep the scope limited to one runtime seam
- [x] Task: Confirm that each child track name is unique in the registry
    - [x] Check the live track registry for collisions
    - [x] Check the archive registry for prior names
- [x] Task: Conductor - Automated Review and Checkpoint 'Launch Child Track Scaffolds' (Protocol in workflow.md)

## Phase 3: Registry And Handoff [checkpoint: ]

- [x] Task: Update the tracks registry to point at the new umbrella program
    - [x] Add the umbrella track entry to the active registry
    - [x] Preserve the existing archived cleanup track entry
- [x] Task: Prepare the handoff path for the next implementation pass
    - [x] Identify which child track should start first
    - [x] Document the next conductor entry point
- [x] Task: Verify that the umbrella program cleanly hands off to the three child tracks
    - [x] Ensure the program has no implementation work left
    - [x] Ensure the child tracks can be created independently afterward
- [x] Task: Conductor - Automated Review and Checkpoint 'Registry And Handoff' (Protocol in workflow.md)
