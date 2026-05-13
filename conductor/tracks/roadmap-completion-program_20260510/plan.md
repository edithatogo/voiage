# Track Implementation Plan: Roadmap Completion Program

## Phase 1: Define the Remaining Workstreams [checkpoint: ]

- [ ] Task: Map the remaining roadmap work into three child tracks
    - [ ] Define the contract-focused child track boundary
    - [ ] Define the frontier-method follow-through boundary
    - [ ] Define the CLI cleanup boundary
- [ ] Task: Record the dependency order between the child tracks
    - [ ] Prefer core contract work before any new binding expansion
    - [ ] Keep CLI cleanup isolated from the contract work
    - [ ] Keep frontier work explicit about what is still experimental
- [ ] Task: Define the exit criteria for each child track
    - [ ] State what completion looks like for the contract track
    - [ ] State what completion looks like for the frontier track
    - [ ] State what completion looks like for the CLI cleanup track
- [ ] Task: Conductor - Automated Review and Checkpoint 'Define The Remaining Workstreams' (Protocol in workflow.md)

## Phase 2: Launch Child Track Scaffolds [checkpoint: ]

- [ ] Task: Draft the child-track brief for `core-api-spec-and-polyglot-contracts`
    - [ ] Capture the remaining core API / polyglot contract scope
    - [ ] Capture the expected fixtures and contract artifacts
- [ ] Task: Draft the child-track brief for `frontier-method-followthrough`
    - [ ] Capture the remaining phase 7 follow-through scope
    - [ ] Separate stable follow-through from experimental expansion
- [ ] Task: Draft the child-track brief for `remove-cli-sequential-step-stub`
    - [ ] Capture the CLI stub replacement or unsupported-path decision
    - [ ] Keep the scope limited to one runtime seam
- [ ] Task: Confirm that each child track name is unique in the registry
    - [ ] Check the live track registry for collisions
    - [ ] Check the archive registry for prior names
- [ ] Task: Conductor - Automated Review and Checkpoint 'Launch Child Track Scaffolds' (Protocol in workflow.md)

## Phase 3: Registry And Handoff [checkpoint: ]

- [ ] Task: Update the tracks registry to point at the new umbrella program
    - [ ] Add the umbrella track entry to the active registry
    - [ ] Preserve the existing archived cleanup track entry
- [ ] Task: Prepare the handoff path for the next implementation pass
    - [ ] Identify which child track should start first
    - [ ] Document the next conductor entry point
- [ ] Task: Verify that the umbrella program cleanly hands off to the three child tracks
    - [ ] Ensure the program has no implementation work left
    - [ ] Ensure the child tracks can be created independently afterward
- [ ] Task: Conductor - Automated Review and Checkpoint 'Registry And Handoff' (Protocol in workflow.md)
