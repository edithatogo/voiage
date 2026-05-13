# Track Implementation Plan: Core API Spec And Polyglot Contracts

## Phase 1: Lock Down The Contract Surface [checkpoint: ]

- [ ] Task: Inventory the remaining core API schema and result contracts
    - [ ] Confirm the stable input objects
    - [ ] Confirm the stable output payloads
    - [ ] Confirm the compatibility aliases that remain intentional
- [ ] Task: Normalize the canonical schema artifacts
    - [ ] Update any contract text that still implies Python-only behavior
    - [ ] Keep the public contract xarray-centered
- [ ] Task: Conductor - Automated Review and Checkpoint 'Lock Down The Contract Surface' (Protocol in workflow.md)

## Phase 2: Validate Fixtures And Loader Boundaries [checkpoint: ]

- [ ] Task: Finalize fixture-format coverage
    - [ ] Keep JSON fixture support explicit
    - [ ] Keep Arrow/Parquet dispatch guarded by optional dependencies
- [ ] Task: Expand validation tests for contract edge cases
    - [ ] Cover mismatch and unsupported-format branches
    - [ ] Cover canonical round-trips for the stable surface
- [ ] Task: Conductor - Automated Review and Checkpoint 'Validate Fixtures And Loader Boundaries' (Protocol in workflow.md)

## Phase 3: Document The Stable Handoff [checkpoint: ]

- [ ] Task: Update user-facing docs for the stable contract
    - [ ] Document the canonical schemas
    - [ ] Document the fixture format boundary
- [ ] Task: Record the polyglot handoff expectations
    - [ ] Describe what future bindings must implement
    - [ ] Keep the Python binding as the reference implementation
- [ ] Task: Conductor - Automated Review and Checkpoint 'Document The Stable Handoff' (Protocol in workflow.md)
