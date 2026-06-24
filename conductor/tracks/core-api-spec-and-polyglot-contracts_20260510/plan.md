# Track Implementation Plan: Core API Spec And Polyglot Contracts

## Phase 1: Lock Down The Contract Surface [checkpoint: ]

- [x] Task: Inventory the remaining core API schema and result contracts
    - [x] Confirm the stable input objects
    - [x] Confirm the stable output payloads
    - [x] Confirm the compatibility aliases that remain intentional
- [x] Task: Normalize the canonical schema artifacts
    - [x] Update any contract text that still implies Python-only behavior
    - [x] Keep the public contract xarray-centered
- [x] Task: Conductor - Automated Review and Checkpoint 'Lock Down The Contract Surface' (Protocol in workflow.md)

## Phase 2: Validate Fixtures And Loader Boundaries [checkpoint: ]

- [x] Task: Finalize fixture-format coverage
    - [x] Keep JSON fixture support explicit
    - [x] Keep Arrow/Parquet dispatch guarded by optional dependencies
- [x] Task: Expand validation tests for contract edge cases
    - [x] Cover mismatch and unsupported-format branches
    - [x] Cover canonical round-trips for the stable surface
- [x] Task: Conductor - Automated Review and Checkpoint 'Validate Fixtures And Loader Boundaries' (Protocol in workflow.md)

## Phase 3: Document The Stable Handoff [checkpoint: ]

- [x] Task: Update user-facing docs for the stable contract
    - [x] Document the canonical schemas
    - [x] Document the fixture format boundary
- [x] Task: Record the polyglot handoff expectations
    - [x] Describe what future bindings must implement
    - [x] Keep the Python binding as the reference implementation
- [x] Task: Conductor - Automated Review and Checkpoint 'Document The Stable Handoff' (Protocol in workflow.md)
