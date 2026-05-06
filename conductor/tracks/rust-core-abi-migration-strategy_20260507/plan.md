# Track Implementation Plan: Rust Core ABI And Migration Strategy

## Phase 1: Inventory Current API Surfaces [checkpoint: ]

- [ ] Task: List the stable public contracts that must survive the migration.
- [ ] Task: Map which parts of the current repo are core, facade, adapter, or
  packaging concerns.
- [ ] Task: Conductor - Automated Review And Checkpoint 'Phase 1: Inventory
  Current API Surfaces' (Protocol in workflow.md)

## Phase 2: Decide The ABI Strategy [checkpoint: ]

- [ ] Task: Evaluate a narrow C ABI, direct Rust-only integration, and
  language-native FFI options.
- [ ] Task: Decide whether TypeScript should prefer WASM/N-API rather than a
  raw C ABI.
- [ ] Task: Write the ABI recommendation and the reasons for it.
- [ ] Task: Conductor - Automated Review And Checkpoint 'Phase 2: Decide The
  ABI Strategy' (Protocol in workflow.md)

## Phase 3: Define The API-Preserving Migration Path [checkpoint: ]

- [ ] Task: Split the migration into pure Rust modularization and external
  adapter preservation.
- [ ] Task: Define versioning and compatibility rules for the public API
  boundary.
- [ ] Task: Document the adapter policy for Python, R, Julia, TypeScript, Go,
  and .NET.
- [ ] Task: Conductor - Automated Review And Checkpoint 'Phase 3: Define The
  API-Preserving Migration Path' (Protocol in workflow.md)

## Phase 4: Handoff And Future Implementation Gates [checkpoint: ]

- [ ] Task: Define the compatibility tests and ABI round-trip checks that
  future implementation tracks must satisfy.
- [ ] Task: Record which follow-on tracks should be created if the ABI pilot is
  ever justified.
- [ ] Task: Conductor - Automated Review And Checkpoint 'Phase 4: Handoff And
  Future Implementation Gates' (Protocol in workflow.md)
