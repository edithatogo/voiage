# Track Implementation Plan: Rust Core Migration Foundation

## Phase 1: Migration Shape And Boundary Decision [checkpoint: ]

- [x] Task: Decide the Rust-core migration model.
  - [x] Record Rust as the authoritative execution core for deterministic VOI kernels and shared result contracts.
  - [x] Record Python as the primary façade for CLI, orchestration, plotting, and compatibility wrappers.
  - [x] Record R, Julia, TypeScript, Go, .NET, and other external languages as thin bindings/adapters over the Rust contract.
  - [x] Define which Python components are transitional only and must not become permanent duplicate cores.
- [x] Task: Record the contract boundary.
  - [x] Define which artifacts belong to the Rust core.
  - [x] Define which artifacts remain binding-specific.
  - [x] Define what must remain schema-backed and fixture-backed across the migration.
  - [x] Define the first-crate baseline for the Rust workspace.
- [x] Task: Conductor - Automated Review and Checkpoint 'Migration Shape And Boundary Decision' (Protocol in workflow.md)

## Phase 2: Workspace And Toolchain Policy [checkpoint: ]

- [x] Task: Define the Rust workspace and crate layout.
  - [x] Define the first core crate for numeric kernels, schema types, and deterministic VOI outputs.
  - [x] Define adapter or interop crate boundaries only where a binding needs FFI glue.
  - [x] Define release naming and package naming conventions.
- [x] Task: Define the Rust build and test policy.
  - [x] Define cargo test, fmt, clippy, doc, and package gates.
  - [x] Define release-tag and provenance expectations.
- [x] Task: Conductor - Automated Review and Checkpoint 'Workspace And Toolchain Policy' (Protocol in workflow.md)

## Phase 3: Migration Sequence And Compatibility Policy [checkpoint: ]

- [x] Task: Define the migration order for core modules.
  - [x] Prioritize stable data structures, `ValueArray`-like containers, and deterministic kernels.
  - [x] Move the current Rust-first domain model and scalar contracts before simulation-heavy methods.
  - [x] Move deterministic method families before simulation-heavy methods.
  - [x] Identify what remains in Python during the transition.
  - [x] Identify the first binding-facing compatibility shims.
- [x] Task: Define versioning and compatibility policy.
  - [x] Define semver expectations for the Rust core and bindings.
  - [x] Define how compatibility is declared between Rust core and front-ends.
  - [x] Define which release artifacts are authoritative once Rust owns execution.
- [x] Task: Conductor - Automated Review and Checkpoint 'Migration Sequence And Compatibility Policy' (Protocol in workflow.md)

## Phase 4: Handoff And Roadmap Sync [checkpoint: ]

- [x] Task: Sync the migration decision into the project plan.
  - [x] Update roadmap direction for the Rust-core program.
  - [x] Update backlog items that depend on the decision.
  - [x] Update release/docs references that describe the implementation model.
- [x] Task: Conductor - Automated Review and Checkpoint 'Handoff And Roadmap Sync' (Protocol in workflow.md)
