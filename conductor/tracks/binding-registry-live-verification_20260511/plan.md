# Track Implementation Plan: Binding Registry Live Verification

## Phase 1: Define the Registry Evidence Contract [checkpoint: ]

- [x] Task: Create a machine-readable registry evidence schema for all language bindings.
  - [x] include package name, registry URL, check timestamp, check status, confidence, and evidence URL.
- [x] Task: Add the schema file under `docs/release/registry_audit_snapshot.json`.
- [x] Task: Conductor - Automated Review And Checkpoint 'Phase 1: Define the Registry Evidence Contract' (Protocol in workflow.md)

## Phase 2: Record Live Status for Each Binding [checkpoint: ]

- [x] Task: Record current live status for Python artifacts and publishable channels.
  - [ ] Keep external conda-forge/CRAN/Julia feedstock/Jupyter tasks as `external_manual` until verified.
- [x] Task: Record current live status for TypeScript and Rust package channels.
- [x] Task: Record current live status for Go module proxy and .NET NuGet channels.
- [x] Task: Add per-binding owners/next-step notes for each unresolved item.
- [x] Task: Conductor - Automated Review And Checkpoint 'Phase 2: Record Live Status for Each Binding' (Protocol in workflow in workflow.md)

## Phase 3: Keep Registry Bookkeeping Obvious [checkpoint: ]

- [x] Task: Update the publication docs and checklist to include the new evidence format.
- [x] Task: Add a short "how to refresh audit evidence" section to the release docs.
- [x] Task: Track unresolved registry confirmations as explicit follow-up, not implicit success.
- [x] Task: Conductor - Automated Review And Checkpoint 'Phase 3: Keep Registry Bookkeeping Obvious' (Protocol in workflow.md)
