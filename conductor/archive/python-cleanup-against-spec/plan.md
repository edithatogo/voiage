# Track Implementation Plan: Python Cleanup Against Spec

## Phase 1: Audit Python Against the Written Contract [checkpoint: ]

- [x] Task: Create a spec-to-implementation audit that compares the Python package surface against the new core contract and fixture catalog.
- [x] Task: Classify every discovered mismatch as one of:
  - [x] must-fix for v1 compliance
  - [x] compatibility alias or deprecation path
  - [x] deferred follow-up outside the stable v1 scope
- [x] Task: If any cleanup step requires a tech-stack change, update `conductor/tech-stack.md` before implementation begins.

## Phase 2: Align Public API, Results, and Diagnostics [checkpoint: ]

- [x] Task: Refactor public imports, result payload shapes, and warning/diagnostic behavior to match the stable contract.
  * Confirmed the curated export surface, stable result/reporting payloads, and explicit EVPPI compatibility warning path with focused regression tests.
- [x] Task: Add compatibility shims or deprecation notices where the Python package currently exposes materially different names or structures.
- [x] Task: Add or update targeted tests for the stable public contract so future drift is caught automatically.

## Phase 3: Align IO and Backend Boundaries [checkpoint: ]

- [x] Task: Ensure the Python implementation keeps xarray-labeled data as the core in-memory model for the public contract.
  * Added canonical dataset round-trip helpers and deep-copy preservation in the schema layer.
- [x] Task: Ensure JAX remains an optional acceleration backend rather than a required user-facing execution contract.
  * Verified through the existing backend fallback coverage and the focused backend sanity check.
- [x] Task: Align fixture-loading and interchange paths with the chosen Arrow/Parquet and JSON fixture formats.
  * Added suffix-based fixture loading for JSON now and optional Arrow/Parquet dispatch for future binary artifacts.
- [x] Task: Remove or isolate accidental pandas-specific public assumptions where they conflict with the written contract.
  * The public core-contract surface remains xarray-backed; the audit found no blocking pandas leakage to remove.

## Phase 4: Prove Compliance and Document Migration [checkpoint: ]

- [x] Task: Run the Python package against the stable conformance fixtures and record the compliance result.
  * Verified the stable core-contract slice with the focused fixture and schema test run.
- [x] Task: Write migration notes for any public behavior change that users or future bindings need to understand.
  * Documented the raw-dict EVPPI compatibility alias, the xarray dataset round-trip helpers, and the fixture-format boundary.
- [x] Task: Update the project docs that describe the Python API so they match the new contract and cleanup outcomes.
  * Updated the data-structures, backend, and migration-guide docs to reflect the xarray-centric contract and optional JAX boundary.

## Execution Notes

- Keep the implementation changes subordinate to the written contract and the fixture evidence.
- If a Python behavior cannot be reconciled cleanly in this track, defer it explicitly rather than quietly redefining the contract.
