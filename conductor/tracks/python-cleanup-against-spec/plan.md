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

- [ ] Task: Ensure the Python implementation keeps xarray-labeled data as the core in-memory model for the public contract.
- [ ] Task: Ensure JAX remains an optional acceleration backend rather than a required user-facing execution contract.
- [ ] Task: Align fixture-loading and interchange paths with the chosen Arrow/Parquet and JSON fixture formats.
- [ ] Task: Remove or isolate accidental pandas-specific public assumptions where they conflict with the written contract.

## Phase 4: Prove Compliance and Document Migration [checkpoint: ]

- [ ] Task: Run the Python package against the stable conformance fixtures and record the compliance result.
- [ ] Task: Write migration notes for any public behavior change that users or future bindings need to understand.
- [ ] Task: Update the project docs that describe the Python API so they match the new contract and cleanup outcomes.

## Execution Notes

- Keep the implementation changes subordinate to the written contract and the fixture evidence.
- If a Python behavior cannot be reconciled cleanly in this track, defer it explicitly rather than quietly redefining the contract.
