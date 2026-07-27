# Track Implementation Plan: Python Cleanup Against Spec

## Phase 1: Audit Python Against the Written Contract [checkpoint: ]

- [x] Task: Create a spec-to-implementation audit that compares the Python package surface against the new core contract and fixture catalog.
- [x] Task: Classify every discovered mismatch as one of:
  - [x] must-fix for v1 compliance
  - [x] compatibility alias or deprecation path
  - [x] deferred follow-up outside the stable v1 scope
- [x] Task: If any cleanup step requires a tech-stack change, update `conductor/tech-stack.md` before implementation begins.

## Phase 2: Align Public API, Results, and Diagnostics [checkpoint: ]

- **Legacy follow-up (not part of completed track acceptance):** Refactor public imports, result payload shapes, and warning/diagnostic behavior to match the stable contract.
- **Legacy follow-up (not part of completed track acceptance):** Add compatibility shims or deprecation notices where the Python package currently exposes materially different names or structures.
- **Legacy follow-up (not part of completed track acceptance):** Add or update targeted tests for the stable public contract so future drift is caught automatically.

## Phase 3: Align IO and Backend Boundaries [checkpoint: ]

- **Legacy follow-up (not part of completed track acceptance):** Ensure the Python implementation keeps xarray-labeled data as the core in-memory model for the public contract.
- **Legacy follow-up (not part of completed track acceptance):** Ensure JAX remains an optional acceleration backend rather than a required user-facing execution contract.
- **Legacy follow-up (not part of completed track acceptance):** Align fixture-loading and interchange paths with the chosen Arrow/Parquet and JSON fixture formats.
- **Legacy follow-up (not part of completed track acceptance):** Remove or isolate accidental pandas-specific public assumptions where they conflict with the written contract.

## Phase 4: Prove Compliance and Document Migration [checkpoint: ]

- **Legacy follow-up (not part of completed track acceptance):** Run the Python package against the stable conformance fixtures and record the compliance result.
- **Legacy follow-up (not part of completed track acceptance):** Write migration notes for any public behavior change that users or future bindings need to understand.
- **Legacy follow-up (not part of completed track acceptance):** Update the project docs that describe the Python API so they match the new contract and cleanup outcomes.

## Execution Notes

- Keep the implementation changes subordinate to the written contract and the fixture evidence.
- If a Python behavior cannot be reconciled cleanly in this track, defer it explicitly rather than quietly redefining the contract.
