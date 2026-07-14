# Track Specification: Python Cleanup Against Spec

## Overview
After the contract and fixtures exist, the Python package must be brought into alignment with them. This track audits and refactors the Python-side surface so it matches the written API, exposes consistent result objects and diagnostics, and respects the chosen long-term boundaries around xarray, JAX, Arrow, and adapter-only tabular tooling.

## Functional Requirements
1. The Python public API must be reconciled against the canonical spec and conformance fixtures.
2. Contract drift in result shapes, diagnostics, warnings, and naming must be corrected or explicitly deprecated.
3. The Python implementation must preserve xarray-labeled data as the core in-memory model while avoiding accidental pandas- or backend-specific leakage into the public contract.
4. JAX must remain optional acceleration infrastructure rather than the user-facing contract.
5. IO and fixture-consumption paths must align with the interchange formats chosen in the conformance-fixture track.

## Non-Functional Requirements
1. The cleanup must prioritize stable public behavior over internal cleverness.
2. Backwards-incompatible changes must be minimized and documented.
3. The resulting code should be easier for a smaller model to maintain because the public contract is explicit and tested.

## Acceptance Criteria
1. A spec-to-implementation audit exists and all actionable mismatches are resolved or explicitly deferred.
2. The Python package passes the new conformance fixtures for the stable v1 scope.
3. Public imports, result payloads, and diagnostics match the written contract.
4. Migration notes exist for any intentional breaking or behavior-shaping changes.

## Out of Scope
1. Authoring non-Python bindings.
2. New scientific methods beyond what is necessary to comply with the stable v1 contract.
