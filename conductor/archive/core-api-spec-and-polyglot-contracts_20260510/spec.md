# Track Specification: Core API Spec And Polyglot Contracts

## Overview

This track completes the remaining core API and polyglot contract work. It
focuses on the machine-readable contract surface, the canonical schemas, and the
fixture format boundaries that future bindings must obey.

## Functional Requirements

1. Finalize the machine-readable definitions for the stable core API surface.
2. Ensure the canonical schemas describe the stable inputs and outputs without
   depending on Python-specific implementation details.
3. Align fixture-loading and validation behavior with the stable contract and
   format rules.
4. Preserve the xarray-centered Python data model while keeping the contract
   language-agnostic.
5. Keep the contract and fixtures ready for future non-Python bindings.

## Non-Functional Requirements

1. Keep the contract deterministic and easy to validate.
2. Avoid introducing runtime behavior changes unrelated to the contract.
3. Prefer explicit schemas and fixtures over implicit conventions.

## Acceptance Criteria

1. The contract artifacts define the stable surface without ambiguity.
2. The fixture and validation paths accept the documented formats only.
3. The Python implementation remains conformant with the core contract.
4. The work is test-backed and documented.

## Out of Scope

1. Implementing external bindings.
2. Frontier-method expansion beyond the stable contract surface.
3. CLI cleanup unrelated to the contract boundary.
