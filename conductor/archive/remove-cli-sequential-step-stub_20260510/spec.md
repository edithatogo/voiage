# Track Specification: Remove CLI Sequential Step Stub

## Overview

This track removes the explicit sequential-step stub in the CLI surface. The
result should be either a real implementation path or an explicit unsupported
path with a documented contract, rather than a hidden placeholder.

## Functional Requirements

1. Remove the named CLI stub as an implementation detail.
2. Decide whether the CLI should call a real step model or reject the path
   explicitly.
3. Keep the user-facing behavior deterministic and documented.
4. Add tests that lock in the chosen behavior.
5. Ensure the CLI behavior remains consistent with the rest of the sequential
   VOI contract.

## Non-Functional Requirements

1. Keep the change limited to the CLI sequential surface.
2. Prefer the smallest behavior change that removes the stub.
3. Avoid broad refactors unrelated to the stub cleanup.

## Acceptance Criteria

1. The named stub is no longer the implementation boundary.
2. The CLI behavior is covered by focused tests.
3. The result is documented in user-facing notes if behavior changes.

## Out of Scope

1. Reworking unrelated CLI commands.
2. Frontier-method expansion.
3. Core contract or fixture changes.
