# Track Specification: Fix Infrastructure and Configuration

## Overview
The repository's development workflow must be trustworthy before any further implementation proceeds. This track stabilizes configuration for testing, linting, typing, packaging metadata, and developer tooling so later tracks can rely on a single reproducible verification path.

## Functional Requirements
1. The project must expose one authoritative validation path through `tox`.
2. Ruff, ty, pytest, and coverage settings must target the actual source and test files instead of excluding large portions of the repository unintentionally.
3. Python-version targets, packaging metadata, and developer instructions must be internally consistent across `pyproject.toml`, `tox.ini`, and contributor-facing documentation.
4. Configuration changes must minimize unrelated application-code churn.

## Non-Functional Requirements
1. The resulting workflow must be non-interactive and CI-safe.
2. Validation failures should be diagnosable from standard tool output without bespoke local setup.
3. Changes must preserve the ability to run the repo on the supported Python floor declared by the package metadata.

## Acceptance Criteria
1. `tox` runs the intended lint, type-check, and test stages without configuration-level blockers.
2. Ruff and ty are pointed at real `voiage/` and `tests/` paths.
3. Pytest no longer silently ignores large test segments due to stale configuration.
4. Contributor-facing setup and verification instructions reflect the active toolchain.

## Out of Scope
1. New scientific methods or public API redesign.
2. Feature-level refactors outside the minimum needed to restore infrastructure correctness.
