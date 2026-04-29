# Track Specification: Activate and Wire Up Public API

## Overview
`voiage` already contains substantial method and plotting implementations, but the public import surface is effectively disabled. This track turns the existing code into a coherent, importable public API without silently exporting unstable internals.

## Functional Requirements
1. Top-level package exports in `voiage/__init__.py` must expose the intended public classes and core methods.
2. Subpackage exports for `voiage.methods`, `voiage.plot`, and any retained `voiage.core` utilities must resolve cleanly.
3. Public imports must avoid circular-import failures and provide clear behavior when optional dependencies are missing.
4. The exported surface must be explicit via `__all__` where appropriate.

## Non-Functional Requirements
1. API activation must preserve backwards compatibility where reasonable for already-documented symbols.
2. The public surface should stay intentionally small and stable enough to support future cross-language alignment.
3. Import-time behavior must remain fast and side-effect-light.

## Acceptance Criteria
1. `from voiage import ...` works for the agreed core public symbols.
2. `from voiage.methods import ...` and `from voiage.plot import ...` work for supported exports.
3. Dedicated import tests cover the activated public surface.
4. Ruff, ty, and the relevant test suite pass after the export wiring changes.

## Out of Scope
1. Designing the future language-agnostic contract itself.
2. Implementing missing scientific methods not already present in the codebase.
