# Track: Activate and Wire Up Public API

## Overview
The voiage codebase has feature-complete implementations across all VOI methods but the public API is entirely disabled. `voiage/__init__.py` is empty, `voiage/methods/__init__.py` has all imports commented out, and `voiage/plot/__init__.py` has all imports commented out. Users cannot do `from voiage import evpi` or `from voiage.methods import structural_evpi`. This track wires everything up.

## Specification
- **Input:** Existing implementation code with commented-out imports
- **Output:** Fully wired public API with clean imports, passing tests, and top-level re-exports
- **Quality Gates:** All imports resolve, no circular import errors, `from voiage import ...` works

## Implementation Plan
See [plan.md](./plan.md)

## Metadata
- **Priority:** 2 (Foundation — must complete after Track 1)
- **Estimated Complexity:** Medium (mostly uncommenting + verification)
- **Dependencies:** Track 1 (fix-infrastructure) must complete first
- **Blocks:** Tracks 3, 4, 5
