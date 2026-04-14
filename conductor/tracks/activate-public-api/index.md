# Track: Activate and Wire Up Public API

## Overview
The voiage codebase has feature-complete implementations across all VOI methods (EVPI, EVPPI, EVSI, NMA, structural, adaptive, portfolio, sequential, observational, calibration) and plotting modules. However, the public API is incomplete — key imports are commented out in `__init__.py` files, preventing clean user-facing imports.

## Specification
- **Input:** Existing implementation code with commented-out imports
- **Output:** Fully wired public API with clean imports, passing tests, and top-level re-exports
- **Quality Gates:** 100% of existing tests pass, no import errors, clean `tox` run

## Implementation Plan
See [plan.md](./plan.md)

## Metadata
- **Priority:** 1 (Foundation — must complete before other tracks)
- **Estimated Complexity:** Medium (mostly uncommenting + verification)
- **Dependencies:** None
- **Blocks:** All other tracks
