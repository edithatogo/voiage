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

## Autonomous Workflow
This track implements the autonomous review-and-progression protocol:
- After each phase: `/conductor:review` → apply fixes → re-verify → commit → progress
- After track completion: `/conductor:review` (full track) → apply fixes → archive → auto-progress to Track 3 (implement-missing-methods)

## GitHub traceability

- Track issue: [#337](https://github.com/edithatogo/voiage/issues/337)
- Parent issue: https://github.com/edithatogo/vop_poc_nz/issues/29
- Project: [VOP–VOIAGE Conductor Roadmap](https://github.com/users/edithatogo/projects/28)
- Pull requests: No pull request proven by recorded commit or track-path history.
