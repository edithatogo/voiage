# Track: CLI Completion and Integration Testing

## Overview
The CLI has only 5 of ~12 needed commands. EVSI, ENBS, adaptive, portfolio, structural EVPPI, and all plotting commands are missing. Meanwhile, ~100 test files are excluded from the test suite. This track completes the CLI surface and brings all integration tests online.

## Specification
- **Input:** Activated API (Track 2), working infrastructure (Track 1), real implementations (Track 3)
- **Output:** Complete CLI for all methods, all integration tests enabled and passing
- **Quality Gates:** Every method has a CLI command, all tests pass, coverage ≥90%

## Implementation Plan
See [plan.md](./plan.md)

## Metadata
- **Priority:** 4 (Usability and quality)
- **Estimated Complexity:** Medium
- **Dependencies:** Track 1 (infrastructure), Track 2 (API activation). Track 3 methods can be CLI'd as they're implemented.
- **Blocks:** Track 5 (documentation depends on CLI being complete)

## Autonomous Workflow
This track implements the autonomous review-and-progression protocol with safety features:
- **Per-phase review:** `/conductor:review` → apply Critical/High fixes → verify → commit → progress
- **Coverage gate:** Returns to 90% (Track 3 used progressive 80% gate)
- **Escape hatch:** After 2 failed fix attempts, revert phase and mark task as `DEFERRED → v1.1`
- **Rollback checkpoint:** Each phase records commit hash for safe rollback
- **After track completion:** `/conductor:review` (full track) → apply fixes → push → verify CI → archive → auto-progress to Track 5
