# Track: Replace Placeholders and Implement Missing Methods

## Overview
Several "complete" methods in the codebase are actually placeholders or degrade to fallbacks. Additionally, several SOTA VOI methods expected in modern libraries are entirely missing. This track replaces placeholder code with real implementations and adds the missing SOTA methods.

## Specification
- **Input:** Activated API from Track 2, working tooling from Track 1
- **Output:** All methods produce real results (no placeholders), plus 4 new SOTA methods implemented
- **Quality Gates:** All methods pass validation against published results or mathematical proofs, coverage ≥90%, strict ruff/mypy clean

## Implementation Plan
See [plan.md](./plan.md)

## Metadata
- **Priority:** 3 (Core functionality)
- **Estimated Complexity:** High (real algorithm implementations)
- **Dependencies:** Track 1 (infrastructure), Track 2 (API activation) — must complete first
- **Blocks:** Track 4 (CLI for new methods), Track 5 (documentation of new methods)

## Autonomous Workflow
This track implements the autonomous review-and-progression protocol:
- After each phase: `/conductor:review` → apply fixes → re-verify → commit → progress
- After track completion: `/conductor:review` (full track) → apply fixes → archive → auto-progress to Track 4 (cli-integration-testing)
