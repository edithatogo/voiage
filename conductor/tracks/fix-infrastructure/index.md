# Track: Fix Infrastructure and Configuration

## Overview
The project has severe configuration issues that prevent reliable testing, linting, and development. `pyproject.toml` excludes all Python files from ruff linting, ignores ~100 test files from pytest, has inconsistent Python version targets, and contains placeholder author/URL metadata. These must be fixed before any other track can produce reliable results.

## Specification
- **Input:** Broken `pyproject.toml`, stale `todo.md`, inconsistent metadata
- **Output:** Clean configuration where tests run, linting works, type checking is configured correctly
- **Quality Gates:** `ruff check voiage/` produces output (not excluded), `pytest` runs without mass-ignoring files, version targets are consistent

## Implementation Plan
See [plan.md](./plan.md)

## Metadata
- **Priority:** 1 (Critical blocker — must complete first)
- **Estimated Complexity:** Low (configuration-only changes)
- **Dependencies:** None
- **Blocks:** All other tracks (unreliable config = unreliable results)

## Autonomous Workflow
This track implements the autonomous review-and-progression protocol:
- After each phase: `/conductor:review` → apply fixes → re-verify → commit → progress
- After track completion: `/conductor:review` (full track) → apply fixes → archive → auto-progress to Track 2 (activate-public-api)
