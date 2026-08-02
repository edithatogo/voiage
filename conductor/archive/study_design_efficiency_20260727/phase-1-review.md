# Phase 1 automated review

Review date: 2026-07-31. Scope: commits after baseline `25722c32` through the
Phase 1 contract review.

## Result

Pass after one remediation loop. No Critical or High findings remain.

## Findings and remediation

- **Moderate — sample-size domain was underspecified.** The contract described
  sample size as finite and non-negative but did not require an integer. It now
  matches the intended participant-count and Rust `u64` domain.
- **Moderate — marginal standard errors could invite invalid ENBS uncertainty.**
  The design input now permits a directly estimated ENBS standard error and
  interval. The result contract continues to prohibit synthesizing ENBS
  uncertainty from marginal EVSI and cost errors without covariance.
- **No compatibility regression.** Stable scalar ENBS, its closed result
  schema, the legacy clinical optimizer, existing plotting helper and CLI
  remain unchanged. Experimental contracts are separate.
- **No authorization or security finding.** The phase contains governance
  documents only, no secrets, external publication, deployment or dependency
  changes.

## Acceptance review

| Criterion | Result | Evidence |
| --- | --- | --- |
| AC-01 | Pass | Common context, commensurability, costs and feasibility are frozen in `contract.md`. |
| AC-02 | Pass | The exact COSS envelope, ties, boundary, uncertainty and plotting data are frozen. |
| AC-03 | Pass | The unclamped ratio, zero EVPI and tolerance-aware bounds behavior are frozen. |
| AC-04 | Pass | `legacy-audit.md` classifies every adjacent stable and heuristic surface. |
| AC-08 | Pass | Registry, issue #571, parent #318, programme #313 and Project 28 links remain intact. |

## Validation

- `uv run --extra ci --extra dev pytest tests/test_conductor_github_cross_references.py tests/test_conductor_followthrough_tracks.py tests/test_conductor_registry_normalization.py tests/test_v1_programme_baseline.py --no-cov -q`: 25 passed; two upstream xarray deprecation warnings.
- `python3 /Users/doughnut/.codex/skills/conductor/scripts/validate_conductor.py --root . --mode full --json`: 143 tracks, zero errors, zero warnings.
- `git diff --check 25722c32..HEAD`: passed.

The Python style guide is not applicable because this phase changes no Python
source. No manifest-selected platform guide applies to the documentation-only
change set.

Scientific review remains pending before stable promotion. Hosted checks,
merge, release and issue closure remain separate gates.
