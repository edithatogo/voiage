# Phase 2 review: disposition and architecture freeze

## Scope

- Phase: `Phase 2 — Disposition and architecture freeze`
- Revision range: `12f48b6c..74604ed5`
- Reviewed surfaces: canonical finding disposition; Rust/Python ownership;
  stable API and C ABI evolution; installed R and Julia packaging targets;
  dependency promotion and rollback; and CI, test, cache, artifact, sharding,
  coverage, profiling, and cancellation policy.
- External boundary: no venue, registry, archive, badge, publication, release,
  or other authenticated submission was performed.

## Findings and remediation

### Medium — profiling experiment identifiers were shifted

The first CI-design revision assigned the intended profiling remedies to
`PROF-002` through `PROF-004`, omitted `PROF-005`, and used `PROF-001` for a
general Scalene limitation rather than the source experiment. The source
profile defines `PROF-001` as lazy schema imports, `PROF-002` as tracked-file
harness traversal, `PROF-003` as bootstrap optimization, `PROF-004` as safe
subprocess consolidation, and `PROF-005` as bounded parallel pytest.

Status: fixed during phase review. The design now preserves all five source
identities, and its contract derives the expected identifiers directly from
the performance baseline and Scalene profile to prevent future omission.

### Preserved programme findings

This phase froze dispositions and implementation rules; it did not claim that
the underlying repository findings were repaired. Every `must_fix` item
remains open for Phases 3 through 6, including the eight full-suite failures,
the xdist-only order-dependent failure, standalone R packaging, installed
binding behavior, and the measured performance bottlenecks.

Preview dependencies and accelerators remain non-authoritative until their
candidate gates pass. A focused or change-selected lane cannot replace the
full main, scheduled-assurance, or release validation paths.

## Style and platform-guide disposition

- Python style guide: **Pass**. The four policy contract modules pass Ruff
  lint and formatting checks.
- JSON and Markdown governance artifacts: **Pass**. Machine-readable policies
  are parsed by focused tests and diff hygiene passes.
- Rust, R, Julia, workflow, and runtime source guides: **Not Applicable**.
  Phase 2 froze targets but changed no executable implementation or workflow.

## Validation

- `uv run pytest tests/test_canonical_finding_ledger.py tests/test_target_architecture_freeze.py tests/test_dependency_promotion_policy.py tests/test_ci_optimization_design.py --no-cov -q`
  — 20 passed.
- `uv run ruff check` on the four Phase 2 contract modules — passed.
- `uv run ruff format --check` on the four Phase 2 contract modules — passed.
- `python3 /Users/doughnut/.codex/skills/conductor/scripts/validate_conductor.py --root . --mode full --json`
  — valid, zero errors, zero warnings, and one active canonical track.
- `python3 /Users/doughnut/.codex/skills/conductor/scripts/evidence_ledger.py . conductor/tracks/pre_submission_comprehensive_hardening_20260829/evidence.jsonl --against-head`
  — passed.
- `git diff --check` — passed.

## Review conclusion

Phase 2 is complete after the identifier correction. The repository now has
one source-bound finding ledger and explicit architecture, dependency, and CI
targets. Phase 3 may begin contract-first repair; no repository-readiness or
submission-readiness claim is made at this checkpoint.
