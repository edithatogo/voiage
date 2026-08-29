# Phase 1 review: current requirements and whole-product gap analysis

## Scope

- Phase: `Phase 1 — Current requirements and whole-product gap analysis`
- Revision range: `5533bdfda8073845bbf1fceffda3ebcb4e521332..bce30f0a0104ba8f8d79d2527a95292ee8964c27`
- Reviewed surfaces: current venue requirements; software and method landscape;
  integration, data, and workflow coverage; structure, API, ABI, and binding
  coverage; dependency frontier; local and hosted performance baselines; and
  Scalene profiles.
- External boundary: no venue, registry, archive, badge, publication, or other
  authenticated submission was performed.

## Findings and remediation

### Medium — local tool identity and timing comparability were ambiguous

The first performance-baseline revision recorded the ambient `pytest` version
rather than the version executed by `uv run`. It also presented hosted focused
unit jobs beside local all-marker runs without stating that their test
selections differ. The baseline now records `pytest 9.1.1` and explicitly
distinguishes the local all-tests selection, hosted focused unit jobs, and the
hosted all-tests Operational Assurance lane.

Status: fixed in the phase-review correction.

### Preserved programme findings

The review did not downgrade or close the findings discovered by the phase.
The audits retain:

- eight integration/data/workflow findings;
- one Critical and six additional structure/API/ABI findings;
- eight dependency/frontier findings;
- seven test/CI performance findings; and
- five profiling experiments requiring disposition.

Those are inputs to Phase 2 rather than defects in the Phase 1 evidence. The
full local suite's eight failures and the xdist-only ninth failure remain
explicit and must be repaired before final assurance.

## Style and platform-guide disposition

- Python style guide: **Pass**. The phase added contract tests only; Ruff check
  and formatting pass, and the tests use typed helpers and module docstrings.
- Markdown and JSON governance artifacts: **Pass**. Focused contract tests parse
  and validate every new machine-readable artifact; diff hygiene passes.
- Platform-specific guides: **Not Applicable**. No manifest-selected cloud,
  browser, mobile, or deployment platform guide intersects these audit and test
  paths.
- Rust, R, and Julia source style guides: **Not Applicable**. The phase measured
  their existing gates but changed no language source.

## Validation

- `uv run pytest tests/test_current_requirements_baseline.py tests/test_voi_landscape_registry.py tests/test_integration_data_workflow_audit.py tests/test_structure_api_abi_audit.py tests/test_dependency_frontier_audit.py tests/test_test_ci_performance_baseline.py --no-cov -q`
  — 21 passed.
- `uv run pytest tests/test_test_ci_performance_baseline.py --no-cov -q` after
  the review correction — 5 passed.
- `python3 /Users/doughnut/.codex/skills/conductor/scripts/validate_conductor.py --root . --mode full --json`
  — valid, zero errors, zero warnings, and one active canonical track.
- `python3 /Users/doughnut/.codex/skills/conductor/scripts/evidence_ledger.py . conductor/tracks/pre_submission_comprehensive_hardening_20260829/evidence.jsonl --against-head`
  — passed.
- `git diff --check` — passed.

The broader baseline also records successful Rust workspace tests, current R
2.1.0 source-package checks, Julia package tests, the repository harness, and
the documentation build. It does not convert the currently failing full Python
suite into a readiness claim.

## Review conclusion

Phase 1 is complete after the correction above. Its purpose was to establish a
current, source-bound and measurable finding set, not to close those findings.
Phase 2 may now disposition the complete ledger and freeze the target
architecture, dependency-promotion policy, and CI optimization design before
implementation begins.
