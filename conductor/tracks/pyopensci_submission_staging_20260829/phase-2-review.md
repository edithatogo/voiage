# Phase 2 review: unposted submission packet

Reviewed on 2026-08-29 against revision range `dd83c0ef..285fcc17`.

## Scope

- the local, visibly unposted pyOpenSci submission draft;
- the hash-bound staging manifest and fail-closed validator;
- candidate, release-receipt, submission-readiness, and JOSS boundaries;
- focused tests, Python quality checks, Vale, Conductor records, and diff
  hygiene; and
- all human-attestation and external-action states.

## Findings and corrections

Two actionable findings were identified and corrected within the permitted
two-loop review budget:

1. The validator originally checked required values without rejecting omitted
   keys. Commit `e0e4a51d` requires the exact human-attestation and
   external-action key sets, with negative regression tests.
2. Markdown hard-break trailing spaces remained in the committed draft.
   Commit `37373672` removed them and rebound the draft digest in the staging
   manifest.

The final review identified no remaining Critical, High, Medium, or actionable
Low findings. The draft recommends `v2.1.0` only for local staging; maintainer
version confirmation remains pending. JOSS remains blocked by its `v2.0.0`
paper and developer-use evidence, the missing non-author evidence, and pending
archive identifiers.

The Python style guide applies to the validator and tests. Their formatting,
lint, and type checks passed. No platform-specific style guide applies to the
remaining Markdown and JSON governance artifacts.

## Validation

The following passed after both corrections were committed:

- 66 focused staging, release-receipt, version, submission-readiness, JOSS,
  Conductor cross-reference, and registry-normalization tests;
- Ruff lint and formatting checks for the validator and tests;
- ty static analysis with zero errors, warnings, or suggestions;
- Vale with warning-level enforcement;
- standalone pyOpenSci staging validation with external actions unperformed;
- submission-readiness validation across 22 targets;
- repository-owned JOSS validation;
- canonical schema-1.0 evidence-ledger validation;
- full Conductor validation across 156 tracks with zero errors and warnings;
- GitHub cross-reference validation; and
- committed-range and worktree Git diff hygiene.

No external contact, survey, issue creation, submission, review, acceptance,
badge, DOI, archive, JOSS referral, pull-request merge, or release was
performed.
