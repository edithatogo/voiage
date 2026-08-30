# Phase 2 Review: Signed Staged and Public Release

The immutable release source is `7af563c8cb373057d30662650b3f332f39e05b83`.
This branch changes only release receipts, Conductor records and status prose.
The earlier phase range also includes PR #1039's already validated pnpm update.
No Python, Rust, R or Julia runtime source changes after the signed release.

## Findings and resolutions

- Kept private staging and review receipts historical. Their unpublished flags
  describe those earlier observations; the separate public receipt records the
  later immutable publication and independently checked registry hashes.
- Retained the exact four reviewed distributions through publication. All eight
  GitHub asset hashes, four PyPI hashes and four TestPyPI hashes match review.
- Verified all four PyPI provenance records online. Two parallel local checks
  hit a shared Sigstore TUF-cache race; all four unchanged artifacts passed
  sequential checks. Verification was not disabled or weakened.
- Reconciled the exact-source documentation deployment and tag-bound SBOM job.
  The missing new Software Heritage snapshot remains explicit; no historical
  snapshot is presented as evidence for v2.2.0. No binding tags, conda submission
  or venue submission was created.
- The next phase must refresh citation and submission packets now that public
  evidence exists. Their current prepublication state is not submission-ready.
  R13, R13a and R13b retain those repairs before any external venue action.

No unresolved Critical or High finding was identified in this release phase.
The track remains in progress because venue acceptance criteria are unfinished.

## Verification

- The tagged source passed the full local 15-environment tox gate before merge.
- Publication run 33303294302 passed all 16 jobs, including 4,523 tests,
  15 skips and 95.12 percent coverage; each TestPyPI Python 3.12, 3.13 and 3.14
  environment passed nine installed-wheel tests and four provenance checks.
- A fresh macOS Python 3.12 PyPI installation outside the checkout passed
  dependency validation and an isolated native EVPI smoke test.
- Evidence-only regression command: `uv run pytest
  tests/test_release_2_2_0_candidate.py tests/test_pyopensci_submission_staging.py
  tests/test_final_hardening_closeout.py --no-cov -q`.
- Full Conductor, append-only ledger, GitHub cross-reference and whitespace
  validation passed. The tagged source's executable gates were not replaced by
  these narrower bookkeeping checks.
- Python style guide: not applicable to this evidence-only phase delta.

The retained main-history source boundary is `7af563c8`; validation checkpoint
commits are separate provenance, not assumed ancestors after a squash merge.
