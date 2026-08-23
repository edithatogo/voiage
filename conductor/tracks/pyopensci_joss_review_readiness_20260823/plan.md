# Track Plan: pyOpenSci & JOSS Review Readiness

## Phases

- [ ] **Phase 1: pyOpenSci Readiness & Maintainer Commitment**
  - [ ] Author `docs/release/pyopensci-readiness.md` with explicit maintenance commitment.
  - [ ] Verify `specs/submission-readiness/pyopensci-evidence.json` via `scripts/validate_submission_readiness.py`.
- [ ] **Phase 2: JOSS Fast-Track Alignment**
  - [ ] Validate `paper.md` against `scripts/validate_joss.py`.
  - [ ] Document fast-track submission procedure in `docs/release/joss-independent-validation.md`.
- [ ] **Phase 3: Automated Verification & Staging**
  - [ ] Run full test suite and quality harness.
  - [ ] Commit, open PR, verify green CI, and merge.
