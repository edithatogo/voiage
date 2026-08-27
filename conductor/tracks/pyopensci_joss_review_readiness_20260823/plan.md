# Track Plan: pyOpenSci & JOSS Review Readiness

## Phases

- [x] **Phase 1: pyOpenSci Readiness & Maintainer Commitment** [checkpoint: `18461d5`]
  - [x] Author `docs/release/pyopensci-readiness.md` with explicit maintenance commitment. (`dc23f90`)
  - [x] Verify `specs/submission-readiness/pyopensci-evidence.json` via `scripts/validate_submission_readiness.py`. (`6483974`)
- [~] **Phase 2: JOSS Fast-Track Alignment**
  - [x] Validate `paper.md` against `scripts/validate_joss.py`. (`651ba91`)
  - [ ] Document fast-track submission procedure in `docs/release/joss-independent-validation.md`.
- [ ] **Phase 3: Automated Verification & Staging**
  - [ ] Run full test suite and quality harness.
  - [ ] Commit, open PR, verify green CI, and merge.
