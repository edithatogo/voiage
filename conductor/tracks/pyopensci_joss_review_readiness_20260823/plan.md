# Track Plan: pyOpenSci & JOSS Review Readiness

## Phases

- [x] **Phase 1: pyOpenSci Readiness & Maintainer Commitment** [checkpoint: `18461d5`]
  - [x] Author `docs/release/pyopensci-readiness.md` with explicit maintenance commitment. (`dc23f90`)
  - [x] Verify `specs/submission-readiness/pyopensci-evidence.json` via `scripts/validate_submission_readiness.py`. (`6483974`)
- [x] **Phase 2: JOSS Fast-Track Alignment** [checkpoint: `2d9062d`]
  - [x] Validate `paper.md` against `scripts/validate_joss.py`. (`651ba91`)
  - [x] Document fast-track submission procedure in `docs/release/joss-independent-validation.md`. (`4a303a0`)
  - [x] **Review Fixes:** Enforce the selected partner route and unperformed external states in the JOSS manifest contract. (`dd4588d`)
- [x] **Phase 3: Automated Verification & Staging** [checkpoint: `c790745`]
  - [x] Run full test suite and quality harness. (`19273d6`)
  - [x] Commit, open PR, verify green CI, and merge. (`1c1eeca`)
