# Track Plan: R CRAN Standalone Distribution & rOpenSci Review

## Phases

- **Migrated:** **Phase 1: Standalone Build Architecture & CRAN Validation**
  - **Migrated:** Document standalone bridge architecture in `docs/release/voiageR-cran-architecture.md`.
  - **Migrated:** Verify `r-package/voiageR/` package checks.
- **Migrated:** **Phase 2: rOpenSci Statistical Standards & Vignettes**
  - **Migrated:** Map `@srrstats` tags in `r-package/voiageR/R/voiageR.R`.
  - **Migrated:** Update `r-package/voiageR/vignettes/voiageR-getting-started.Rmd`.
- **Migrated:** **Phase 3: Automated Verification & Staging**
  - **Migrated:** Run full test suite and quality harness.
  - **Migrated:** Commit, open PR, verify green CI, and merge.

## Supersession

- [x] Close this source track as superseded after hash-binding and migrating
  every pending task to `pre_submission_comprehensive_hardening_20260829`.

This track is closed as superseded on 2026-08-29. Completed work and evidence remain historical truth. Every pending or in-progress checkbox is hash-bound in `conductor/archive/pre_submission_comprehensive_hardening_20260829/migration-manifest.md` and migrates to that canonical track; no pending item is represented as implemented by this closeout.
