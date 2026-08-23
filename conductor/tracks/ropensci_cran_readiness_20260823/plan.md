# Track Plan: R CRAN Standalone Distribution & rOpenSci Review

## Phases

- [ ] **Phase 1: Standalone Build Architecture & CRAN Validation**
  - [ ] Document standalone bridge architecture in `docs/release/voiageR-cran-architecture.md`.
  - [ ] Verify `r-package/voiageR/` package checks.
- [ ] **Phase 2: rOpenSci Statistical Standards & Vignettes**
  - [ ] Map `@srrstats` tags in `r-package/voiageR/R/voiageR.R`.
  - [ ] Update `r-package/voiageR/vignettes/voiageR-getting-started.Rmd`.
- [ ] **Phase 3: Automated Verification & Staging**
  - [ ] Run full test suite and quality harness.
  - [ ] Commit, open PR, verify green CI, and merge.
