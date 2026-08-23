# Track: R CRAN Standalone Distribution & rOpenSci Statistical Software Review

**Track ID:** `ropensci_cran_readiness_20260823`  
**GitHub Issue:** [#1024](https://github.com/edithatogo/voiage/issues/1024)  
**Status:** In Progress  
**Specification:** [`spec.md`](./spec.md)  
**Execution Plan:** [`plan.md`](./plan.md)  

---

## Objectives
1. Design and document the standalone compilation bridge architecture for `voiageR` (vendored Rust core / `rextendr`).
2. Enforce strict `R CMD check --as-cran` passing with 0 errors, 0 warnings, and 0 notes.
3. Annotate Bayesian sensitivity analysis standards (`@srrstats`) in `r-package/voiageR/`.
4. Prepare rOpenSci review handoff artifact while scheduling submission post-JOSS.
