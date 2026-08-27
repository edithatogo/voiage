# Track: R CRAN Standalone Distribution & rOpenSci Statistical Software Review

- [Specification](./spec.md)
- [Implementation Plan](./plan.md)
- [Metadata](./metadata.json)
- [Evidence](./evidence.jsonl)
- [GitHub issue #1024](https://github.com/edithatogo/voiage/issues/1024)
- [Project 28](https://github.com/users/edithatogo/projects/28)
- [Registration PR #1027](https://github.com/edithatogo/voiage/pull/1027)

Status: in progress. Repository implementation remains pending, and CRAN or
rOpenSci submission remains maintainer-controlled.

---

## Objectives
1. Design and document the standalone compilation bridge architecture for `voiageR` (vendored Rust core / `rextendr`).
2. Enforce strict `R CMD check --as-cran` passing with 0 errors, 0 warnings, and 0 notes.
3. Annotate Bayesian sensitivity analysis standards (`@srrstats`) in `r-package/voiageR/`.
4. Prepare rOpenSci review handoff artifact while scheduling submission post-JOSS.
