# Review Report: Remaining Backlog Delivery B6.2 Polars providers

## Summary

The source declaration slice is ready for protected delivery; native qualification and the full track remain incomplete.

## Verification Checks

- [x] **Plan Compliance**: Partial — exact frontend/runtime sources and declaration checks are implemented; compiler/bootstrap, Arrow and native execution gates remain open.
- [x] **Style Compliance**: Pass — affected Ruff checks and full local lint pass.
- [x] **New Tests**: Yes — source identity, missing inputs, dependency/compiler/default drift, incomplete pairs and conditional declarations have negative controls.
- [x] **Test Coverage**: Partial — source/recipe behavior is tested and both actual Spack classes load, but native installation, numerical/Arrow/module and compiler-matrix tests are unexecuted.
- [x] **Test Results**: Passed — 45 focused tests and all 15 local tox environments; 5,465 full-suite tests passed, 19 skipped, 95% package coverage. Resource guard completed below its memory/time limits.

Applied review fixes reject empty provider sets, a mismatched compiler executable path, an unversioned SDK substituted into both recipe and packet, and dependencies hidden behind false conditions. Both authentic upstream archives pass after those fixes. No native acceptance is inferred.

The private catalogue is pinned to its observed commit. A native range probe shows the existing core compiler constraint can also admit the dated nightly; explicit compiler role separation remains required in the next slice. OrbStack is now running, but its existing containers were left unchanged and physical free space remains below the retained EasyBuild floor.
