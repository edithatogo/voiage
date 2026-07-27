# rOpenSci local validation

Run `scripts/run_ropensci_pkgcheck.sh` to install rOpenSci `pkgcheck` in a
temporary R library from the review-tools r-universe and inspect the local R
package. The script neither writes to a user R library nor creates a review
issue.

The result is evidence for local package quality only. A self-contained FFI
distribution, maintainer commitment, scope decision, inquiry, review, and
acceptance remain separate gates.

Run `scripts/validate_r_ffi_install.sh` to build the Rust FFI, install
`voiageR` into a temporary R library, and call installed-package `evpi()`.
This makes the current native prerequisite reproducible without pretending it
is a distributed R-package artifact.
