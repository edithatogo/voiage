# rOpenSci native system-dependency gate

`voiageR` uses the versioned Rust `voiage-ffi` shared library for its direct
EVPI path. The repository demonstrates a clean build, R installation, and
installed-package numerical smoke test in `scripts/validate_r_ffi_install.sh`.
That is repeatable checkout evidence, not evidence that the rOpenSci package
checking environment can provide the native library.

rOpenSci's [author guide](https://devguide.ropensci.org/softwarereview_author.html)
states that packages with unusual system dependencies may need a pull request
to its base checking Dockerfile. This package also requires a distribution
decision for the matching `voiage-ffi` library; adding a Rust compiler to an
image would not supply a release-compatible native artifact. Shipping a host
binary in the R source package would undermine its
portable source-package model.

Accordingly, this criterion is explicitly **external** until a maintainer has
chosen and evidenced an rOpenSci-compatible native distribution route and, if
needed, the rOpenSci checking-environment change has been accepted. No inquiry,
pull request, upload, or submission is authorized by this record.

Local completion evidence:

- `scripts/validate_r_ffi_install.sh` builds the locked Rust FFI, installs
  `voiageR` into a temporary library, and verifies `evpi()`.
- `scripts/run_ropensci_pkgcheck.sh` runs rOpenSci `pkgcheck` in a temporary
  library without creating a review issue.
