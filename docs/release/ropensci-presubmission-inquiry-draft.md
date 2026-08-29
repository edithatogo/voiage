# rOpenSci pre-submission inquiry — unposted draft

Status: **prepared locally; not posted or submitted**

## Proposed package

- Package: `voiageR` 2.1.0
- Repository: `https://github.com/edithatogo/voiage`
- Package subdirectory: `r-package/voiageR`
- Licence: Apache License 2.0 or later

`voiageR` is the bounded R facade for the Rust-owned numerical core in
`voiage`. It installs as a self-contained compiled R source package: the EVPI
and signed-ENBS paths call registered native routines built from the bundled,
offline Rust crate. EVPPI and EVSI remain explicitly documented optional
Python-backed paths rather than being represented as standalone native R
implementations.

## Scope and overlap

The proposed review scope is the R package, its native interface, input and
error contracts, numerical-reference corpus, documentation, and statistical
software standards mapping. It is not a request to review every Python,
Julia, or C ABI feature in the parent repository.

The package overlaps in domain with `BCEA`, `hesim`, `voi`, `dampack`, and
SAVI, but its proposed contribution is a compact cross-language facade with a
single Rust execution authority, shared numerical fixtures, explicit stable
capability boundaries, and installed-package parity checks. The inquiry asks
whether that contribution and repository layout fit rOpenSci review scope.

## Current repository evidence

- `R CMD check --as-cran` completes with zero errors and warnings for the exact
  `voiageR_2.1.0.tar.gz` source archive.
- The pinned `srr` pre-submit check accepts all applicable standards; the
  repository maps 68 general and 14 probability-distribution standards with
  item-level compliance or justified non-applicability tags.
- Installed native EVPI and ENBS tests and eight shared EVPI reference cases
  pass without an ambient `voiage` shared library.
- Direct `covr` execution reports 89.47% line coverage.
- The pinned repository-aware `pkgcheck` passes package metadata, examples,
  vignettes, statistical standards, 89.5% coverage, `R CMD check`, and the
  public default-branch `R CMD Check and Retained Bindings CI` workflow at
  merged revision `b7c58db2e58eb11e24119d6c919a10f349358c5d`.

## Questions for an editor

1. Is a package living in an R subdirectory of a polyglot repository eligible
   when its R source archive, documentation, tests, and compiled native code
   are self-contained?
2. Is the bounded native EVPI/ENBS scope sufficient for review when optional
   EVPPI/EVSI helpers truthfully retain a Python dependency?
3. Would rOpenSci prefer the R package to have a dedicated repository before a
   full submission, despite the shared ABI and fixture governance benefits of
   the current monorepository?

## Unperformed actions and remaining authority

This draft has not been posted, submitted, or sent to rOpenSci. Before any
maintainer-authorized inquiry, refresh the author guide, review package
lifecycle and support commitments, and confirm that contemporaneous pyOpenSci or JOSS review
does not conflict with rOpenSci policy. Editorial scope advice, review,
acceptance, and onboarding remain external decisions.
