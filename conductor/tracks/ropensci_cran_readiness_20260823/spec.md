# Track Specification: R CRAN Standalone Distribution & rOpenSci Review

## Scope & Purpose
Transform `voiageR` into an independently compilable R package compliant with CRAN policy and rOpenSci statistical software guidelines.

## Requirements
1. **Self-Contained Build**: Elimination of manual `VOIAGE_FFI_LIBRARY` setup for end-users on CRAN installations.
2. **CRAN Conformance**: Clean `R CMD check --as-cran` across Linux, macOS, and Windows.
3. **Statistical Standards**: Adherence to rOpenSci General (G1.0–G5.0) and Bayesian (BS1.0–BS7.0) guidelines.
4. **Documentation**: PDF reference manual and vignette explaining workflow differences relative to `voi` and `BCEA`.
5. **Staged Execution**: Defer external submission until post-JOSS milestone.
