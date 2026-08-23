# rOpenSci Statistical Software Standards Mapping for `voiageR`

**Package:** `voiageR` (version `2.1.0`)  
**Scope:** Value of Information (VOI) analysis, Expected Value of Perfect Information (EVPI), Expected Value of Partial Perfect Information (EVPPI), Expected Value of Sample Information (EVSI), and Expected Net Benefit of Sampling (ENBS).  
**Standards Reference:** [rOpenSci Statistical Software Peer Review Guidelines](https://stats-devguide.ropensci.org/)

---

## 1. General Standards (G1.0 – G5.0)

| Category | Standard ID | Requirement Summary | `voiageR` Implementation & Evidence |
| :--- | :--- | :--- | :--- |
| **Design** | `G1.0` | Clear, coherent, and bounded API design | `r-package/voiageR/R/voiageR.R` exports intuitive functions: `evpi()`, `evppi()`, `evsi()`, `enbs()`. |
| **Inputs** | `G2.0` | Type validation and explicit dimension checking | Matrix dimensions validated (`n_samples >= 1`, `n_strategies >= 2`), missing/non-finite inputs rejected fail-fast. |
| **Inputs** | `G2.1` | Explicit handling of `NA`, `NaN`, and `Inf` | `stopifnot(all(is.finite(nb)))` raises descriptive errors before native dispatch. |
| **Algorithms** | `G3.0` | Numerical stability and verified algorithms | C ABI dispatch to Rust numerical kernels with double-precision IEEE-754 floating point arithmetic. |
| **Outputs** | `G4.0` | Predictable, standard return types | Functions return standard numeric vectors or scalar floats with non-negative bounds. |
| **Testing** | `G5.0` | Multi-environment automated testing | `tests/testthat/test-voiageR.R`, `test-native-ffi.R`, and `test-zz-numerical-reference.R` run across Linux, macOS, and Windows. |
| **Testing** | `G5.1` | Numerical tolerance comparisons | Tested against exact reference fixtures (`specs/numerical-reference/v1/evpi-cases.json`) with `< 1e-12` relative error. |

---

## 2. Bayesian & Monte Carlo Sensitivity Analysis Standards (BS1.0 – BS7.0)

| Category | Standard ID | Requirement Summary | `voiageR` Implementation & Evidence |
| :--- | :--- | :--- | :--- |
| **Model Structure** | `BS1.0` | Explicit definition of decision space and state draws | Net benefit matrix inputs ($S \times D$, $S$ samples, $D$ decisions/strategies) represent Monte Carlo draws from joint prior/posterior. |
| **Sampling & Seeds** | `BS2.0` | Deterministic reproducibility under explicit RNG seeds | Verified in `tests/testthat/test-voiageR.R` and `test_deterministic_simulation.py`. |
| **Convergence** | `BS3.0` | Sample size scaling and sensitivity evaluation | `test-voiageR.R` validates that EVSI monotonically increases with study sample size ($N$). |
| **Uncertainty** | `BS4.0` | Propagation of parameter uncertainty into value metrics | EVPPI isolates parameter subsets ($\theta_k$) from joint PSA parameter sets. |
| **Missingness** | `BS5.0` | Reject incomplete or un-aligned parameter sets | Enforces length equality between net benefit samples and parameter draws. |
| **Reference** | `BS6.0` | Comparison against published analytical benchmarks | Validated against conjugate Normal-Normal analytic EVSI and discrete canonical EVPI tables. |

---

## 3. Packaging & Lifecycle Boundaries

- **Self-Contained Installation Gate:** Currently requires `VOIAGE_FFI_LIBRARY` or `reticulate` fallback; tracked as `repository_blocked` in `specs/submission-readiness/ropensci-evidence.json` until an embedded Rust compilation bridge (`rextendr`) is integrated.
- **Review Exclusivity:** Submission to rOpenSci is scheduled after the primary JOSS publication milestone to avoid dual-venue review overlap.
