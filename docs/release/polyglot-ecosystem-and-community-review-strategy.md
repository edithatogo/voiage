# Polyglot Ecosystem & Community Review Strategy for `voiage`

**Version:** 2.1.0  
**Status:** Canonical Strategy & Readiness Evaluation  
**Date:** 23 August 2026  
**Scope:** Polyglot Language Registries, Scientific Software Journals, High-Performance Computing (HPC) Distributions, and Open-Source Sustainability Affiliations.

---

## 1. Executive Summary & Master Cross-Venue Evaluation Matrix

| Category | Venue / Registry | Target Surface | Role / Mechanism | Suitability | Maintenance Overhead | Exclusivity & Overlap Rules | Recommended Timing |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Community Review** | **pyOpenSci** | Python (`voiage`) | Open GitHub peer review | **Essential** | Low-to-Moderate | Direct fast-track partnership with JOSS; no paper copyright conflict | **Immediate / Near-Term** |
| **Modular Journal** | **JOSS** | Polyglot (`paper.md`) | Open Journals GitHub review | **Essential** | Low | Fast-track via pyOpenSci/rOpenSci; requires short software paper | **Immediate (via pyOpenSci)** |
| **Julia Registry** | **Yggdrasil / General** | Julia (`Voiage.jl`) | BinaryBuilder C ABI + JuliaRegistrator | **Essential** | Moderate | Standard package distribution; no journal exclusivity | **Immediate / In-Flight** |
| **Julia Ecosystem** | **JuliaHealth** | Julia (`Voiage.jl`) | Domain listing & community | **High** | Negligible | Community directory; no publication overlap | **Phase 2 (Post-General)** |
| **Julia Conference** | **JuliaCon Proceedings** | Julia (`Voiage.jl`) | Open Journals conference proceedings | **High** | Moderate | Requires JuliaCon talk presentation; distinct Julia focus | **Phase 3 (Event-Driven)** |
| **Rust Registry** | **crates.io / docs.rs** | Rust Core Crates | Cargo OIDC trusted publishing | **Essential** | Low | Standard library distribution | **Immediate / Release-Bound** |
| **R Registry** | **r-universe** | R (`voiageR`) | Automated binary build tracking | **High** | Negligible | Hosted companion binaries; zero CRAN conflict | **Immediate (Active)** |
| **R Registry** | **CRAN** | R (`voiageR`) | Canonical R repository | **High** | High | Strict R CMD check; requires standalone compilation | **Phase 2 (Post-rextendr)** |
| **Community Review** | **rOpenSci** | R (`voiageR`) | Statistical Software Review (`srr`) | **High** | High | Cannot be under concurrent journal review; JOSS/R Journal link | **Phase 3 (Post-JOSS)** |
| **Statistical Journal** | **The R Journal** | R (`voiageR`) | Single-blind peer-reviewed journal | **High** | Moderate | Package must be hosted on CRAN first; distinct R focus | **Phase 3 (Post-CRAN)** |
| **Genomics Registry** | **Bioconductor** | R (`voiageR`) | High-throughput genomics repository | **Unsuitable** | N/A | **Explicitly Reject**: Out of domain scope (genomics vs decision theory) | **Do Not Submit** |
| **HPC Distribution** | **Spack** | Python / Rust Core | HPC package manager recipe (`py-voiage`) | **High** | Low | Non-exclusive source recipe | **Phase 2 (Near-Term)** |
| **HPC Distribution** | **EasyBuild** | European HPC | Easyconfig recipe (`foss-2023a/24a`) | **High** | Moderate | Non-exclusive build recipe | **Phase 2 (Near-Term)** |
| **HPC Stack** | **E4S** | Exascale GPU/HPC | DOE curated binary stack | **Moderate** | Moderate | Depends on upstream merged Spack package | **Phase 3 (Post-Spack)** |
| **Sustainability** | **Open Source Collective** | Core Project | 501(c)(6) fiscal hosting & sponsors | **Immediate** | Very Low | Non-exclusive fiscal host; supports GitHub Sponsors | **Immediate** |
| **Sustainability** | **NumFOCUS Affiliated** | Core Project | Scientific community affiliation | **High** | Low | Requires demonstrated multi-user adoption & paper DOI | **Phase 3 (Post-JOSS)** |
| **Sustainability** | **NumFOCUS FSP** | Core Project | Full 501(c)(3) fiscal sponsorship | **Premature** | Very High | Ineligible under solo-maintainer model; requires multi-org board | **Long-Term (v3.0+)** |
| **Methodology Journal**| **JSS (Journal of Stat Software)** | Polyglot (`paper/main.tex`)| Full-length 25–35 page monograph | **High** | Very High | Must present substantial methodological/algorithmic depth beyond JOSS | **Phase 4 (Long-Term)** |

---

## 2. Polyglot Language Registries & HPC Packaging Strategy

```
+---------------------------------------------------------------------------------------+
|                    POLYGLOT ECOSYSTEM PACKAGING ROADMAP                                |
+---------------------------------------------------------------------------------------+
  │
  ├── 🐍 PYTHON: PyPI (Published: v2.1.0) ──> conda-forge (PR #34308 passing)
  │
  ├── 🟣 JULIA: Yggdrasil PR #14292 (Buildkite 31972 passing) ──> Julia General (Voiage.jl)
  │                                                                 │
  │                                                                 └──> JuliaHealth Listing
  │
  ├── 🦀 RUST: crates.io (4 core crates: domain, numerics, serialization, diagnostics)
  │             └──> docs.rs & Lib.rs automated indexing
  │
  ├── 📊 R: r-universe (Active hosted builds) ──> rextendr bridge ──> CRAN (voiageR)
  │
  └── ⚡ HPC: Spack PR (py-voiage) ──> EasyBuild Easyconfig ──> E4S Stack Curation
```

### A. Julia Ecosystem (`Voiage.jl`)
1. **Yggdrasil (`voiage_ffi_jll`)**:
   - **Rationale:** `Voiage.jl` uses the C ABI (`extern "C"` functions in `voiage-ffi`). Distributing precompiled binaries via BinaryBuilder provides multi-architecture support (x86_64, aarch64, Windows, macOS, Linux glibc/musl) with zero local compiler requirements.
   - **Status:** Yggdrasil PR #14292 passed all 15 included targets on Buildkite run 31972.
2. **Julia General Registry**:
   - **Action:** Upon upstream Yggdrasil merge, update `bindings/julia/Project.toml` to declare `voiage_ffi_jll` as a dependency and trigger `@JuliaRegistrator register subdir=bindings/julia`.
3. **JuliaHealth Directory**:
   - **Rationale:** Value of Information analysis has extensive applications in Health Technology Assessment (HTA). Listing `Voiage.jl` under JuliaHealth connects the library with domain modelers.

### B. Rust Ecosystem (`rust/crates/`)
1. **crates.io Modular Publishing**:
   - **Published Core Crates:** `voiage-domain`, `voiage-numerics`, `voiage-serialization`, `voiage-diagnostics`.
   - **Internal Crates (`publish = false`):** `voiage-ffi` (C ABI), `voiage-python` (PyO3 module), `voiage-test-support`.
   - **Rationale:** Exposes zero-overhead, highly optimized Value of Information kernels to Rust simulation engineers and native CLI tools.

### C. R Ecosystem (`r-package/voiageR/`)
1. **r-universe (`edithatogo.r-universe.dev/voiageR`)**:
   - **Status:** Active. Continuously builds and hosts precompiled binaries for macOS, Windows, and Linux.
2. **CRAN Submission Path**:
   - **Prerequisite:** CRAN policy strictly forbids runtime downloads of binary shared libraries. `voiageR` must incorporate self-contained compilation (e.g. `rextendr` vendored Rust compilation in `src/`) before submitting to CRAN.
3. **Bioconductor Scope Determination**:
   - **Decision: EXPLICITLY REJECT / DO NOT SUBMIT.** Bioconductor is exclusively dedicated to high-throughput molecular genomics and bioinformatics. VoI is general decision theory and health economics; submission would result in immediate scope rejection.

### D. High-Performance Computing (HPC) Distributions
1. **Spack (`py-voiage`) & EasyBuild**:
   - **Rationale:** Standard package managers across US National Laboratories (ORNL, ANL, LLNL) and European Tier-1 supercomputing sites (LUMI, CSCS).
   - **Timing:** Phase 2 (post-v2.1 release).

---

## 3. Scientific Software Journals & Peer Review Sequencing

```
+---------------------------------------------------------------------------------------+
|                    SCIENTIFIC JOURNAL & PEER REVIEW SEQUENCE                          |
+---------------------------------------------------------------------------------------+

   [Phase 1: Near-Term]      [Phase 2: R Ecosystem]      [Phase 3: Deep Methodology]
   ────────────────────      ──────────────────────      ───────────────────────────
   pyOpenSci Review          CRAN Acceptance             Comprehensive Monograph
          │                        │                               │
          ▼ (Fast-Track)           ▼                               ▼
      JOSS Paper             rOpenSci Review             Journal of Statistical
      (paper.md)             (srr Standards)                 Software (JSS)
                                   │
                                   ▼
                             The R Journal
```

### A. Phase 1: pyOpenSci $\rightarrow$ JOSS Fast-Track
- **Mechanism:** pyOpenSci conducts a comprehensive, open peer review of the Python codebase on GitHub (verifying tests, packaging, documentation, API design, and maintenance commitment).
- **Fast-Track to JOSS:** Under the official pyOpenSci–JOSS partnership, passing pyOpenSci review waives JOSS software review, requiring only an editorial review of `paper.md`.
- **Advantage:** Maximum reputational accreditation with Diamond Open Access (free), zero dual-submission friction, and indexed JOSS DOI.

### B. Phase 2: CRAN $\rightarrow$ rOpenSci $\rightarrow$ The R Journal
- **Sequence:**
  1. Complete standalone compilation bridge for `voiageR` and publish on CRAN.
  2. Submit `voiageR` to rOpenSci for statistical software certification (`srr` standards mapped in `specs/submission-readiness/ropensci-standards-mapping.md`).
  3. Submit a dedicated R-focused article to *The R Journal* highlighting `voiageR`'s integration with tidyverse, comparison against `voi` and `BCEA`, and native Rust acceleration.

### C. Phase 3: Comprehensive Statistical Methodology Monograph (JSS)
- **Scope Distinction:** A short JOSS paper (~1,000 words) describes software structure; it does **not** preclude publishing a comprehensive 25–35 page methodological monograph in the *Journal of Statistical Software (JSS)* or *Medical Decision Making (MDM)*.
- **Content:** Full mathematical derivations of structural/implementation uncertainty, streaming EVPI/EVPPI algorithms, cross-language benchmarks (Rust/NumPy/JAX/R/Julia), and real-world policy case studies.

---

## 4. Sustainability Affiliations, Identifiers & Governance

### A. Sustainability Affiliations
1. **Open Source Collective (OSC) — Immediate:**
   - Low-friction 501(c)(6) fiscal hosting. Enables community donations, GitHub Sponsors matching, and institutional project support without complex board bureaucracy.
2. **NumFOCUS Affiliated Project — Medium-Term (Post-JOSS):**
   - Provides scientific computing prestige, Google Summer of Code (GSoC) eligibility, and small development grants ($5,000–$10,000).
   - Requires published paper DOI and evidence of external research adoption.
3. **NumFOCUS Fiscally Sponsored Project (FSP) — Long-Term (v3.0+):**
   - Full 501(c)(3) fiscal home for multi-million-dollar federal research grants (NSF, Wellcome, CZI).
   - Ineligible under the current solo-maintainer model; requires transitioning `GOVERNANCE.md` to a multi-institution steering committee.

### B. Persistent Identifiers & Archival
1. **SciCrunch RRID:** Form submitted; awaiting curator assignment of `RRID:SCR_######` to enable automated literature tracking in PubMed Central / Europe PMC.
2. **Software Heritage (SWHID):** Fully active (`swh:1:snp:31f89375852737bb9eb62ebc03fadfbc7ff70c2d` in `CITATION.cff`).
3. **Zenodo (DOI):** Ready for release DOI minting and automated JOSS synchronization.

---

## 5. Decision & Execution Roadmap

| Milestone | Target Venues | Primary Deliverables / Actions | Authority Boundary |
| :--- | :--- | :--- | :--- |
| **Milestone 1** | pyOpenSci & JOSS | Author opens pyOpenSci inquiry; execute JOSS fast-track upon approval | Author & pyOpenSci Editors |
| **Milestone 2** | Julia General | Monitor Yggdrasil PR #14292; trigger `@JuliaRegistrator` for `Voiage.jl` | Maintainer & JuliaRegistrator |
| **Milestone 3** | Open Source Collective | Create collective for `voiage`; connect GitHub Sponsors | Maintainer |
| **Milestone 4** | HPC (Spack & EasyBuild) | Submit `py-voiage` recipe to Spack; submit easyconfig to EasyBuild | Maintainer / Contributor |
| **Milestone 5** | R (CRAN & rOpenSci) | Refactor `voiageR` with `rextendr` native compilation; submit to CRAN, then rOpenSci | Maintainer & rOpenSci Reviewers |
| **Milestone 6** | The R Journal & JSS | Prepare R Journal paper; expand `paper/main.tex` for JSS monograph | Author & Journal Editors |
