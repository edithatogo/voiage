# voiage Project Roadmap (v3)

## Vision

To establish `voiage` as the premier, cross-domain, high-performance library for Value of Information analysis. It will be distinguished by its analytical rigor, computational performance, and exceptional user experience.

## Current Status (As of April 2026)

The project has a solid foundation with core VOI methods implemented, modern CI/CD, and automated publishing pipelines.

*   **Phase 1 (Foundation & API Refactoring):** ✅ **Complete** - Core OO API, data structures, CI/CD, and documentation are all in place.
*   **Phase 2 (Health Economics Core):** ✅ **Complete** - EVPI, EVPPI, EVSI (two-loop), NMA VOI, structural VOI, and plotting are implemented.
*   **Phase 3 (Advanced Methods & Cross-Domain):** ✅ **Complete** - Structural VOI, NMA VOI, JAX JIT compilation, and cross-domain support implemented.

---

### Phase 1: Foundation & API Refactoring ✅ **COMPLETE**

**Goal:** Solidify the library's foundation by implementing a more robust, extensible, and user-friendly API.

1.  **Object-Oriented API Redesign & Functional Wrappers:**
    *   **Status: `✅ Done`**
    *   `DecisionAnalysis` class encapsulates core logic with functional wrappers.
2.  **Domain-Agnostic Data Structures:**
    *   **Status: `✅ Done`**
    *   `ParameterSet`, `ValueArray`, `TrialDesign`, and other structures in `voiage/schema.py` using xarray backend.
3.  **CI/CD & Documentation Website:**
    *   **Status: `✅ Done`**
    *   Full CI/CD pipeline: uv, Ruff, CodeQL, Benchmarks, Sphinx docs, GitHub Pages, automated publishing to PyPI/TestPyPI/Conda.
4.  **Community Guidelines:**
    *   **Status: `✅ Done`**
    *   `CONTRIBUTING.md`, `AGENTS.md`, Renovate for dependency updates.

---

### Phase 2: State-of-the-Art Health Economics Core ✅ **COMPLETE**

**Goal:** Implement the most critical features for health economists.

1.  **Robust EVSI Implementation:**
    *   **Status: `✅ Done`**
    *   Two-loop Monte Carlo method implemented in `sample_information.py`.
    *   Regression-based EVSI with metamodel support (GAM, RandomForest, BART via metamodels module).
2.  **Network Meta-Analysis (NMA) VOI:**
    *   **Status: `✅ Done`**
    *   `calculate_nma_evpi()` and `calculate_nma_evppi()` in `voiage/methods/network_meta_analysis.py`.
    *   CLI command: `voiage calculate-nma-voi`.
3.  **Structural Uncertainty VOI:**
    *   **Status: `✅ Done`**
    *   `structural_evpi()` and `structural_evppi()` with JAX JIT compilation in `voiage/methods/structural.py`.
    *   CLI commands: `voiage calculate-structural-evpi`, `voiage calculate-structural-evppi`.
4.  **Validation & Benchmarking:**
    *   **Status: `✅ Done`**
    *   Integration tests with realistic health economics and diabetes NMA scenarios.
    *   Performance benchmarks comparing NumPy vs JAX implementations.
5.  **Advanced Plotting Module & Core Examples:**
    *   **Status: `✅ Done`**
    *   CEAC plotting in `voiage/plot/ceac.py`.
    *   VOI curves in `voiage/plot/voi_curves.py`.
    *   CLI example generation and documentation.

---

### Phase 3: Advanced Methods & Cross-Domain Expansion ✅ **COMPLETE**

**Goal:** Broaden capabilities to advanced VOI methods and cross-domain support.

1.  **Structural VOI:**
    *   **Status: `✅ Done`**
    *   Full implementation with JAX JIT acceleration.
2.  **Calibration VOI:**
    *   **Status: `✅ Done`**
    *   `voi_calibration()` in `voiage/methods/calibration.py`.
3.  **Adaptive Trial VOI:**
    *   **Status: `✅ Done`**
    *   `adaptive_evsi()` and sophisticated trial simulator in `voiage/methods/adaptive.py`.
4.  **Cross-Domain Support:**
    *   **Status: `✅ Done`**
    *   Multi-domain module (`voiage/multi_domain.py`) with healthcare, financial, environmental, and engineering support.
    *   Domain-specific analysis classes and utilities.
5.  **XArray Integration:**
    *   **Status: `✅ Done`**
    *   All core data structures built on xarray Dataset backend.
6.  **High-Performance JAX Backend:**
    *   **Status: `✅ Done`**
    *   JIT-compiled versions of structural EVPI/EVPPI.
    *   JAX backend in `voiage/main_backends.py` with GPU acceleration support.

---

### Phase 4: Ecosystem, Community & Future Ports (Ongoing) 🚧 **IN PROGRESS**

**Goal:** Grow the user and contributor community and lay the groundwork for R and Julia versions.

1.  **Automated Publishing Pipeline:**
    *   **Status: `✅ Done`**
    *   TestPyPI → PyPI → Conda-forge automated publishing on `v*` tags.
2.  **Dependency Management:**
    *   **Status: `✅ Done`**
    *   uv for package management, Renovate for automated updates.
3.  **Security & Quality:**
    *   **Status: `✅ Done`**
    *   CodeQL security scanning, Ruff linting/security rules, ty type checking, and mutation testing support.
4.  **Community Engagement:**
    *   **Status: `🚧 In Progress`**
    *   Repository structured for contributions, Conductor workflow for AI-assisted development.
5.  **Language-Agnostic API Specification:**
    *   **Status: `📋 Planned`**
    *   Define a stable core contract around `ValueArray`, `ParameterSet`, `TrialDesign`, and method outputs.
    *   Use spec-first development with conformance fixtures before expanding bindings.
    *   Target a core API that can be surfaced consistently from Python first, then R, Julia, TypeScript, Go, and Rust.
    *   Prioritize deterministic validation, explicit schemas, and backend-agnostic behavior over language-specific convenience wrappers.
6.  **Planning for R/Julia Ports:**
    *   **Status: `📋 Planned`**
    *   Treat R and Julia as the first external ports of the shared core API.
    *   Keep the Python implementation as the reference binding, with additional bindings generated or hand-wrapped from the same canonical spec.
    *   Treat each external binding as a releasable package with a registry target, automated CI, conformance-fixture validation, and release automation before it is considered complete.

---

### Phase 5: Spec, Fixtures & Polyglot Bindings 📋 **PLANNED**

**Goal:** Mature the library into a broadly usable core analysis engine with stable cross-language contracts.

1.  **Core API Specification:**
    *   Define method signatures, schema invariants, and error behavior for the public VOI surface.
2.  **Conformance Fixtures:**
    *   Build canonical input/output fixtures that every binding must pass before release.
3.  **Python Cleanup and Stabilization:**
    *   Finish the Python-side normalization needed to make the canonical API implementation simple and durable.
4.  **First External Bindings:**
    *   Deliver R and Julia bindings against the same contract, then extend to TypeScript, Go, and Rust if adoption warrants it.
    *   Publishing targets must be planned with the implementation:
        - Python: PyPI, TestPyPI, and Conda-forge.
        - R: CRAN when mature, with r-universe or GitHub Releases for early distribution.
        - Julia: Julia General registry.
        - TypeScript: npm.
        - Go: tagged Go modules consumable through the Go module proxy, with GitHub Releases for release notes/artifacts.
        - Rust: crates.io.
        - .NET: NuGet, targeting .NET 11 (`net11.0`).
    *   CI/CD must be language-specific and release-aware for every binding:
        - Build, lint/format, type/static checks, unit tests, docs checks, and shared conformance fixtures.
        - Package dry-run validation on pull requests.
        - Trusted or token-scoped publishing on version tags/releases.
        - Registry-specific provenance and changelog generation where supported.

---

### Phase 6: Ecosystem Integrations 📋 **PLANNED**

**Goal:** Make `voiage` useful as a stable VOI engine for upstream modelling
packages while preserving a clean dependency boundary.

1.  **lifecourse Integration Contract:**
    *   **Status: `📋 Planned`**
    *   Define a `lifecourse` VOI artifact profile covering net benefits,
        parameter samples, strategy names, WTP thresholds, scaling metadata,
        provenance, method settings, and diagnostics.
    *   Align the artifact profile with HEOML as the candidate shared
        health-economic interchange profile.
    *   Use portable artifacts rather than pickle or internal Python objects.
    *   Keep `voiage` independent of `lifecourse` runtime internals.
    *   Support optional adapter use from `lifecourse` once version,
        dependency, and fixture compatibility are stable.
    *   Use shared conformance fixtures so both repositories can validate EVPI,
        EVPPI, EVSI, and ENBS behavior consistently.
2.  **Ecosystem Module Incubation:**
    *   **Status: `📋 Planned`**
    *   Define `voiage` as the VOI engine in the HEOR ecosystem spanning
        `lifecourse`, `innovate`, `mars`, HEOML, and future sibling modules.
    *   Keep the ecosystem scope focused on health economics, outcomes
        research, HTA, reimbursement, implementation uncertainty, and
        health-policy evaluation.
    *   Keep integrations optional, artifact-first, versioned, and fixture-tested.
    *   Reserve HEOML extension alignment for VOI handoff and VOI result metadata.
    *   Treat `mars` as a fixed-API optional metamodel backend rather than a
        package whose core API should change for VOI-specific needs.
    *   Maintain the local contract outline under `docs/ecosystem/` and
        `specs/ecosystem/` so each sibling module can align against the same
        portable VOI boundary before adapter work begins.
