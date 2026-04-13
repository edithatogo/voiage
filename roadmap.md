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
    *   CodeQL security scanning, Ruff linting, Bandit security checks, mutation testing support.
4.  **Community Engagement:**
    *   **Status: `🚧 In Progress`**
    *   Repository structured for contributions, Conductor workflow for AI-assisted development.
5.  **Language-Agnostic API Specification:**
    *   **Status: `📋 Planned`**
6.  **Planning for R/Julia Ports:**
    *   **Status: `📋 Planned`**
