# voiage Project Roadmap (v2)

## Vision

To establish `voiage` as the premier, cross-domain, high-performance library for Value of Information analysis. It will be distinguished by its analytical rigor, computational performance, and exceptional user experience. In the long term, its core, language-agnostic API will be ported to R and Julia, making it a true cross-platform standard.

## Current Status (As of Q3 2025)

The project has a solid foundation and is partially through its initial refactoring and feature implementation phases.

*   **Phase 1 (Foundation & API Refactoring):** Mostly complete. The core object-oriented API is in place, and the CI/CD pipeline is functional. The primary remaining task is to finalize the domain-agnostic data structures.
*   **Phase 2 (Health Economics Core):** In progress. Work has begun on the EVSI implementation and the plotting module, but these are not yet complete.

---

### Phase 1: Foundation & API Refactoring (Target: 8 Weeks)

**Goal:** Solidify the library's foundation by implementing a more robust, extensible, and user-friendly API. Address the key structural and software engineering weaknesses identified in the initial reviews.

1.  **Object-Oriented API Redesign & Functional Wrappers:**
    *   **Status: `Done`**
    *   The library now features a `DecisionAnalysis` class that encapsulates the core logic, with lightweight functional wrappers for convenience.
2.  **Domain-Agnostic Data Structures:**
    *   **Status: `In Progress`**
    *   The new data structures (`ParameterSet`, `ValueArray`) are defined in `voiage/schema.py`. Backward-compatible wrappers (`PSASample`, `NetBenefitArray`) exist in `voiage/core/data_structures.py` to support legacy code. The transition is ongoing.
3.  **CI/CD & Documentation Website:**
    *   **Status: `Done`**
    *   A full CI/CD pipeline is established in GitHub Actions, automating tests, linting, and formatting. The basic structure for the documentation website is in place.
4.  **Community Guidelines:**
    *   **Status: `Done`**
    *   The `CONTRIBUTING.md` and `AGENTS.md` files have been created to provide clear guidelines for human and AI contributors.

---

### Phase 2: State-of-the-Art Health Economics Core (Target: 12 Weeks)

**Goal:** Implement the most critical features for the initial target audience (health economists) to make `voiage` a compelling alternative to existing tools.

1.  **Robust EVSI Implementation:**
    *   **Status: `In Progress`**
    *   A basic `evsi` function with a two-loop method exists. This needs to be expanded with a regression-based approach, robust Bayesian updating, and integration of advanced metamodels. The existing tests are incomplete and need to be activated.
2.  **Network Meta-Analysis (NMA) VOI:**
    *   **Status: `Not Started`**
    *   Implement `evsi_nma`, a critical and highly-demanded feature in HTA. This will involve handling multivariate parameter distributions for treatment effects.
3.  **Validation & Benchmarking:**
    *   **Status: `Not Started`**
    *   For each core method, create a validation notebook that replicates the results of a published study or an example from an established R package (e.g., `BCEA`, `voi`).
4.  **Advanced Plotting Module & Core Examples:**
    *   **Status: `In Progress`**
    *   The `voiage/plot` module has been created, but the core plotting functions (CEACs, VOI curves, etc.) are not yet implemented.
    *   A detailed tutorial notebook for a core health economics use case is needed.

---

### Phase 3: Advanced Methods & Cross-Domain Expansion (Target: 16 Weeks)

**Goal:** Broaden the library's capabilities to cover advanced VOI methods and actively support non-health applications.

1.  **Portfolio Optimization:**
    *   **Status: `Not Started`**
    *   Implement a robust `portfolio_voi` method.
2.  **Structural & Sequential VOI:**
    *   **Status: `Not Started`**
    *   Implement the placeholder methods for `structural_voi` and `sequential_voi`.
3.  **Cross-Domain Example Notebooks:**
    *   **Status: `Not Started`**
    *   Develop at least two detailed tutorial notebooks for non-health problems.
4.  **XArray Integration:**
    *   **Status: `In Progress`**
    *   The core data structures in `voiage/schema.py` are already built on `xarray`. This integration needs to be leveraged more deeply throughout the library.

---

### Phase 4: Ecosystem, Community & Future Ports (Ongoing)

**Goal:** Grow the user and contributor community and lay the groundwork for R and Julia versions.

1.  **Future High-Performance Backend (JAX/XLA):**
    *   **Status: `Not Started`**
2.  **Community Engagement:**
    *   **Status: `In Progress`**
    *   The repository is being structured for easier contribution.
3.  **Language-Agnostic API Specification:**
    *   **Status: `Not Started`**
4.  **Planning for R/Julia Ports:**
    *   **Status: `Not Started`**
