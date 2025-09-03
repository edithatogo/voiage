# voiage Project Roadmap (v3)

## Vision

To establish `voiage` as the premier, cross-domain, high-performance library for Value of Information analysis. It will be distinguished by its analytical rigor, computational performance, and exceptional user experience. In the long term, its core, language-agnostic API will be ported to R and Julia, making it a true cross-platform standard.

## Current Status (As of Q3 2025)

The project has made significant progress and is transitioning from early development to a more mature phase. The core foundation is solid, and several key features are implemented.

*   **Phase 1 (Foundation & API Refactoring):** Completed. The core object-oriented API is fully in place, and the CI/CD pipeline is functional and comprehensive.
*   **Phase 2 (Health Economics Core):** In progress. Work has begun on the EVSI implementation and the plotting module, with EVSI partially implemented and plotting capabilities expanding.
*   **Phase 3 (Advanced Methods & Cross-Domain Expansion):** In progress. Portfolio optimization is fully implemented, and work has begun on other advanced methods.

---

### Phase 1: Foundation & API Refactoring (Completed)

**Goal:** Solidify the library's foundation by implementing a more robust, extensible, and user-friendly API. Address the key structural and software engineering weaknesses identified in the initial reviews.

1.  **Object-Oriented API Redesign & Functional Wrappers:**
    *   **Status: `Done`**
    *   The library now features a `DecisionAnalysis` class that encapsulates the core logic, with lightweight functional wrappers for convenience.
2.  **Domain-Agnostic Data Structures:**
    *   **Status: `Done`**
    *   The new data structures (`ParameterSet`, `ValueArray`) are defined in `voiage/schema.py`. Backward-compatible wrappers (`PSASample`, `NetBenefitArray`) exist in `voiage/core/data_structures.py` to support legacy code. The transition is complete.
3.  **CI/CD & Documentation Website:**
    *   **Status: `Done`**
    *   A full CI/CD pipeline is established in GitHub Actions, automating tests, linting, and formatting. The basic structure for the documentation website is in place.
4.  **Community Guidelines:**
    *   **Status: `Done`**
    *   The `CONTRIBUTING.md` and `AGENTS.md` files have been created to provide clear guidelines for human and AI contributors.

---

### Phase 2: State-of-the-Art Health Economics Core (Target: Q4 2025)

**Goal:** Implement the most critical features for the initial target audience (health economists) to make `voiage` a compelling alternative to existing tools.

1.  **Robust EVSI Implementation:**
    *   **Status: `In Progress`**
    *   A basic `evsi` function with a two-loop method exists in `voiage/methods/sample_information.py`. This needs to be expanded with a regression-based approach, robust Bayesian updating, and integration of advanced metamodels. The existing tests are incomplete and need to be activated.
    *   **Next Steps:**
        *   Complete regression-based method implementation
        *   Activate and fix existing tests
        *   Add comprehensive documentation and examples
2.  **Network Meta-Analysis (NMA) VOI:**
    *   **Status: `Not Started`**
    *   Implement `evsi_nma`, a critical and highly-demanded feature in HTA. This will involve handling multivariate parameter distributions for treatment effects.
    *   **Next Steps:**
        *   Define required data structures in `voiage/schema.py`
        *   Implement core NMA VOI calculation in `voiage/methods/network_nma.py`
        *   Add comprehensive test coverage
3.  **Validation & Benchmarking:**
    *   **Status: `Not Started`**
    *   For each core method, create a validation notebook that replicates the results of a published study or an example from an established R package (e.g., `BCEA`, `voi`).
    *   **Next Steps:**
        *   Create validation notebooks for EVPI, EVPPI, and EVSI
        *   Benchmark performance against R implementations
        *   Document validation results
4.  **Advanced Plotting Module & Core Examples:**
    *   **Status: `In Progress`**
    *   The `voiage/plot` module has been created with basic plotting functions in `voiage/plot/voi_curves.py`, but the core plotting functions (CEACs, VOI curves, etc.) are not yet fully implemented.
    *   A detailed tutorial notebook for a core health economics use case is needed.
    *   **Next Steps:**
        *   Complete CEAC plotting functionality
        *   Expand VOI curve plotting capabilities
        *   Create detailed tutorial notebooks

---

### Phase 3: Advanced Methods & Cross-Domain Expansion (Target: Q1-Q2 2026)

**Goal:** Broaden the library's capabilities to cover advanced VOI methods and actively support non-health applications.

1.  **Portfolio Optimization:**
    *   **Status: `Completed`**
    *   Implemented a robust `portfolio_voi` method in `voiage/methods/portfolio.py` with multiple optimization algorithms (greedy, integer programming).
2.  **Structural & Sequential VOI:**
    *   **Status: `Not Started`**
    *   Implement the placeholder methods for `structural_voi` and `sequential_voi`.
    *   **Next Steps:**
        *   Complete implementation of `structural_voi` in `voiage/methods/structural.py`
        *   Complete implementation of `sequential_voi` in `voiage/methods/sequential.py`
        *   Add comprehensive test coverage
3.  **Cross-Domain Example Notebooks:**
    *   **Status: `Not Started`**
    *   Develop at least two detailed tutorial notebooks for non-health problems.
    *   **Next Steps:**
        *   Create tutorial notebook for business strategy applications
        *   Create tutorial notebook for environmental policy applications
        *   Validate with domain experts
4.  **XArray Integration:**
    *   **Status: `Completed`**
    *   The core data structures in `voiage/schema.py` are already built on `xarray` and fully integrated throughout the library.

---

### Phase 4: Ecosystem, Community & Future Ports (Ongoing)

**Goal:** Grow the user and contributor community and lay the groundwork for R and Julia versions.

1.  **Future High-Performance Backend (JAX/XLA):**
    *   **Status: `In Progress`**
    *   JAX backend implementation started in `voiage/backends.py` and metamodels in `voiage/metamodels.py`.
    *   **Next Steps:**
        *   Optimize JAX backend implementation
        *   Add more comprehensive JAX support
        *   Document performance benefits
2.  **Community Engagement:**
    *   **Status: `In Progress`**
    *   The repository is being structured for easier contribution with comprehensive documentation.
    *   **Next Steps:**
        *   Continue building community engagement
        *   Improve documentation and examples
        *   Respond to community feedback
3.  **Language-Agnostic API Specification:**
    *   **Status: `Not Started`**
    *   **Next Steps:**
        *   Define JSON Schema for `DecisionAnalysis` inputs/outputs
        *   Document API specification
        *   Create validation tools
4.  **Planning for R/Julia Ports:**
    *   **Status: `Not Started`**
    *   **Next Steps:**
        *   Begin experimental implementations of `rvoiage` (R) and `Voiage.jl` (Julia)
        *   Validate the cross-language design
        *   Document the porting process

---

## Version Roadmap

### v0.3.0 (Target: Q4 2025)
- Complete EVSI implementation with both two-loop and regression methods
- Fully functional plotting module with CEAC and VOI curve plotting
- Validation notebooks for core methods
- Improved documentation and examples

### v0.4.0 (Target: Q1 2026)
- Network Meta-Analysis VOI implementation
- Structural and Sequential VOI implementations
- Cross-domain tutorial notebooks
- Performance optimizations

### v0.5.0 (Target: Q2 2026)
- Advanced metamodeling capabilities
- Real-time VOI calculation support
- Language-agnostic API specification
- Prototype R and Julia implementations

### v1.0.0 (Target: Q4 2026)
- Feature-complete stable release
- Comprehensive documentation
- Extensive validation against established tools
- Active community of users and contributors