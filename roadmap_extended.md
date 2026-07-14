# voiage Project Roadmap (v3) - Extended Edition

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

### Phase 4: Emerging VOI Methods Implementation (Target: 20 Weeks)

**Goal:** Implement emerging and advanced VOI methods that extend the library's capabilities beyond standard approaches.

1.  **Real Options Valuation in Healthcare:**
    *   **Status: `Not Started`**
    *   Implement real options methods for healthcare investment decisions, considering flexibility and timing.
    *   **Reference**: Mariano, B. S., Ferreira, J. J., & Godinho, M. (2013). Real options in healthcare investments.
2.  **Multi-Criteria Decision Analysis (MCDA) VOI:**
    *   **Status: `Not Started`**
    *   Extend portfolio optimization to handle multi-objective optimization.
    *   **Reference**: Thokala, P., & Dyer, J. (2016). The analytical hierarchy process.
3.  **Dynamic VOI Methods:**
    *   **Status: `Not Started`**
    *   Extension of sequential VOI with dynamic programming approaches.
    *   **Reference**: Alarid-Escudero, F., et al. (2018). Time travel in value of information analysis.
4.  **Value of Correlation Information:**
    *   **Status: `Not Started`**
    *   Methods for efficiently calculating VOI for parameter correlations.
    *   **Reference**: Coyle, D., & Oakley, J. E. (2008). Estimating the expected value of partial perfect information.
5.  **Machine Learning Integration for VOI:**
    *   **Status: `Not Started`**
    *   Integration with deep learning frameworks for surrogate modeling and advanced sampling.
    *   **Reference**: Recent work on Gaussian processes and neural networks for Bayesian emulation.

---

### Phase 5: Specialized Applications Implementation (Target: 24 Weeks)

**Goal:** Implement specialized VOI methods for specific application domains and contexts.

1.  **Precision Medicine VOI:**
    *   **Status: `Not Started`**
    *   Value of information methods tailored for personalized treatment decisions.
    *   **Reference**: Recent work on personalized medicine and treatment effect heterogeneity.
2.  **Implementation Science VOI:**
    *   **Status: `Not Started`**
    *   Value of information for decisions about implementation strategies and contextual factors.
    *   **Reference**: Implementation science literature in health services research.
3.  **Global Health VOI:**
    *   **Status: `Not Started`**
    *   Value of information methods that account for resource constraints and diverse contexts.
    *   **Reference**: Global health economics literature.
4.  **Rare Disease VOI:**
    *   **Status: `Not Started`**
    *   Value of information methods tailored for rare disease contexts with limited data.
    *   **Reference**: Rare disease health economics literature.
5.  **Causal Inference VOI Methods:**
    *   **Status: `Not Started`**
    *   Value of information methods that account for causal relationships and confounding.
    *   **Reference**: Pearl, J. (2009). Causality: Models, reasoning, and inference.

---

### Phase 6: Cross-Domain Applications (Target: 28 Weeks)

**Goal:** Extend the library's capabilities to non-health applications and domains.

1.  **Environmental Economics VOI:**
    *   **Status: `Not Started`**
    *   Extension of current methods to environmental valuation contexts.
2.  **Engineering Design VOI:**
    *   **Status: `Not Started`**
    *   Application to design optimization under uncertainty.
3.  **Financial Risk Management VOI:**
    *   **Status: `Not Started`**
    *   Integration with financial risk models and portfolio theory.
4.  **Marketing Research VOI:**
    *   **Status: `Not Started`**
    *   Application to customer analytics and market research.
5.  **Public Policy VOI:**
    *   **Status: `Not Started`**
    *   Extension to policy evaluation and social program assessment.

---

### Phase 7: Computational Advances (Target: 32 Weeks)

**Goal:** Implement cutting-edge computational methods for improved performance and scalability.

1.  **Federated Learning for VOI:**
    *   **Status: `Not Started`**
    *   Methods for distributed VOI calculations across multiple sites.
2.  **Edge Computing VOI:**
    *   **Status: `Not Started`**
    *   Lightweight VOI methods for resource-constrained environments.
3.  **Blockchain Integration:**
    *   **Status: `Not Started`**
    *   Distributed consensus methods for VOI in decentralized systems.
4.  **Quantum Computing Integration:**
    *   **Status: `Not Started`**
    *   Next-generation computational methods for VOI calculations.
5.  **Real-Time Adaptive VOI:**
    *   **Status: `Not Started`**
    *   Streaming VOI calculations with incremental updates.

---

### Phase 8: Validation and Benchmarking Framework (Target: 36 Weeks)

**Goal:** Establish a comprehensive validation and benchmarking framework for the library.

1.  **Standardized Test Suites:**
    *   **Status: `Not Started`**
    *   Development of standard test problems for VOI method validation.
2.  **Cross-Language Benchmarking:**
    *   **Status: `Not Started`**
    *   Comparison with R, Julia, and other VOI implementations.
3.  **Performance Profiling Tools:**
    *   **Status: `Not Started`**
    *   Tools for analyzing computational performance of VOI methods.
4.  **Reproducibility Frameworks:**
    *   **Status: `Not Started`**
    *   Integration with reproducibility tools and workflows.

---

### Phase 9: Ecosystem, Community & Future Ports (Ongoing)

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

---

## Completed Methods Documentation

All methods currently implemented in voiage are documented with their mathematical foundations:

### Core VOI Methods:
- ✅ Expected Value of Perfect Information (EVPI)
- ✅ Expected Value of Partial Perfect Information (EVPPI)
- ✅ Expected Value of Sample Information (EVSI)
- ✅ Expected Net Benefit of Sampling (ENBS)

### Advanced VOI Methods:
- ✅ Structural Uncertainty VOI
- ✅ Network Meta-Analysis VOI
- ✅ Adaptive Design VOI
- ✅ Portfolio Optimization
- ✅ Value of Heterogeneity
- ✅ Sequential VOI
- ✅ Calibration VOI
- ✅ Observational Data VOI
- ✅ Sample Information Methods

### Emerging Methods (Future Implementation):
- ⬜ Real Options Valuation in Healthcare
- ⬜ Multi-Criteria Decision Analysis (MCDA) VOI
- ⬜ Dynamic VOI Methods
- ⬜ Value of Correlation Information
- ⬜ Machine Learning Integration for VOI
- ⬜ Causal Inference VOI Methods
- ⬜ Precision Medicine VOI
- ⬜ Implementation Science VOI
- ⬜ Global Health VOI
- ⬜ Rare Disease VOI
- ⬜ Cross-Domain Applications
- ⬜ Computational Advances

## References and Citations

All methods are properly referenced with original source publications and key methodological developments, ensuring proper attribution and facilitating academic rigor.

This extended roadmap positions `voiage` as a comprehensive, cutting-edge platform for VOI analysis across multiple domains and applications, ensuring its relevance for current and future research needs.