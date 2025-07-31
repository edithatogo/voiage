# voiage Project Roadmap (v2)

## Vision

To establish `voiage` as the premier, cross-domain, high-performance library for Value of Information analysis. It will be distinguished by its analytical rigor, computational performance, and exceptional user experience. In the long term, its core, language-agnostic API will be ported to R and Julia, making it a true cross-platform standard.

---

### Phase 1: Foundation & API Refactoring (Target: 8 Weeks)

**Goal:** Solidify the library's foundation by implementing a more robust, extensible, and user-friendly API. Address the key structural and software engineering weaknesses identified in the initial reviews.

1.  **Object-Oriented API Redesign & Functional Wrappers:**
    *   Refactor the current function-based API into an object-oriented one centered around a `DecisionAnalysis` class. This class will encapsulate the model, data, and decision options, providing a fluent interface (e.g., `analysis.evpi()`, `analysis.evsi(...)`).
    *   Provide a simple, stateless functional API (e.g., `voiage.evpi(...)`) as a lightweight convenience layer. These functions will act as wrappers, calling the core OO-API internally.
2.  **Domain-Agnostic Data Structures:**
    *   Generalize the core data structures to be domain-agnostic. Rename `PSASample` -> `ParameterSet`, `TrialArm` -> `DecisionOption`, `NetBenefitArray` -> `ValueArray`. This is critical for attracting a broader user base.
3.  **High-Performance Backend (JAX):**
    *   Implement a JAX-based computational backend for core calculations (EVPI, EVPPI). This will provide a significant performance advantage and serve as a key differentiator.
    *   Create a backend dispatch system (e.g., `voiage.set_backend('jax')`) to allow users to switch between `numpy` and `jax`.
4.  **CI/CD & Documentation Website:**
    *   Establish a full CI/CD pipeline in GitHub Actions to automate testing, linting, and code coverage checks for every PR.
    *   Deploy a professional documentation website using Sphinx and a modern theme (e.g., Furo), hosted on GitHub Pages.
5.  **Community Guidelines:**
    *   Create a comprehensive `CONTRIBUTING.md` detailing the development process, code style, and PR submission guidelines.

---

### Phase 2: State-of-the-Art Health Economics Core (Target: 12 Weeks)

**Goal:** Implement the most critical features for the initial target audience (health economists) to make `voiage` a compelling alternative to existing tools.

1.  **Robust EVSI Implementation:**
    *   Move beyond the current stub to a full-featured two-loop EVSI calculation.
    *   Implement proper Bayesian updating, starting with conjugate priors and extending to MCMC-based updates via integration with NumPyro.
    *   Integrate advanced metamodels (Gaussian Processes, BART) for the inner loop, moving beyond simple linear regression.
2.  **Network Meta-Analysis (NMA) VOI:**
    *   Implement `evsi_nma`, a critical and highly-demanded feature in HTA. This will involve handling multivariate parameter distributions for treatment effects.
3.  **Validation & Benchmarking:**
    *   For each core method, create a validation notebook that replicates the results of a published study or an example from an established R package (e.g., `BCEA`, `voi`).
    *   Publish performance benchmarks comparing the `numpy` and `jax` backends.
4.  **Advanced Plotting Module & Core Examples:**
    *   Implement a `plot` module capable of generating publication-quality graphics for CEACs, CE-planes, VOI curves, and EVPPI surfaces.
    *   Develop a detailed tutorial notebook for the core health economics use case (e.g., new drug evaluation), including synthetic data generation and interpretation of results.

---

### Phase 3: Advanced Methods & Cross-Domain Expansion (Target: 16 Weeks)

**Goal:** Broaden the library's capabilities to cover advanced VOI methods and actively support non-health applications.

1.  **Portfolio Optimization:**
    *   Implement a robust `portfolio_voi` method, using the new OO-API. Support both greedy algorithms and integer programming (via optional dependencies like `pulp`).
2.  **Structural & Sequential VOI:**
    *   Implement the placeholder methods for `structural_voi` and `sequential_voi`. These are crucial for complex policy and scientific modeling decisions.
3.  **Cross-Domain Example Notebooks:**
    *   Develop at least two detailed tutorial notebooks showcasing `voiage` on non-health problems (e.g., an environmental policy decision on sea-level rise, a business strategy problem on portfolio optimization). This is critical for demonstrating the library's domain-agnostic capabilities.
4.  **XArray Integration:**
    *   Begin integrating `xarray` into the core data structures to provide labeled dimensions, enhancing usability for complex, multi-dimensional analyses.

---

### Phase 4: Ecosystem, Community & Future Ports (Ongoing)

**Goal:** Grow the user and contributor community and lay the groundwork for R and Julia versions.

1.  **Community Engagement:**
    *   Actively manage the GitHub repository, using Issues, Discussions, and Projects to engage with the community.
    *   Publish "State of `voiage`" updates on the documentation website's blog.
    *   Present the library at academic and industry conferences.
2.  **Language-Agnostic API Specification:**
    *   Formalize the core `DecisionAnalysis` object and its data structures using a language-agnostic format (e.g., JSON Schema). This specification will serve as the formal blueprint for the R and Julia ports.
3.  **Planning for R/Julia Ports:**
    *   Begin prototyping the R port (`rvoiage`?) using `R6` classes to mirror the Python OO-API. The focus will be on API consistency and leveraging R's strengths in statistics and plotting.
    *   Investigate Julia's multiple dispatch system as a powerful paradigm for the Julia port (`Voiage.jl`?), potentially leading to even more elegant and performant code.
