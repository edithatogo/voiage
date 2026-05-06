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
    *   Full CI/CD pipeline: uv, Ruff, CodeQL, Benchmarks, Sphinx docs, GitHub Pages, automated publishing to PyPI/TestPyPI, plus conda-forge feedstock recipe updates with external feedstock approval.
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
    *   TestPyPI → PyPI publishing on `v*` tags, plus conda-forge feedstock recipe updates with the external feedstock merge remaining outside this repository.
    *   Polyglot release workflows now publish npm, crates.io, and NuGet packages and attach GitHub release artifacts for Go, Julia, and R bindings. The R package currently ships source archives on GitHub Releases, while CRAN remains the maturity target and r-universe remains an optional external indexing path; registry-side indexing or approval still depends on the external ecosystem for conda-forge, CRAN/r-universe, and the Julia General registry.
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
    *   Keep the R binding documentation track explicit: the package help pages, a narrative vignette, and a deterministic PDF reference manual are part of the package docs surface, and the completed track is archived with the build/verification guidance centered on `tools/build-manual.R` and the non-interactive `R CMD check --as-cran --no-manual` flow.
    *   Keep the polyglot tutorial surface explicit so the Python notebooks, the R vignette/manual, and the non-Python binding walkthroughs stay aligned around the same canonical use cases; the track is now complete and archived, with the repo-level smoke checks covering the binding walkthrough READMEs.

---

### Phase 5: Spec, Fixtures & Polyglot Bindings 📋 **PLANNED**

**Goal:** Mature the library into a broadly usable core analysis engine with stable cross-language contracts.

1.  **Core API Specification:**
    *   Define method signatures, schema invariants, and error behavior for the public VOI surface.
    *   Covered by Conductor tracks: `core-api-spec-foundation`, `canonical-schemas-core-contracts`.
2.  **Conformance Fixtures:**
    *   Build canonical input/output fixtures that every binding must pass before release.
    *   Covered by Conductor tracks: `cross-language-conformance-fixtures`, `python-cleanup-against-spec`.
3.  **Python Cleanup and Stabilization:**
    *   Finish the Python-side normalization needed to make the canonical API implementation simple and durable.
    *   Covered by Conductor track: `python-cleanup-against-spec`.
4.  **First External Bindings:**
    *   Deliver R and Julia bindings against the same contract, then extend to TypeScript, Go, and Rust if adoption warrants it.
    *   Publishing targets must be planned with the implementation:
        - Python: PyPI, TestPyPI, and conda-forge feedstock recipe updates, with the feedstock PR/merge remaining external.
        - R: GitHub Releases for early source distribution, CRAN when mature, and optional r-universe indexing; the package docs story includes a deterministic vignette and PDF manual built from the same source tree, while external registry approval remains outside the repository.
        - Julia: Julia General registry with TagBot sync and external registry registration.
        - TypeScript: npm with provenance.
        - Go: tagged Go modules consumable through the Go module proxy, with GitHub Releases for release notes/artifacts.
        - Rust: crates.io.
        - .NET: NuGet, targeting .NET 11 (`net11.0`).
    *   CI/CD must be language-specific and release-aware for every binding:
        - Build, lint/format, type/static checks, unit tests, docs checks, and shared conformance fixtures.
        - Package dry-run validation on pull requests.
        - Trusted or token-scoped publishing on version tags/releases.
        - Registry-specific provenance and changelog generation where supported.
    *   Covered by Conductor tracks: `cross-language-conformance-fixtures`, `first-external-bindings_20260430`, and future binding-specific tracks as they are added.
    *   Contract semantics, maturity metadata, and extension rules are covered by `numerics-diagnostics-extension-model`.

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
    *   Covered by Conductor track: `lifecourse-integration-contract_20260429`.
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
    *   Covered by Conductor track: `ecosystem-module-incubation_20260429`.
3.  **HEOR Module Naming Brainstorm:**
    *   **Status: `📋 Planned`**
    *   Keep the candidate sibling module names short and consistent:
        `calibrate`, `evidence`, `process`, `report`, `registry`, `workflow`,
        `quality`, `engines`, and `heoml`.
    *   Treat PM4Py as an ecosystem-only process-mining capability.
    *   Require CLI support for every future module and decide whether MCP adds
        value on a module-by-module basis.
    *   Keep the naming discussion as brainstorming, not a commitment to add
        every module now.
    *   Covered by Conductor track: `heor_module_naming_brainstorm_20260429`.
    *   CLI and docs implementation support for the ecosystem-facing surface is
        covered by `cli-integration-testing` and `docs-developer-experience`.

---

### Phase 7: SOTA VOI Frontier 📋 **PLANNED**

**Goal:** Move `voiage` beyond parity with existing VOI packages by adding
frontier methods that are rarely or not at all available in general-purpose
VOI tooling.

1.  **Value of Perspective:**
    *   **Status: `🚧 Experimental`**
    *   Treat decision perspective as an explicit analysis dimension rather than
        a hidden modelling assumption.
    *   Compare payer, societal, patient, provider, regulator, equity-weighted,
        and custom stakeholder perspectives side by side.
    *   Compute perspective-specific optimal strategies, cross-perspective
        regret, value of switching perspective, robust consensus strategies,
        and Pareto/non-dominated strategies across perspectives.
    *   Experimental Python API, CLI, plotting helper, and v1 contract scaffold
        are available; deterministic screening-program fixtures now anchor the
        contract, and stable status still requires cross-language conformance.
2.  **Distributional, Equity, and Implementation-Adjusted VOI:**
    *   **Status: `🚧 Experimental`**
    *   Extend Value of Heterogeneity toward distributional and equity-weighted
        VOI.
    *   Add implementation-adjusted VOI for uptake, adherence, coverage,
        implementation delay, and implementation uncertainty.
    *   Experimental Python APIs now exist for both families; deterministic
        fixture sets now anchor both contracts, and cross-language parity is
        the next gate.
3.  **Preference, Validation, Threshold, and Robust VOI:**
    *   **Status: `🔄 In Progress`**
    *   Implement value of preference information and value of individualized care.
    *   The preference heterogeneity contract scaffold now lives under
        `specs/frontier/preference/v1/` and mirrors the multi-profile analysis
        shape used by Value of Perspective; the runtime surface, CLI
        entrypoint, docs wiring, and fixture-backed conformance are
        implemented, and the remaining work is any cross-language parity
        follow-through.
    *   Add value of external validation and model-discrepancy reduction.
    *   The model-validation contract scaffold now lives under
        `specs/frontier/validation/v1/` and mirrors the multi-profile analysis
        shape used by Value of Perspective. The runtime slice, fixture-backed
        conformance slice, CLI entrypoint, and docs wiring are implemented in
        `model-validation-voi_20260506`.
    *   Add threshold/tipping-point VOI and robust or ambiguity-aware VOI.
    *   The threshold contract scaffold now lives under
        `specs/frontier/threshold/v1/` and mirrors the multi-profile analysis
        shape used by Value of Perspective. The runtime slice,
        fixture-backed conformance slice, CLI entrypoint, and docs wiring are
        implemented in `threshold-robust-voi_20260506`.
    *   Extend sequential VOI toward dynamic real-options style decisions where
        delay, irreversibility, and policy lock-in affect value.
    *   Dynamic real-options VOI is now tracked as a dedicated frontier phase
        in `dynamic-real-options-voi_20260430` and mirrored in the frontier
        umbrella track with staged-evidence and policy-lock-in subphases. The
        contract scaffold now lives under `specs/frontier/dynamic-real-options/v1/`.
4.  **Adjacent Frontier Extensions:**
    *   **Status: `📋 Planned`**
    *   Triage causal-identification, transportability, and external-validity
        VOI for target-population decision problems.
    *   Triage data-quality, measurement-error, data-acquisition, privacy, and
        linkage VOI where the information source has operational constraints.
    *   Triage computational VOI, value of model refinement, expert-elicitation
        VOI, and evidence-synthesis design VOI as possible extension tracks.
    *   These families are now split into explicit follow-on phases in
        `adjacent-frontier-extensions_20260430` and mirrored in the frontier
        umbrella track so they can be implemented and fixture-backed
        independently. The causal-identification, transportability, and
        external-validity family now has a contract scaffold under
        `specs/frontier/causal-transportability/v1/`, and the data-quality,
        measurement-error, privacy, and linkage family now has a contract
        scaffold under `specs/frontier/data-quality/v1/`. The computational
        and model-refinement family now has a contract scaffold under
        `specs/frontier/computational/v1/`, and the expert-elicitation and
        evidence-synthesis design family now has a contract scaffold under
        `specs/frontier/expert-synthesis/v1/`. The shared maturity and handoff
        conventions for all adjacent families now live under
        `specs/frontier/shared-maturity/v1/`, and deterministic normative
        fixtures are now committed for the causal, data-quality,
        computational, and expert-synthesis adjacent families.
5.  **Documentation and Evidence:**
    *   **Status: `📋 Planned`**
    *   Maintain the frontier-method rationale in `docs/sota_voi_frontier.md`.
    *   Add CHEERS-VOI reporting metadata, schemas, deterministic fixtures,
        examples, CLI coverage, and method maturity metadata before marking
        frontier methods stable.
    *   The current docs now reflect the fixture-backed Value of Perspective,
        validation, threshold, distributional/equity, and implementation-
        adjusted slices, and the experimental result payloads now carry shared
        CHEERS-VOI reporting objects. The reporting envelope also now covers
        the standard scalar CLI outputs (EVPI, EVPPI, EVSI, ENBS) and adjacent
        summary outputs such as CEAF, dominance, and Value of Heterogeneity.
        The remaining work is to expand those fields to the rest of the
        frontier families. Value of Perspective, validation, threshold,
        distributional/equity, and implementation-adjusted VOI now each have
        deterministic fixture sets anchoring their contracts.
    *   Covered by Conductor track: `sota-voi-frontier_20260429`.

### Phase 8: Rust Core Migration Program 📋 **PLANNED**

**Goal:** Move `voiage` toward a Rust execution core with Python as the primary
façade, thin language bindings/adapters over the same contract, and
scalar-first profiling while keeping the cross-language contract stable and the
binding story explicit.

1.  **Migration Foundation:**
    *   Decide the Rust-core boundary, workspace policy, and compatibility model.
    *   Rust is the authoritative execution core for deterministic VOI kernels,
        shared result contracts, and serialization behavior; Python remains the
        façade for CLI, orchestration, plotting, and compatibility wrappers.
    *   Covered by Conductor track: `rust-core-migration-foundation_20260504`.
2.  **Domain Model Port:**
    *   Port the stable data model, result envelopes, diagnostics, and reporting metadata into Rust.
    *   Covered by Conductor track: `rust-core-domain-model_20260504`.
3.  **Numerics Engine Port:**
    *   Port the deterministic VOI methods and fixture-backed kernels into Rust.
    *   Completed by Conductor track: `rust-core-numerics-engine_20260504` (archived).
4.  **Scalar-First Profiling And Backend Strategy:**
    *   Establish scalar-first CPU, memory, throughput, SIMD, GPU, and accelerator feasibility baselines.
    *   Covered by Conductor track: `rust-core-performance-and-profiling_20260504`.
5.  **Bindings And Release Adaptation:**
    *   Recast Python as the façade and R, Julia, TypeScript, Go, and .NET as
        thin bindings/adapters over the Rust core, then update the release
        matrix accordingly.
    *   Covered by Conductor track: `rust-core-bindings-and-release_20260504`.

### Phase 9: Rust EVSI Stochastic Kernel Follow-On ✅ **COMPLETED**

**Goal:** Promote the EVSI sample-information computation from a Rust summary
contract into a Rust-owned stochastic kernel while preserving the existing
contract, diagnostics, and reporting envelope. The summary envelope is already
owned by Rust core; this phase is kernel-only.

1.  **Kernel Contract And Fixture Harness:**
    *   Define the Rust EVSI kernel inputs, output shape, and fixture-backed
        parity harness.
    *   Completed by Conductor track: `rust-evsi-stochastic-kernel_20260506` (archived).
2.  **Two-Loop Kernel Port:**
    *   Port the stochastic EVSI kernel into Rust and validate it against the
        Python reference and deterministic fixtures.
    *   Completed by Conductor track: `rust-evsi-stochastic-kernel_20260506` (archived).
3.  **Approximation Policy And Optional Kernel Variants:**
    *   Decide which EVSI approximation variants belong in Rust core versus a
        façade-side implementation.
    *   Completed by Conductor track: `rust-evsi-stochastic-kernel_20260506` (archived).
4.  **Benchmark Baseline And Handoff:**
    *   Record representative EVSI kernel baselines and document the handoff
        contract for future optimization work.
    *   Completed by Conductor track: `rust-evsi-stochastic-kernel_20260506` (archived).
