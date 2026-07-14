# `voiage` Task List

This document lists the actionable tasks for `voiage` development. Agents should pick tasks from the "To Do" list.

## To Do
*None at the moment.*

## In Progress

*None at the moment.*

## Done

*   [x] Archive completed Conductor track records and repair their regression
    test and documentation references.
    *   Moved all completed registered tracks from `conductor/tracks/` to
        `conductor/archive/`, updated the registry/index links, and verified
        the archived evidence and status tests.

*   [x] Remediate the current lockfile dependency advisories.
    *   Updated the Python lockfile through the patched Jupyter Server,
        JupyterLab, mistune, soupsieve, bleach, tornado, idna, urllib3, and
        pytest releases; upgraded Astro/Starlight within a compatible major
        line; pinned the transitive esbuild resolution to a patched release;
        and updated the Starlight configuration for the current APIs.
    *   Verified with `pnpm audit --prod`, Astro check/build, and the complete
        tox matrix including Python 3.10 through 3.14, minimum/maximum
        dependency environments, coverage, docs, type, contract, and harness
        gates.

*   [x] Set up maximal repository harness engineering and GitHub security and
    quality controls.
    *   Added the fail-closed workflow/governance harness and tox gate,
        immutable action pins, least-privilege permissions, CodeQL gate repair,
        dependency review, OpenSSF Scorecard, Zizmor auditing, Dependabot, and
        release provenance attestation.
    *   Applied repository merge/signoff settings and the active
        `main-maximal-quality` GitHub ruleset requiring pull requests,
        signed commits, linear history, resolved review threads, and required
        CI, dependency-review, and CodeQL checks. The review count is zero for
        the repository's single-maintainer operating model.
    *   Kept organization-owned policy, environment approvals, and unavailable
        plan-dependent secret-scanning enhancements explicitly external.

*   [x] Remove tracked generated artifacts and reconcile roadmap status labels.
    *   Removed committed pytest temp output, coverage reports, macOS metadata,
        legacy egg-info generated artifacts, and R CMD check output from
        version control.
    *   Removed tracked backup snapshots and added ignore/test coverage for
        common editor backup suffixes.
    *   Removed root-level legacy test scratch files and test result logs so
        active verification remains anchored under `tests/`.
    *   Removed stale root completion/coverage report mirrors, obsolete
        local-path mutation/demo scripts, and generated result outputs.
    *   Removed generated nbconvert notebooks and normalized local file links
        to portable repository-relative paths.
    *   Removed stale Qoder, repository-map, and legacy roadmap mirror
        documents that duplicated canonical project status files and described
        outdated placeholder or v0.1-era state.
    *   Removed remaining root Qoder quest output and duplicate planning,
        branch-architecture, CLI-changelog, repository-structure, and outdated
        todo mirrors.
    *   Removed legacy root JAX audit/setup, coverage-analysis, benchmark,
        verification, review, and testing-approach scaffolding that was not
        wired into maintained package, docs, tox, or CI surfaces.
    *   Fixed the Taskipy profile command and profiling script so the entry
        point targets the tracked script and current public API.
    *   Removed root-level generated CLI sample CSVs and pointed the README
        and CLI example script at the maintained `examples/cli_samples/`
        fixtures.
    *   Added the XML parser used by the ecosystem integration module to the
        base runtime dependencies and made the CLI example fail fast on
        command failures.
    *   Rewrote the stale root product, technology stack, workflow, and product
        guideline mirrors so they match the current roadmap, AGENTS workflow,
        Sphinx/Starlight documentation boundary, and >90% coverage gate.
    *   Fixed Sphinx heading and nested-list formatting in HPC developer docs
        so the warning-as-error docs gate passes.
    *   Added repo-local ignore coverage for macOS, tooling-cache, coverage,
        and temporary-output paths so local verification does not pollute
        future cleanup passes.
    *   Updated the roadmap so completed Conductor/spec/ecosystem phases are
        marked as repository-complete while external registry and hardware
        evidence gates remain explicit.

*   [x] Add free-runner pre-silicon FPGA/ASIC evidence tracks.
    *   Added a deterministic fixed-point EVPI-style RTL kernel, CPU fixture,
        manifest generator, committed FPGA/ASIC probe manifests, and GitHub
        Actions smoke workflow.
    *   Updated the FPGA and ASIC Conductor tracks so free CI evidence is
        first-pass progress while physical board, Tiny Tapeout, SkyWater MPW,
        and fabricated-silicon runtime remain external follow-up gates.

*   [x] Clean up HPC and Conductor tracking records after Colab GPU/TPU evidence capture.
    *   Normalized the HPC roadmap, accelerator/distribution docs, ASIC feasibility notes, and completed spec-track metadata so they distinguish visibility/parity evidence from production speedup evidence.
    *   Added regression coverage for Colab evidence paths, hardware-scope wording, and Conductor status contradictions.

*   [x] Add a Colab accelerator validation notebook for the HPC workflow.
    *   Added a compact notebook that clones the repo in Colab, installs the
        package, probes JAX CPU/GPU/TPU device visibility, checks EVPI parity,
        writes an evidence packet, and confirms FPGA/ASIC remain explicit
        placeholder adapters.
    *   Captured and persisted successful Colab T4 GPU and v5e TPU evidence
        payloads under the HPC acceleration abstraction handoff.
    *   Updated the HPC docs and Conductor feasibility records to cite the
        GPU/TPU evidence while keeping production speedup, FPGA, and ASIC
        claims evidence-gated.

*   [x] Create the SOTA strategy orchestration and dependency matrix guide.
    *   Codified the dependency graph, shared gates, and parallel lanes for the strategy work, then linked the guide into the developer docs.

*   [x] Complete the SOTA packaging review readiness, HPC distribution and
    acceleration strategy, Rust-core ABI and migration strategy, and polyglot
    repo/docs architecture tracks.
    *   Closed the strategy-layer roadmap work for submission readiness, HPC
        distribution policy, Rust ABI policy, and docs/repo structure, and
        captured the current-state / future-state summary in the roadmap.

*   [x] Make the community engagement surface explicit with support, security, and contributor guidance.
    *   Added repository support, code of conduct, and security policy documents, plus a support-question issue template and community guide links in the main docs.
    *   Covered by Conductor track: `community-engagement-support_20260507`.

*   [x] Make versioning SOTA by centralizing the canonical release version, synchronizing binding manifests, and adding a validator gate.
    *   Recorded the canonical versioning policy in the developer and release docs, synchronized the binding manifests to the canonical repo version, and wired the validator into tox and CI.
    *   Covered by Conductor track: `unified-versioning-policy_20260507`.

*   [x] Create the Starlight documentation platform track and record the versioning/plugin baseline in the roadmap and conductor setup.
    *   Added a dedicated Starlight docs-platform track, documented the versioning and plugin baseline, and marked the migration boundary against the current Sphinx docs.

*   [x] Align the validation and threshold frontier maturity labels and their docs.
    *   Updated the runtime, schemas, fixtures, examples, user guide, and setup docs so both surfaces now consistently report fixture-backed status.

*   [x] Complete CLI integration and docs wiring for the model-validation and threshold VOI tracks.
    *   Added `calculate-validation` and `calculate-threshold`, updated the frontier and migration docs, and closed the corresponding track checkpoints after the runtime, CLI, and fixture-backed slices landed.

*   [x] Implement the model-validation VOI runtime and fixture-backed conformance slices.
    *   Added the model-validation runtime shape, deterministic fixture-backed expectations, and the shared contract scaffold; the CLI entrypoint and docs wiring are now included in the completed track.
*   [x] Implement the threshold, tipping-point, and robust VOI runtime and fixture-backed conformance slices.
    *   Added the threshold runtime shape, deterministic fixture-backed expectations, and the shared contract scaffold; the CLI entrypoint and docs wiring are now included in the completed track.
*   [x] Create the Rust EVSI stochastic-kernel follow-on track and keep the EVSI summary boundary explicit.
    *   The Rust core already owns the EVSI summary contract; the new track now exists to port the stochastic kernel underneath that contract, add fixture-backed parity tests, and document the approximation policy.

*   [x] Add the Rust-core handoff, parallelism, and accelerator guidance pages.
    *   Documented the Rust-core boundary, the Rayon/SIMD feasibility policy,
        and the GPU/TPU/custom-circuit feasibility limits in the developer
        guide, then linked the new pages from the main docs index and the Rust
        binding README.
*   [x] Archive the Rust-core performance and profiling track after the new
    guidance pages closed the remaining feasibility work.
    *   Moved the completed track into `conductor/archive/` and updated the
        registry so the active backlog reflects the remaining numerics work
        only.
*   [x] Archive the Rust-core numerics engine track at the current contract
    boundary.
    *   Marked the EVSI summary boundary explicit, moved the completed track
        into `conductor/archive/`, and updated the registry so the active
        backlog no longer shows the numerics engine as open.

*   [x] Add the Rust EVPPI summary contract and close the Rust profiling memory/throughput measurement checkpoint.
    *   Implemented a deterministic Rust EVPPI analysis envelope and finalized the memory/throughput profiling phase-2 checkpoint for the Rust core.

*   [x] Add the Rust EVSI summary contract and the Rust profiling memory/throughput artifact and CI gate slices.
    *   Implemented a deterministic Rust EVSI analysis envelope, added a machine-readable memory/throughput profiling artifact for the scalar baseline, and wired the Rust benchmark job into CI.

*   [x] Reconcile the Rust numerics and profiling track bookkeeping.
    *   Closed the benchmark CI checkpoint and narrowed the remaining Rust-core open work to EVPPI, EVSI/frontier-adjacent kernels, and the unfinished profiling measurements/guidance phases.

*   [x] Archive the completed Rust-core migration foundation, domain model, bindings/release, and SOTA VOI frontier tracks after their final review checkpoints cleared.
    *   Moved the finished track folders into `conductor/archive/` and updated the registry so the completed work no longer appears as active backlog.

*   [x] Advance the Rust-core migration with the domain-model handoff, deterministic summary methods, scalar CPU benchmark baseline, and preference heterogeneity surface.
    *   Tightened the Rust core boundary, added deterministic dominance/CEAF/heterogeneity contracts in `voiage-core`, documented the migration handoff, and wired the preference front-end into the public Python API.

*   [x] Start the Rust-core migration program with the core boundary, domain model, deterministic scalar methods, and profiling baseline.
    *   Locked in Rust as the authoritative execution core, implemented the core data/reporting envelopes, added deterministic EVPI and ENBS kernels, and seeded a scalar CPU benchmark baseline for the Rust crate.

*   [x] Create the Rust-core migration program tracks for foundation, domain model, numerics, profiling, and bindings/release.
    *   Moved the execution core toward Rust in a staged way while keeping the Python façade and thin bindings/adapters aligned to the same contract.
    *   Kept profiling scalar-first and made the migration plan explicit about core ownership versus binding ownership.

*   [x] Complete the remaining documentation and CLI integration track bookkeeping.
    *   Archived the docs-developer-experience and cli-integration-testing tracks after closing the remaining doc, notebook, link-check, benchmark, and CLI branch-coverage slices.

*   [x] Add the docs-developer-experience notebook scaffold for EVPI, EVPPI, EVSI, and the getting-started tutorial.
    *   Added runnable validation/tutorial notebooks under `examples/` and verified they execute with `nbconvert`.
*   [x] Create the advanced methods tutorial notebook.
    *   Added `examples/advanced_methods.ipynb` covering structural VOI, NMA VOI, adaptive trial VOI, portfolio VOI, sequential VOI, efficient EVSI, CEAF, and extended dominance.

*   [x] Add the CLI benchmark and regression scaffolding for the core VOI methods.
    *   Added a lightweight `tests/benchmarks/` benchmark suite, backend comparison coverage, and deterministic regression payloads under `tests/regression_data/`.
*   [x] Add the end-to-end workflow, DecisionAnalysis, and cross-domain integration tests.
    *   Added smoke coverage for the EVPI/EVPPI/EVSI-to-plot path, the main `DecisionAnalysis` methods, and healthcare/financial/environmental factory entrypoints.
*   [x] Finish the R package documentation/manual track for the PDF reference manual and narrative vignette.
    *   Kept the package help pages, vignette, and deterministic PDF manual aligned with the release and verification guidance, then archived the completed track.
*   [x] Create the polyglot tutorial surface track for notebooks, vignettes, and language-specific walkthroughs.
    *   Aligned the Python notebooks, the R vignette/manual, and the non-Python binding walkthrough READMEs around the same canonical VOI use cases, then added repo-level smoke checks for the binding tutorials.

*   [x] Add the core API extension-evolution contract and archive the completed numerics track.
    *   Defined the additive-extension, versioning, and deprecation rules in `specs/core-api/extension-evolution.md`, then moved the completed numerics track into the archive registry.

*   [x] Create the missing user-facing method and reference pages for the docs surface.
    *   Added `docs/methods/`, `docs/plotting/index.rst`, `docs/cli_reference.rst`, `docs/data_structures.rst`, and `docs/backends.rst`, then wired them into the main docs index.

*   [x] Archive the completed HEOR naming track and clean the Conductor registry of duplicate live entries.
    *   Moved the naming brainstorm track into the archive and removed the redundant live duplicates for the already-complete cross-language, Python cleanup, and ecosystem tracks.

*   [x] Create the method implementation guide for the stable core VOI surface.
    *   Added a contributor-facing guide for extending the stable EVPI, EVPPI, EVSI, ENBS, CEAF, dominance, and heterogeneity methods, with a signature template and implementation checklist.

*   [x] Create the developer onboarding architecture and contribution guides.
    *   Added `docs/developer_guide/architecture.rst` and `docs/developer_guide/how_to_contribute.rst`, and wired them into the developer-guide index.

*   [x] Complete the Phase 1 core API conformance-fixture scaffold with a deterministic input bundle and runner helpers.
    *   Added the normative input bundle, wired the fixture manifest to input/output pairs, and taught the validator to load fixture cases directly.

*   [x] Align the polyglot release documentation and Conductor bookkeeping with the actual registry automation.
    *   Clarified which release paths are automated in-repo and which still depend on external feedstocks or registry processes.

*   [x] Add Vale prose linting for Markdown and reStructuredText documentation.
    *   Added a repo-local Vale config and style set, hooked the CI prose job to the real docs paths, and documented the command in `CONTRIBUTING.md`.

*   [x] Add CHEERS-VOI reporting objects and contract examples for the experimental frontier methods.
    *   Added shared reporting payload helpers, attached them to Value of Perspective, distributional/equity VOI, and implementation-adjusted VOI, and mirrored the fields in the frontier contract examples and schemas.

*   [x] Add CHEERS-VOI reporting metadata, structured result fields, and reproducibility outputs for every frontier method family.
    *   Added reporting envelopes to the remaining frontier contracts so every frontier result payload now carries the shared CHEERS-VOI baseline.

*   [x] Close the remaining SOTA frontier automated review checkpoints.
    *   Marked the documentation/release, dynamic real-options, and adjacent frontier review checkpoints complete after the corresponding track work landed.

*   [x] Define the cross-language core API runner contract and smoke validator.
    *   Added the runner guide for Python, R, Julia, and future TypeScript, Go, and Rust bindings, plus a narrow smoke validator for the fixture catalog layout.

*   [x] Add deterministic fixtures and reviewable example payloads for dynamic real-options VOI.
    *   Added the dynamic real-options contract scaffold, fixtures, registry entry, and contract tests.

*   [x] Define causal-identification, transportability, and external-validity VOI contracts.
    *   Added the causal-transportability contract scaffold with schema and example payloads for source-to-target population shifts.

*   [x] Define data-quality, measurement-error, data-acquisition, privacy, and linkage VOI contracts.
    *   Added the data-quality contract scaffold with schema and example payloads for acquisition costs, privacy constraints, and linkage weights.

*   [x] Define computational VOI and model-refinement VOI contract scope.
    *   Added the computational contract scaffold with schema and example payloads for compute budgets, approximation error, and refinement value.

*   [x] Define expert-elicitation VOI and evidence-synthesis design VOI contract scope.
    *   Added the expert-synthesis contract scaffold with schema and example payloads for elicitation costs and synthesis penalties.

*   [x] Define the shared maturity labels, diagnostics, reporting metadata, and handoff path for the adjacent frontier families.
    *   Added the adjacent-frontier shared-maturity contract scaffold with maturity labels, fixture-backed criteria, next-step requirements, and CHEERS-VOI reporting conventions.

*   [x] Add deterministic fixtures and registry entries for the adjacent frontier families.
    *   Added deterministic normative fixtures and registry entries for the causal, data-quality, computational, and expert-synthesis adjacent frontier contracts.

*   [x] Add CHEERS-VOI reporting objects to Value of Heterogeneity.
    *   Added the same reporting payload helper to the base distributional surface so the reporting model extends naturally from Value of Heterogeneity into distributional/equity VOI.

*   [x] Extend the CHEERS-VOI reporting baseline to CEAF and dominance.
    *   Added shared reporting payloads to the frontier summary outputs that feed the main cost-effectiveness analysis workflow.

*   [x] Extend the CHEERS-VOI reporting baseline to core scalar CLI outputs.
    *   Added shared reporting payloads to EVPI, EVPPI, EVSI, and ENBS JSON/CSV result output so the main command surface has a consistent reporting envelope.

*   [x] Add deterministic screening-program fixtures for Value of Perspective.
    *   Added a normative fixture manifest plus input/output payloads that anchor the fixture-backed CLI contract for the screening-program comparison surface.

*   [x] Implement distributional/equity VOI and implementation-adjusted VOI as experimental frontier methods.
    *   Added `value_of_distributional_equity` and `value_of_implementation` with `DecisionAnalysis` wrappers, curated exports, and regression tests.
    *   Kept the results explicit about subgroup weights, implementation uptake, adherence, coverage, delay, uncertainty, and maturity metadata.
    *   Added versioned experimental contract folders with deterministic JSON schemas and example payloads for both frontier families.

*   [x] Define the ecosystem module incubation policy for `voiage`.
    *   Documented the `voiage` role alongside `lifecourse`, `innovate`, `mars`, and HEOML.
    *   Kept the ecosystem scope focused on health economics and outcomes research.
    *   Reserved the HEOML `voiage` extension boundary for VOI handoff and VOI result metadata.
    *   Defined optional adapter gates and compatibility-fixture expectations.
    *   Kept `mars` as a fixed-API optional metamodel backend.

*   [x] Define the `lifecourse` integration contract and compatibility fixture plan.
    *   Added the v1 artifact profile scaffold for consuming `lifecourse` PSA outputs.
    *   Aligned the profile with HEOML while preserving `voiage` VOI-specific schemas.
    *   Documented the optional adapter and dependency policy.
    *   Added a shared result-envelope schema overlay and illustrative fixture for EVPI, EVPPI, EVSI, and ENBS metadata.
    *   Documented portable interchange formats and excluded pickle from the shared contract.
    *   Seeded a deterministic local compatibility fixture with EVPI and EVPPI reference payloads.
*   [x] Add compatibility versioning and exchange-validation docs for the `lifecourse` integration contract.
    *   Recorded the supported `voiage`, `lifecourse`, and HEOML profile versions in the fixture manifest and illustrative result envelope.
    *   Added deterministic EVSI and ENBS expectation payloads to the normative `lifecourse` compatibility bundle.
    *   Documented the artifact-exchange validation order, optional adapter boundary, and fixture-check expectations in the integration docs.

*   [x] Add CLI `--verbose` debug logging that writes diagnostics to stderr without changing stdout output.
*   [x] Add example config generation for `voiage generate-config evsi > evsi_config.json`.
*   [x] Ensure all CLI `--help` output includes working examples.
*   [x] Add CLI end-to-end smoke tests covering the full command surface with `CliRunner`, including the experimental perspective command pair.

*   [x] Define polyglot tooling parity and observability plan.
    *   Documented the Python-first tooling stack, mapped the per-language CI/package gates, and captured the logging and versioning contract for the polyglot bindings.
    *   Added a developer guide page that records the tooling split, versioning contract, and logging policy.
*   [x] Implement dynamic-programming portfolio VOI optimization.
    *   Replaced the placeholder with memoized budget-constrained subset selection, optional dependency-group value discounting, and regression tests proving DP can outperform greedy selection.
*   [x] Implement Value of Heterogeneity.
    *   Added subgroup-specific decision value calculations, numeric subgroup binning, optimal subgroup identification, and subgroup plotting.
*   [x] Implement dominance analysis and plotting.
    *   Added strong dominance, extended dominance, frontier extraction, ICER helpers, and a cost-effectiveness plane plot.
*   [x] Implement Cost-Effectiveness Acceptability Frontier (CEAF).
    *   Added CEAF calculation, plotting, uncertainty bands, and package export coverage.
*   [x] Add efficient and moment-based EVSI methods plus an EVSI CLI command.
    *   Added `evsi(..., method="efficient")`, `evsi(..., method="moment_based")`, efficient metamodel selection, and `voiage calculate-evsi`.
*   [x] Default calibration VOI to the built-in modeler.
    *   Made `voi_calibration` usable without a custom modeler by defaulting to the existing built-in calibration modeler.
*   [x] Add a built-in observational VOI modeler.
    *   Added a default observational modeler for explicit net-benefit or cost/effect PSA samples with sample-size and bias-strength uncertainty adjustment.
*   [x] Replace the JAX two-loop EVSI placeholder.
    *   Implemented a JAX-assisted posterior update and resampling path and added a regression test proving the NumPy fallback is not used.
*   [x] Replace the sequential VOI step-level EVPI placeholder.
    *   Implemented the standard EVPI formula for explicit net-benefit samples and added regression tests for payoff extraction, monotonic learning behavior, and resolved-uncertainty cases.
*   [x] Enforce 90% branch-aware Python coverage.
    *   Enabled branch coverage in the active coverage configuration and added targeted tests across schema, backend, structural VOI, config, financial-risk, healthcare, memory-optimization, and network meta-analysis paths.
*   [x] Scaffold polyglot binding package CI and release publishing.
    *   Added TypeScript/npm, Go module, Rust/crates.io, Julia, .NET 11/NuGet, and R package validation paths.
*   [x] Add regression coverage for the TreeAge invalid XML fail-soft path and warning emission.
    *   Locked in the empty-dict fallback and `UserWarning` emission for malformed TreeAge XML imports.
*   [x] Add regression coverage for callable import resolution and schema round-trips.
    *   Covered `import_callable` builtin resolution and model round-trip serialization for `DecisionOption` and `TrialDesign`.
*   [x] Lock the backend and method module export contracts.
    *   Added regression coverage for the curated `__all__` surfaces in `voiage.backends` and `voiage.methods`.
*   [x] Expanded the core API contract validator test to cover the full versioned schema/example matrix.
    *   Added matrix coverage for all versioned core API schema/example pairs.
*   [x] Add regression coverage for EVPPI handling of raw dictionary parameter samples.
    *   Confirmed `evppi` accepts raw dict parameter samples in addition to `ParameterSet` inputs.
*   [x] Complete the Python cleanup phase for public imports, result payload shapes, and diagnostic behavior.
    *   Confirmed the curated export surface, stable reporting payloads, and explicit EVPPI compatibility warning path against the stable contract.
*   [x] Add deterministic provenance metadata validation for normative core API fixtures in the versioned manifest.
    *   Added provenance validation for normative manifest entries and hardened validator tests.
*   [x] Seed the normative and illustrative core API conformance fixtures for `ceac` under `specs/core-api/fixtures/v1/`.
    *   Added normative and illustrative CEAC payloads and registered them in the versioned manifest.
*   [x] Sync the core API v1 README indexes with the CEAC contract.
    *   Added the CEAC schema and example references to the versioned results and examples README indexes.
*   [x] Sync the core API v1 schema README index with the full contract matrix.
    *   Expanded the schema README from the entity-only list to the complete stable v1 entity and result contract matrix.
*   [x] Add a stable core API result schema and example for CEAC outputs.
    *   Added the CEAC result schema, versioned example payload, and validator coverage.
*   [x] Stabilize the curated package export surface for `voiage.core`, `voiage.methods`, and `voiage.plot`.
    *   Replaced placeholder subpackage entrypoints with explicit re-exports and added regression coverage to lock in the package-level import surface.
*   [x] Stabilize the top-level `voiage` package export surface.
    *   Added a root package facade that re-exports the primary submodules and locked it in with regression coverage.
*   [x] Seed the illustrative core API conformance fixture for `evsi` under `specs/core-api/fixtures/v1/illustrative/`.
    *   Added an EVSI illustrative example fixture and registered it in the versioned manifest.
*   [x] Seed the normative and illustrative core API conformance fixtures for `enbs` under `specs/core-api/fixtures/v1/`.
    *   Added ENBS normative and illustrative benchmark payloads and registered them in the versioned manifest.
*   [x] Seed the illustrative core API conformance fixture for `evppi` under `specs/core-api/fixtures/v1/illustrative/`.
    *   Added an EVPPI illustrative example fixture and registered it in the versioned manifest.
*   [x] Seed the first illustrative core API conformance fixture under `specs/core-api/fixtures/v1/illustrative/`.
    *   Added an EVPI illustrative example fixture and registered it in the versioned manifest.
*   [x] Seed the normative core API conformance fixture for `evsi` under `specs/core-api/fixtures/v1/normative/`.
*   [x] Seed the first normative core API conformance fixture under `specs/core-api/fixtures/v1/normative/`.
*   [x] Seed the normative core API conformance fixture for `evppi` under `specs/core-api/fixtures/v1/normative/`.
*   **[TEST]** Added validation coverage for the versioned core API fixture manifest scaffold.
    *   Added the initial `manifest.json` contract and regression tests for manifest versioning, artifact resolution, and missing-artifact failures.
*   **[SPEC]** Create the versioned fixture layout under `specs/core-api/fixtures/v1/`.
    *   Materialized the normative/illustrative fixture tree and README scaffold for the Phase 1 conformance-fixture track.
*   **[INFRA]** Raise enforced coverage to 90% with targeted regression tests for deterministic public modules.
    *   Added focused regression coverage for deterministic backend GPU helpers and advanced backend delegation paths.
*   **[TEST]** Add focused CLI regression coverage for NMA VOI validation and error branches.
    *   Locked in NMA CLI config-file handling, invalid JSON reporting, and unexpected-exception branches.
*   **[TEST]** Added deterministic regression coverage for plotting and ecosystem integration modules.
    *   Locked in plotting validation and optional-branch behavior together with ecosystem import/export edge cases.
*   **[TEST]** Added targeted regression coverage for deterministic runtime modules.
    *   Expanded backend, ecosystem-integration, health-economics, CLI, schema, exception, and fluent-API tests to lock in current behavior and raise measured suite coverage substantially.
*   **[TEST]** Added focused regression coverage for NICE HTA scoring and decision thresholds.
    *   Locked in NICE evaluation scoring for evidence quality, cost-effectiveness, budget impact, and the resulting approval/rejection decisions.
*   **[DOCS]** Create a validation notebook for EVPI and EVPPI.
    *   The notebook replicates the benchmark case documented in `docs/validation_comparison_report.md` and covers EVPI, EVPPI, EVSI, and plotting checks.
*   **[DOCS]** Create the remaining compact validation notebooks for NMA and structural VOI.
    *   Added runnable `examples/nma_validation.ipynb` and `examples/structural_voi_validation.ipynb` slices that execute the published-style NMA and structural VOI surfaces.
*   **[INFRA]** Created `AGENTS.md` to establish a protocol for AI agents.
*   **[INFRA]** Created `CONTRIBUTING.md` with technical development guidelines.
*   **[DOCS]** Updated `roadmap.md` to reflect the current project status.
*   **[INFRA]** Set up a `tox` configuration for automated testing and linting.
*   **[INFRA]** Implemented a pre-commit hook configuration for quality assurance.
*   **[INFRA]** Stabilized the `tox` test environment by fixing pytest marker configuration, restoring missing runtime/test dependencies, and resolving compatibility failures across schema, GPU, memory, clinical-trial, and metamodel code paths.
*   **[API]** Refactored the core logic into a `DecisionAnalysis` class.
*   **[API]** Established the initial domain-agnostic data schemas in `voiage/schema.py`.
*   **[DATA]** Finalized the transition to domain-agnostic data structures.
    *   Replaced internal direct dependency on legacy data-structure wrappers with `voiage.schema`.
    *   Standardized `DecisionAnalysis` and method signatures around `ParameterSet` and `ValueArray`.
*   **[EVSI]** Completed the EVSI implementation.
    *   Restored the `two_loop` path, added regression-based EVSI, and hardened the associated tests.
*   **[PLOT]** Implemented the core plotting functions.
    *   Added `voiage.plot.ceac` and `voiage.plot.voi_curves` for EVPI/EVSI visualization.
*   **[NMA]** Began the Network Meta-Analysis VOI implementation.
    *   Added the `voiage/methods/network_nma.py` workflow and supporting schema/tests.
