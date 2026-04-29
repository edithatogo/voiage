# `voiage` Task List

This document lists the actionable tasks for `voiage` development. Agents should pick tasks from the "To Do" list.

## To Do

*   [ ] Continue the SOTA VOI frontier track after the experimental Value of Perspective slice.
    *   [x] Add experimental Value of Perspective contracts, Python API, `DecisionAnalysis` wrapper, CLI command, plot helper, and tests.
    *   [ ] Promote Value of Perspective from experimental to fixture-backed once deterministic conformance fixtures and cross-language expectations are complete.
    *   [x] Add deterministic screening-program fixtures for Value of Perspective.
    *   [x] Add deterministic fixture sets for distributional/equity and implementation-adjusted VOI.
    *   [x] Add repository-level validation for frontier fixture manifests.
    *   [x] Add a frontier fixture registry manifest for discovery of the committed experimental contract set.
    *   [x] Add a registry schema for the frontier fixture discovery layer.
    *   [x] Add a reusable frontier contract validation script.
    *   [x] Wire the frontier contract validator into tox and CI.
    *   [x] Update the README and frontier research note to mention the registry-backed contract layer.
    *   [x] Update the migration guide to reflect the registry-backed frontier contracts.
    *   [x] Add frontier examples to the canonical advanced VOI guide.
    *   [x] Add explicit experimental API warnings and release-note style guidance.
    *   Plan distributional/equity VOI, implementation-adjusted VOI, preference-information VOI, validation VOI, threshold/tipping-point VOI, robust VOI, and dynamic real-options VOI.
    *   [x] Define preference heterogeneity and value of individualized care contracts.
    *   [x] Define model-validation VOI contracts.
    *   [x] Define threshold, tipping-point, and robust VOI contracts.
    *   Triage causal/transportability VOI, data-quality and privacy VOI, computational/model-refinement VOI, expert-elicitation VOI, and evidence-synthesis design VOI as adjacent frontier extensions.
    *   Add CHEERS-VOI reporting metadata, structured result fields, and reproducibility outputs for every frontier method family.
    *   Add schemas, deterministic fixtures, docs, CLI coverage, and maturity metadata before marking any frontier method stable.
    *   [x] Add first deterministic fixtures for equity and implementation-adjusted VOI.
    *   [x] Add CHEERS-VOI reporting objects to the experimental frontier result payloads.
    *   [x] Add CHEERS-VOI reporting objects to Value of Heterogeneity as the base distributional surface.
    *   [x] Extend the CHEERS-VOI reporting baseline to CEAF and dominance outputs.
    *   [x] Extend the CHEERS-VOI reporting baseline to core scalar CLI outputs (EVPI, EVPPI, EVSI, ENBS).

*   [ ] Define the HEOR module naming brainstorm for `calibrate`, `evidence`, `process`, `report`, `registry`, `workflow`, `quality`, `engines`, `heoml`, and PM4Py as ecosystem-only.
    *   Keep the list as a naming and boundary exercise, not an implementation plan.
    *   Require CLI support and an explicit MCP decision for any future module.
    *   Keep PM4Py in the process-mining ecosystem-only bucket.

## In Progress

*None at the moment.*

## Done

*   [x] Add Vale prose linting for Markdown and reStructuredText documentation.
    *   Added a repo-local Vale config and style set, hooked the CI prose job to the real docs paths, and documented the command in `CONTRIBUTING.md`.

*   [x] Add CHEERS-VOI reporting objects and contract examples for the experimental frontier methods.
    *   Added shared reporting payload helpers, attached them to Value of Perspective, distributional/equity VOI, and implementation-adjusted VOI, and mirrored the fields in the frontier contract examples and schemas.

*   [x] Add CHEERS-VOI reporting objects to Value of Heterogeneity.
    *   Added the same reporting payload helper to the base distributional surface so the reporting model extends naturally from Value of Heterogeneity into distributional/equity VOI.

*   [x] Extend the CHEERS-VOI reporting baseline to CEAF and dominance.
    *   Added shared reporting payloads to the frontier summary outputs that feed the main cost-effectiveness analysis workflow.

*   [x] Extend the CHEERS-VOI reporting baseline to core scalar CLI outputs.
    *   Added shared reporting payloads to EVPI, EVPPI, EVSI, and ENBS JSON/CSV result output so the main command surface has a consistent reporting envelope.

*   [x] Add deterministic screening-program fixtures for Value of Perspective.
    *   Added a normative fixture manifest plus input/output payloads that anchor the experimental CLI contract for the screening-program comparison surface.

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
    *   Documented portable interchange formats and excluded pickle from the shared contract.
    *   Seeded a deterministic local compatibility fixture with EVPI and EVPPI reference payloads.

*   [x] Add CLI `--verbose` debug logging that writes diagnostics to stderr without changing stdout output.
*   [x] Add example config generation for `voiage generate-config evsi > evsi_config.json`.
*   [x] Ensure all CLI `--help` output includes working examples.
*   [x] Add CLI end-to-end smoke tests covering the full command surface with `CliRunner`.

*   [x] Define polyglot tooling parity and observability plan.
    *   Documented the Python-only tooling stack, mapped the per-language CI/package gates, and captured the logging and versioning contract for the polyglot bindings.
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
