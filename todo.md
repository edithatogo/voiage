# `voiage` Task List

This document lists the actionable tasks for `voiage` development. Agents should pick tasks from the "To Do" list.

## To Do

## In Progress

*None at the moment.*

## Done

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
