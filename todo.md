# `voiage` Task List

This document lists the actionable tasks for `voiage` development. Agents should pick tasks from the "To Do" list.

## To Do

*   **[INFRA]** Raise enforced coverage to 90% with targeted regression tests for deterministic public modules.
    *   Continue adding focused tests for the remaining low-coverage deterministic code paths until the full `tox` suite passes under the stricter threshold.

## In Progress

*None at the moment.*

## Done

*   **[TEST]** Added deterministic regression coverage for plotting and ecosystem integration modules.
    *   Locked in plotting validation and optional-branch behavior together with ecosystem import/export edge cases.
*   **[TEST]** Added targeted regression coverage for deterministic runtime modules.
    *   Expanded backend, ecosystem-integration, health-economics, CLI, schema, exception, and fluent-API tests to lock in current behavior and raise measured suite coverage substantially.
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
