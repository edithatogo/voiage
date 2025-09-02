# `voiage` Task List

This document lists the actionable tasks for `voiage` development. Agents should pick tasks from the "To Do" list.

## To Do

-   **[DATA]** Finalize the transition to domain-agnostic data structures.
    -   Remove `voiage.core.data_structures` and replace all internal usages with `voiage.schema`.
    -   Update `DecisionAnalysis` and method signatures to use `ParameterSet` and `ValueArray` directly.
-   **[EVSI]** Complete the EVSI implementation.
    -   Un-comment the tests in `tests/test_sample_information.py`.
    -   Fix the `two_loop` implementation to pass the tests.
    -   Implement the regression-based method for EVSI.
-   **[PLOT]** Implement the core plotting functions.
    -   Create `voiage.plot.ceac` for Cost-Effectiveness Acceptability Curves.
    -   Create `voiage.plot.voi_curves` for plotting EVPI and EVSI against willingness-to-pay.
-   **[DOCS]** Create a validation notebook for EVPI and EVPPI.
    -   The notebook should replicate results from a published study or another VOI package.
-   **[NMA]** Begin implementation of Network Meta-Analysis VOI (`evsi_nma`).
    -   Define the required data structures in `voiage.schema`.
    -   Create the file `voiage/methods/network_nma.py` with a placeholder function.

## In Progress

*None at the moment.*

## Done

*   **[INFRA]** Created `AGENTS.md` to establish a protocol for AI agents.
*   **[INFRA]** Created `CONTRIBUTING.md` with technical development guidelines.
*   **[DOCS]** Updated `roadmap.md` to reflect the current project status.
*   **[INFRA]** Set up a `tox` configuration for automated testing and linting.
*   **[INFRA]** Implemented a pre-commit hook configuration for quality assurance.
*   **[API]** Refactored the core logic into a `DecisionAnalysis` class.
*   **[API]** Established the initial domain-agnostic data schemas in `voiage/schema.py`.
