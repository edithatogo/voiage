# `voiage` Task List (Updated)

This document lists the actionable tasks for `voiage` development. Agents should pick tasks from the "To Do" list.

## In Progress

-   **[NMA]** Enhance implementation of Network Meta-Analysis VOI (`evsi_nma`).
    -   Add more sophisticated NMA models.
    -   Implement additional NMA-specific functionality.
-   **[PLOT]** Enhancing plotting capabilities.
    -   Expanding VOI curve plotting functionality.
    -   Improving documentation and examples.

## Done

-   **[DOCS]** Expand documentation and examples.
    -   Create comprehensive API documentation.
    -   Add cross-domain examples and tutorials.
    -   Expand user guides and best practices.
-   **[ADVANCED]** Complete implementation of advanced methods.
    -   Finish implementing methods in `voiage/methods/adaptive.py`.
    -   Finish implementing methods in `voiage/methods/calibration.py`.
    -   Finish implementing methods in `voiage/methods/observational.py`.
-   **[EVSI]** Complete the EVSI implementation.
    -   Activate and fix the tests in `tests/test_sample_information.py`.
    -   Implement the regression-based method for EVSI.
    -   Improve the `two_loop` implementation to pass the tests.
-   **[DATA]** Finalize the transition to domain-agnostic data structures.
    -   Remove `voiage.core.data_structures` and replace all internal usages with `voiage.schema`.
    -   Update `DecisionAnalysis` and method signatures to use `ParameterSet` and `ValueArray` directly.
-   **[PLOT]** Complete the core plotting functions.
    -   Finish implementing `voiage.plot.ceac` for Cost-Effectiveness Acceptability Curves.
    -   Add additional plotting options to `voiage.plot.voi_curves`.
-   **[DOCS]** Create validation notebooks for core methods.
    -   Create notebooks that replicate results from published studies or established R packages.
    -   Validate EVPI, EVPPI, and EVSI implementations.
-   **[NMA]** Begin implementation of Network Meta-Analysis VOI (`evsi_nma`).
    -   Define the required data structures in `voiage.schema`.
    -   Create the implementation in `voiage/methods/network_nma.py`.
-   **[STRUCT]** Complete implementation of Structural VOI methods.
    -   Finish implementing `structural_voi` in `voiage/methods/structural.py`.
-   **[SEQ]** Complete implementation of Sequential VOI methods.
    -   Finish implementing `sequential_voi` in `voiage/methods/sequential.py`.
-   **[INFRA]** Created `AGENTS.md` to establish a protocol for AI agents.
-   **[INFRA]** Created `CONTRIBUTING.md` with technical development guidelines.
-   **[DOCS]** Updated `roadmap.md` to reflect the current project status.
-   **[INFRA]** Set up a `tox` configuration for automated testing and linting.
-   **[INFRA]** Implemented a pre-commit hook configuration for quality assurance.
-   **[API]** Refactored the core logic into a `DecisionAnalysis` class.
-   **[API]** Established the initial domain-agnostic data schemas in `voiage/schema.py`.
-   **[PORT]** Implemented portfolio optimization with multiple algorithms.
-   **[XARRAY]** Completed integration of xarray throughout the library.
-   **[BACKEND]** Started implementation of JAX backend support.

## Future Considerations

-   **[ML]** Implement machine learning-based metamodels.
-   **[REALTIME]** Add support for real-time VOI calculations.
-   **[API]** Establish language-agnostic API specification.
-   **[PORTS]** Begin planning for R and Julia ports.