# Track Specification: lifecourse Integration Contract

## Overview

This track defines how `voiage` should interoperate with the separate
`edithatogo/lifecourse` microsimulation project. `lifecourse` generates
health-economic simulation outputs and reproducible run bundles. `voiage`
provides Value of Information methods and should consume `lifecourse` outputs
through stable artifacts rather than through `lifecourse` internals.

## Functional Requirements

1. Define a language-neutral artifact contract for `lifecourse` PSA outputs that
   `voiage` can consume.
2. The contract must cover net-benefit arrays, strategy names, parameter
   samples, willingness-to-pay thresholds, population scaling, time horizon,
   discount rate, method settings, run provenance, and package versions.
3. The contract must define portable interchange formats and explicitly exclude
   pickle from compatibility guarantees.
4. Map the artifact contract to existing `voiage` concepts: `ValueArray`,
   `ParameterSet`, `TrialDesign`, and EVPI, EVPPI, EVSI, and ENBS result
   contracts.
5. Define which project owns each responsibility.
6. Specify optional adapter behavior for consumers that want to call `voiage`
   directly from `lifecourse`.
7. Identify upstream `voiage` changes needed to make the integration safe,
   including dependency extras and stable result schemas.
8. Link to the corresponding `lifecourse` documentation and Conductor track.

## Non-Functional Requirements

1. `voiage` must not require `lifecourse` as a runtime dependency.
2. `lifecourse` should not need `voiage` for base simulation execution.
3. Integration should be possible through artifact exchange, optional Python
   imports, or CLI/subprocess workflows.
4. The public contract must avoid Python-only assumptions so future bindings can
   consume the same fixtures.
5. Experimental `voiage` methods must be labelled experimental when surfaced
   through `lifecourse`.
6. Existing standards such as ONNX, PMML, PharmML, OMOP CDM, and TreeAge-style
   exports should be considered as references or adapters, not as the primary
   contract unless later evidence justifies it.

## Acceptance Criteria

1. A strategy document exists under `docs/integrations/`.
2. A versioned integration contract scaffold exists under
   `specs/integrations/lifecourse/v1/`.
3. The track plan identifies implementation work for artifacts, adapters,
   fixtures, dependency policy, and release compatibility.
4. `conductor/tracks.md`, `roadmap.md`, `todo.md`, and `changelog.md` reference
   the integration work.
5. The docs point readers to the corresponding `lifecourse` track and
   interchange strategy.
6. The docs identify HEOML as the candidate shared artifact profile while
   preserving `voiage` ownership of VOI-specific schemas and result contracts.

## Out of Scope

1. Implementing the `lifecourse` adapter in this planning increment.
2. Making either project depend on the other by default.
3. Replacing `lifecourse` internal VOI helpers before parity is proven.
4. Merging the repositories.
