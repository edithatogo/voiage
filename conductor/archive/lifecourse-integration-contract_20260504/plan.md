# Track Implementation Plan: lifecourse Integration Contract

## Phase 1: Integration Boundary And Ownership [checkpoint: ]

- [x] Task: Document the project boundary.
  - [x] Record that `lifecourse` owns model execution, PSA generation, scenario configuration, provenance, validation targets, and reporting bundles.
  - [x] Record that `voiage` owns VOI algorithms, method settings, result schemas, diagnostics, and conformance fixtures.
  - [x] Document non-goals, including no default cross-dependency and no repo merge.
- [x] Task: Create the integration strategy document.
  - [x] Add `docs/integrations/lifecourse.md`.
  - [x] Link the strategy to existing core API schemas and fixture conventions.
  - [x] Link to `/Users/doughnut/GitHub/lifecourse/conductor/tracks/voiage_integration_20260428/`.
  - [x] Link to `/Users/doughnut/GitHub/lifecourse/docs/interchange/model_interchange_strategy.md`.
  - [x] Link to `/Users/doughnut/GitHub/lifecourse/heoml/` as the candidate shared artifact profile.

## Phase 2: Artifact Contract Scaffold [checkpoint: ]

- [x] Task: Define the v1 `lifecourse` VOI artifact profile.
  - [x] Specify required metadata for net benefits, strategies, parameter samples, WTP thresholds, population, time horizon, discount rate, and provenance.
  - [x] Specify preferred interchange formats: JSON metadata plus CSV, Arrow, or Parquet tables depending on artifact size and downstream tooling.
  - [x] Explicitly reject pickle as a portable interchange format.
  - [x] Specify deterministic fixture mode and tolerance policy references.
- [x] Task: Record external standard fit.
  - [x] Document why ONNX, PMML, and PFA are not primary formats for this integration.
  - [x] Document why PharmML and OMOP CDM are useful references but not direct replacements.
  - [x] Document where TreeAge-style feature mapping may be useful later.
- [x] Task: Add the versioned scaffold under `specs/integrations/lifecourse/v1/`.
  - [x] Include a README describing the profile.
  - [x] Reserve locations for examples, fixtures, and schema overlays.
  - [x] Align the scaffold with HEOML run-bundle conventions while retaining `voiage` VOI-specific schemas.

## Phase 3: Adapter And Dependency Policy [checkpoint: ]

- [x] Task: Define optional Python adapter behavior.
  - [x] Map `lifecourse` net-benefit artifacts to `voiage.ValueArray`.
  - [x] Map `lifecourse` parameter samples to `voiage.ParameterSet`.
  - [x] Require clear errors when optional dependencies are unavailable.
- [x] Task: Define dependency policy for integration consumers.
  - [x] Keep `lifecourse` out of `voiage` runtime dependencies.
  - [x] Recommend that `lifecourse` uses an optional `voiage` extra only after version and dependency policy are stable.
  - [x] Consider splitting heavy `voiage` dependencies into extras if the integration needs a lighter core install.
- [x] Task: Define stable result object needs.
  - [x] Ensure EVPI, EVPPI, EVSI, and ENBS results can carry method settings, scaling metadata, diagnostics, package version, and provenance.
  - [x] Keep result payloads compatible with existing core API result schemas.
  - [x] Include HEOML profile version and manifest identifiers in integration provenance.

## Phase 4: Shared Fixtures And Parity [checkpoint: ]

- [x] Task: Add a small `lifecourse`-style compatibility fixture.
  - [x] Include a net-benefit matrix, strategy names, parameter samples, and scaling metadata.
  - [x] Keep the fixture small enough for CI and future language bindings.
- [x] Task: Add conformance expectations.
  - [x] Define EVPI and EVPPI expected outputs first.
  - [x] Add EVSI and ENBS expectations once study-design fields are agreed.
  - [x] Document tolerances and method settings.

## Phase 5: Release And Documentation Coordination [checkpoint: ]

- [x] Task: Define compatibility versioning.
  - [x] Record supported `voiage` and `lifecourse` contract versions.
  - [x] Record the HEOML profile version that the fixture set targets.
  - [x] Require changelog notes for public contract changes.
- [x] Task: Add user and contributor documentation.
  - [x] Explain artifact exchange, optional adapter usage, limitations, and experimental method status.
  - [x] Document how downstream packages and bindings should validate against the shared fixtures.
  - [x] Link the shared profile and run-bundle docs back to `heoml/`.

## Execution Notes

- Prefer artifact compatibility before direct package integration.
- Keep the first adapter thin; do not import `lifecourse` from `voiage`.
- Avoid depending on experimental `lifecourse` internals.
- Coordinate changes with the corresponding `lifecourse` Conductor track: `voiage_integration_20260428`.
