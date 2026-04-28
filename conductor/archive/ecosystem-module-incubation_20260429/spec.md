# Ecosystem Module Incubation

## Overview

Define how `voiage` should participate in a broader ecosystem of modelling
projects without becoming a dependency sink or absorbing non-VOI concerns.
`voiage` should remain the Value of Information engine. It should integrate
with upstream model generators and downstream reporting or workflow tools
through portable artifacts, schemas, adapters, and conformance fixtures.

## Ecosystem Roles

- `voiage` owns EVPI, EVPPI, EVSI, ENBS, VOI study design, VOI diagnostics,
  VOI metamodel workflows, and VOI result contracts.
- `lifecourse` owns health-economic simulation, scenario execution, run
  bundles, and health-economic reporting artifacts.
- `innovate` owns intervention uptake, diffusion, policy adoption, and network
  spread models that can inform implementation scenarios.
- `mars` owns a fixed-API surrogate/metamodel package. `voiage` may use it as
  an optional metamodel backend, but must not require changes to its core API.
- HEOML owns portable health-economic run-bundle and extension artifacts.

## Goals

- Define `voiage` as a stable VOI consumer and producer in the ecosystem.
- Keep all ecosystem integrations optional and artifact-first.
- Reserve a HEOML `voiage` extension namespace for VOI handoff and VOI result
  metadata.
- Define how `voiage` can use `mars` as a metamodel backend without changing
  the `mars` API.
- Define how `innovate` outputs could become inputs to VOI workflows when
  diffusion/adoption uncertainty affects decision value.
- Align with the `lifecourse` ecosystem-module incubation plan.

## Functional Requirements

- Document the ecosystem boundary for `voiage`, `lifecourse`, `innovate`,
  `mars`, HEOML, and future sibling modules.
- Define which artifacts `voiage` should consume and produce:
  net-benefit matrices, parameter samples, strategy names, WTP thresholds,
  trial designs, EVPI/EVPPI/EVSI/ENBS results, diagnostics, and method settings.
- Define optional adapter policy for `lifecourse`, `innovate`, and `mars`.
- Add a compatibility-fixture strategy for cross-repo validation.
- Define dependency gates for optional extras, including CI smoke checks,
  Renovate coverage, security checks, documentation, and deprecation paths.

## Non-Functional Requirements

- No `lifecourse`, `innovate`, or `mars` dependency should enter the base
  `voiage` install through this planning work.
- `voiage` must not import sibling project internals for stable integrations.
- Pickle must remain outside the portable interchange contract.
- Cross-repo integrations must be versioned and fixture-tested before being
  described as supported.

## Acceptance Criteria

- `docs/ecosystem/module_incubation_strategy.md` documents the `voiage` role in
  the ecosystem, optional integration policy, candidate sibling modules, and
  promotion criteria.
- `specs/ecosystem/README.md` defines the first ecosystem contract outline and
  HEOML extension alignment for `voiage`.
- `conductor/tracks.md`, `roadmap.md`, `todo.md`, and `changelog.md` reference
  the new ecosystem-module incubation work.

## Out Of Scope

- Implementing adapters in this track.
- Adding runtime dependencies on `lifecourse`, `innovate`, or `mars`.
- Creating new external repositories.
- Modifying the `mars` core API.
- Replacing existing `voiage` core API contracts.
