# lifecourse VOI Artifact Profile v1

This directory reserves the versioned integration profile for consuming
`lifecourse` outputs in `voiage`.

This profile should align with HEOML, currently embedded at
`/Users/doughnut/GitHub/lifecourse/heoml/`, while keeping `voiage`'s core API
schemas as the VOI-specific result contracts.
Shared provenance fields should include the HEOML profile version and a
manifest identifier so consumers can check the portable contract before
loading artifacts.
The public `voiage.load_heoml_run_bundle()` helper reads this scaffold
directly into the core `ValueArray` and `ParameterSet` types and returns a
`HeomlRunBundle` wrapper with provenance.

## Purpose

The profile defines the minimum artifact shape needed for `voiage` to calculate
Value of Information results from `lifecourse` PSA outputs without depending on
`lifecourse` internals.

## Required Artifact Groups

- metadata: schema version, producer, package versions, run identifier,
  provenance, and diagnostics
- net benefits: PSA samples by strategy
- strategies: stable strategy names and optional intervention metadata
- parameters: PSA parameter samples aligned to net-benefit samples
- scaling: population, time horizon, discount rate, and willingness-to-pay
  thresholds
- method settings: VOI method name, approximation method, seeds, and tolerances

## Mapping To Existing Core API Contracts

- net benefits should map to `specs/core-api/schemas/v1/value-array.schema.json`
- parameter samples should map to
  `specs/core-api/schemas/v1/parameter-set.schema.json`
- EVPI, EVPPI, EVSI, and ENBS outputs should map to the result schemas under
  `specs/core-api/schemas/v1/results/`
- HEOML run-bundle manifests should map artifact references into the VOI-specific
  schemas above.

## Reserved Layout

- `examples/`: small human-reviewable example payloads
- `fixtures/`: deterministic compatibility fixtures shared with `lifecourse`
- `schemas/`: integration-specific schema overlays if the core API schemas need
  producer metadata or bundle metadata

## Local Scaffold

- [examples/README.md](./examples/README.md)
- [fixtures/README.md](./fixtures/README.md)
- [fixtures/manifest.json](./fixtures/manifest.json)
- [schemas/README.md](./schemas/README.md)

## Non-Portable Formats

Pickle files are outside this profile. They are Python-specific, unsafe for
untrusted interchange, and unsuitable for R, Julia, Go, Rust, TypeScript, or
.NET consumers.

## External Standards

This profile may reference external standards, but it does not adopt one as the
primary interchange format:

- ONNX, PMML, and PFA are ML-oriented interchange formats.
- PharmML is pharmacometrics-oriented.
- OMOP CDM is observational data-oriented.
- TreeAge feature mapping may be useful later as an ecosystem reference.

## Initial Status

This is a scaffold. The first implementation phase should add a small
`lifecourse`-style fixture with net benefits, parameter samples, strategy names,
population scaling metadata, and expected EVPI/EVPPI outputs.
