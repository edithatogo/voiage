# Specification: Standardized Dataset Ingestion

## Overview

Extend VOIAGE with a two-stage data architecture:

1. a versioned, format-neutral `NormalizedInputBundle` inside the runtime
   contract and calculation-conductor boundary; and
2. optional ingestion providers that translate Croissant ML and Frictionless
   Data Packages into that bundle.

The repository's `conductor/` directory remains the programme-governance
surface. Runtime contracts belong under `voiage/contracts/`; format-specific
providers belong under `voiage/ingestion/`.

```text
croissant.json   -> Croissant provider   --\
                                             -> NormalizedInputBundle
datapackage.json -> Frictionless provider --/          |
                                                         v
                                                VOI input preparation
                                                         |
                                                         v
                                              calculation kernels/results
```

## Objectives

- Make modern standardized datasets first-class VOIAGE inputs.
- Preserve one numerical implementation regardless of source format.
- Keep optional parsers out of the stable core dependency and import graph.
- Preserve schema, integrity, transformation, and source provenance.
- Require explicit VOI semantic bindings instead of silently guessing meaning.
- Maintain backward compatibility with existing NumPy, xarray, CSV,
  `ValueArray`, `ParameterSet`, and `AnalysisSpec` entry points.

## Requirements

### R1 — Canonical normalized input contract

The runtime MUST expose a versioned `NormalizedInputBundle` containing:

- a strict immutable dataset manifest;
- named `pyarrow.Table` objects;
- table, field, key, and relationship descriptions;
- explicit VOI semantic bindings;
- source, parser, integrity, and transformation provenance; and
- stable validation diagnostics.

The contract MUST support sample, strategy, net-benefit, cost, outcome,
parameter, design, target, weight, perspective, and split roles. Scientific
roles MUST NOT be inferred solely from column names.

### R2 — Deterministic identity and interchange

The normalized contract MUST provide deterministic canonical JSON, logical
Arrow schema fingerprints, content digests, and deterministic IPC/Parquet
round trips. Credentials, authorization headers, signed URL query strings, and
other secrets MUST NOT enter provenance or diagnostics.

### R3 — Existing VOI runtime preparation

A format-neutral preparation layer MUST convert normalized bundles into
`AnalysisSpec`, `ValueArray`, and optional `ParameterSet` instances. It MUST
validate keys, joins, sample alignment, strategies, missing values, dtypes,
cardinality, and method-specific input capabilities before dispatch.

Existing direct Python and CSV APIs MUST continue to work unchanged.

### R4 — Optional provider protocol

An `IngestionProvider` protocol and registry MUST support explicit provider
selection, conservative probing, validation, loading, source-access policy, and
stable error categories.

Core contracts and calculation kernels MUST NOT import `mlcroissant`,
`frictionless`, or format-specific models. Base `import voiage` MUST work
without either optional dependency.

### R5 — Croissant ML provider

An optional Croissant provider MUST validate version declarations, JSON-LD
identities and references, resources, `RecordSet` fields, keys, relationships,
splits, integrity data, and supported transformations. It MUST publish its
supported Croissant version/feature profile and reject unsupported constructs
explicitly.

ML labels, model candidates, and dataset splits MUST NOT be silently treated as
VOI outcomes, decision strategies, or study designs.

### R6 — Frictionless Data provider

An optional Frictionless provider MUST validate package descriptors and data,
preserve package/resource metadata, Table Schema types and constraints, CSV
dialects, primary/composite/foreign keys, integrity information, and supported
resource formats. It MUST publish its supported profile and reject unsupported
or ambiguous resources explicitly.

### R7 — Cross-format conformance

One canonical decision fixture MUST be represented as Croissant, Frictionless,
Arrow IPC, Parquet, and direct NumPy/xarray inputs. These representations MUST
produce equivalent normalized identities, runtime objects, and EVPI, EVPPI,
EVSI, and applicable downstream results within declared tolerances.

### R8 — Product surface

The Python and CLI surfaces MUST support separate validation, inspection,
normalization, and calculation operations. Ambiguous inputs MUST require a
binding file or embedded namespaced VOIAGE binding profile.

Installation extras MUST remain separate:

- `voiage[croissant]`
- `voiage[frictionless]`
- `voiage[ingestion]`

### R9 — Security and resource governance

Untrusted descriptors and resources MUST be constrained by an explicit
`SourceAccessPolicy`, including offline behavior, schemes, hosts, redirects,
timeouts, checksums, byte/row limits, and archive limits.

The implementation MUST block traversal, unsafe archive members, local-device
files, implicit authenticated access, SSRF by default, descriptor-supplied
code execution, and arbitrary custom transforms.

### R10 — Quality and release evidence

The programme MUST retain the repository's TDD, greater-than-90-percent Python
coverage, typing, Ruff, dependency-frontier, Arrow round-trip,
numerical-equivalence, CPU-fallback, clean-install, SBOM/license, harness, tox,
documentation, security, mutation, performance, and hosted-check gates.

## Acceptance criteria

- **AC-01:** The normalized contract is versioned, deterministic, immutable,
  Arrow-compatible, and format-neutral.
- **AC-02:** Existing VOI inputs and normalized inputs produce equivalent
  runtime objects and results.
- **AC-03:** The provider protocol works in a base installation with neither
  optional parser installed.
- **AC-04:** The Croissant provider passes its supported-profile fixtures and
  fails explicitly for unsupported or unsafe inputs.
- **AC-05:** The Frictionless provider passes its supported-profile fixtures
  and fails explicitly for unsupported or unsafe inputs.
- **AC-06:** The cross-format fixture matrix satisfies schema, provenance,
  digest, and numerical-equivalence assertions.
- **AC-07:** CLI/API users can validate and inspect before normalization or
  calculation, and diagnostics do not leak secrets.
- **AC-08:** Dependency, security, performance, compatibility, packaging,
  documentation, and hosted release gates pass.
- **AC-09:** Conductor, GitHub issues, native sub-issue relationships, and
  Project 28 remain reconciled.

## Non-functional constraints

- Python 3.12–3.14 remain supported.
- Arrow/Parquet remain the public tabular interchange; xarray remains the
  established compute-facing labeled-array representation.
- Polars remains an adapter/consumer rather than the canonical compute model.
- New code is fully typed and uses NumPy-style public docstrings.
- Fixture generation is deterministic, offline by default, and reviewable.
- Large inputs are streamed or batched where practical.
- Format-specific metadata may enrich provenance but may not change numerical
  meaning without an explicit binding or transformation.

## External gates

- Authoritative live interoperability probes depend on the continued
  availability of third-party public examples and are non-authoritative for
  local correctness.
- Upstream parser defects or unsupported standard features remain external and
  MUST be recorded without weakening the supported profile.
- No external dataset publication, registry submission, or authenticated
  service action is authorized by this track.

## Out of scope

- Replacing Arrow/Parquet or xarray as established repository contracts.
- Moving runtime code into the `conductor/` governance directory.
- Reimplementing Croissant or Frictionless specifications from scratch when a
  suitable optional library provides verified behavior.
- Automatic scientific-semantic inference from names or ML task metadata.
- Executing arbitrary descriptor-defined code or plugins.
- Supporting every media, nested, transformation, archive, and remote-storage
  feature in the first provider profile.
- Changing the numerical definition of any VOI method.

## GitHub hierarchy

Parent:
[#325](https://github.com/edithatogo/voiage/issues/325)

- [#326](https://github.com/edithatogo/voiage/issues/326) — normalized core
  contract
- [#327](https://github.com/edithatogo/voiage/issues/327) — VOI input
  preparation
- [#328](https://github.com/edithatogo/voiage/issues/328) — provider protocol
  and registry
- [#329](https://github.com/edithatogo/voiage/issues/329) — Croissant provider
- [#330](https://github.com/edithatogo/voiage/issues/330) — Frictionless
  provider
- [#331](https://github.com/edithatogo/voiage/issues/331) — cross-format
  conformance
- [#332](https://github.com/edithatogo/voiage/issues/332) — CLI, packaging,
  diagnostics, and documentation
- [#333](https://github.com/edithatogo/voiage/issues/333) — security,
  performance, compatibility, and release gates

All eight children are native GitHub sub-issues of #325 and are items in
[Project 28](https://github.com/users/edithatogo/projects/28).

## Authoritative inputs

- Repository protocol: `AGENTS.md` at
  `f86957f6acb08284619523f75a326795650af38f`.
- Runtime contracts: `voiage/contracts/analysis.py`,
  `voiage/contracts/adapters.py`, `voiage/contracts/interchange.py`, and
  `voiage/contracts/kernel.py` at the same revision.
- Existing compute-facing inputs: `voiage/schema.py` and `voiage/core/io.py`.
- Core API rules: `specs/core-api/foundation.md` and
  `specs/core-api/extension-evolution.md`.
- Dependency/interchange policy:
  `docs/astro-site/src/content/docs/developer-guide/quality-and-security.mdx`.
- Croissant 1.1:
  <https://docs.mlcommons.org/croissant/docs/croissant-spec-1.1.html> and
  conformance URI `http://mlcommons.org/croissant/1.1`.
- Frictionless Data Package v1:
  <https://specs.frictionlessdata.io/data-package/>.
- Frictionless Table Schema v1:
  <https://specs.frictionlessdata.io/table-schema/>.
- User-approved issue architecture captured in GitHub issue #325 and its
  native sub-issues #326–#333 on 2026-07-23.

## Approval

The user approved incorporation of the previously presented architecture and
issue plan into the repository's Conductor system, GitHub issues/sub-issues,
and project on 2026-07-23. This approval initializes planning only; it does not
authorize publication, external submission, or relaxation of repository gates.
