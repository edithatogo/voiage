# Implementation Plan: Standardized Dataset Ingestion

`plan.md` is the task source of truth. Tests precede implementation in every
functional phase. Each phase ends with automated review, focused validation,
and a Conductor checkpoint under `conductor/workflow.md`.

## Phase 1 — Freeze the normalized input contract (#326)

- [ ] **P1-T1 / AC-01:** Write failing contract tests for strict validation,
  immutability, explicit VOI bindings, canonical JSON, schema/content
  fingerprints, secret redaction, and Arrow IPC/Parquet round trips.
- [ ] **P1-T2 / AC-01:** Add the versioned dataset, resource, table, field,
  relationship, provenance, diagnostic, and binding models under
  `voiage/contracts/`.
- [ ] **P1-T3 / AC-01:** Add
  `specs/core-api/schemas/v2/normalized-input-bundle.schema.json`, compatibility
  rules, exports, and deterministic golden fixtures.
- [ ] **P1-T4 / AC-10:** Define the independently versioned VOI binding profile,
  its JSON Schema, canonical serialization, digest, precedence rules, units,
  perspectives, transformations, and method-applicability validation.
- [ ] **P1-T5 / AC-01, AC-10:** Add unsupported-version, conflicting-binding,
  stale-reference, unit-incompatibility, and namespaced-extension tests.
- [ ] **P1-T6 / AC-01:** Verify that the core contract imports no external
  ingestion dependency.
- [ ] **P1-T7 / AC-01, AC-10:** Run automated review, focused tests, schema validation,
  Arrow/Polars fresh-process round trips, and the phase checkpoint protocol.

## Phase 2 — Prepare existing VOI runtime inputs (#327)

- [ ] **P2-T1 / AC-02:** Write failing tests for wide and long net-benefit
  tables, cost/outcome derivation, perspectives, parameter preparation, sample
  alignment, joins, strategies, nulls, dtypes, and cardinality.
- [ ] **P2-T2 / AC-02:** Implement `PreparedAnalysisInputs` and the
  format-neutral preparation layer.
- [ ] **P2-T3 / AC-02, AC-11:** Emit a machine-readable data-quality report for
  row counts, missingness, uniqueness, keys, join coverage, coercions,
  exclusions, and selected records or partitions.
- [ ] **P2-T4 / AC-02:** Propagate artifact IDs and normalized input digests
  through `AnalysisSpec`, `RunContext`, diagnostics, and provenance.
- [ ] **P2-T5 / AC-02:** Add the method/input capability matrix and preserve
  existing direct Python and CSV entry points.
- [ ] **P2-T6 / AC-02:** Verify that filtering, projection, sampling,
  aggregation, and exclusion cannot occur implicitly.
- [ ] **P2-T7 / AC-02, AC-11:** Run direct-versus-normalized numerical-equivalence
  tests, automated review, validation, and the phase checkpoint protocol.

## Phase 3 — Establish the optional provider boundary (#328)

- [ ] **P3-T1 / AC-03:** Write failing fake-provider, missing-extra,
  import-isolation, error-taxonomy, source-policy, and resolver-injection tests.
- [ ] **P3-T2 / AC-03:** Implement the `IngestionProvider` protocol, provider
  registry, stable errors, and conservative provider probing.
- [ ] **P3-T3 / AC-03:** Implement `SourceAccessPolicy` and the deterministic,
  dependency-injected resource-resolution boundary.
- [ ] **P3-T4 / AC-03:** Add provider capability declarations for versions,
  media types, transforms, projection, filtering, streaming, and random access.
- [ ] **P3-T5 / AC-03:** Add opt-in, allow-listed Python entry-point discovery;
  prohibit automatic third-party imports during package import and probing.
- [ ] **P3-T6 / AC-03:** Document the third-party provider extension contract.
- [ ] **P3-T7 / AC-03:** Verify clean base installation, run automated review,
  focused validation, and the phase checkpoint protocol.

## Phase 4 — Implement the two source-format providers (#329, #330)

### Croissant ML

- [ ] **P4-T1 / AC-04:** Run `uv lock --upgrade` and
  `python scripts/dependency_frontier.py . --strict`; review and record the
  Croissant parser dependency and supported-profile decision before changing
  dependencies.
- [ ] **P4-T2 / AC-04:** Write failing offline fixtures/tests for versioning,
  identities, resources, record sets, fields, keys, references, splits,
  supported transformations, integrity failures, archives, nesting, and
  ambiguous semantics.
- [ ] **P4-T3 / AC-04, AC-11:** Add fixtures for Croissant 1.1 conformance,
  parser-feature gaps, live datasets, citations, PROV, usage information,
  ODRL, and RAI metadata preservation.
- [ ] **P4-T4 / AC-04:** Implement the lazy optional Croissant provider and
  publish separate standard-conformance and parser-capability profiles.
- [ ] **P4-T5 / AC-04, AC-11:** Add Croissant inspection, diagnostics,
  provenance, governance metadata, and one opt-in authoritative live
  interoperability probe.

### Frictionless Data

- [ ] **P4-T6 / AC-05:** Review and record the Frictionless dependency and
  supported-profile decision through the same dependency-frontier evidence.
- [ ] **P4-T7 / AC-05:** Write failing offline fixtures/tests for package and
  data validation, resources, schemas, dialects, types, constraints, missing
  values, keys, integrity, governance metadata, supported tabular formats, and
  ambiguous resources.
- [ ] **P4-T8 / AC-05:** Implement the lazy optional Frictionless provider and
  documented supported profile.
- [ ] **P4-T9 / AC-05, AC-11:** Add Frictionless inspection, diagnostics,
  provenance, licence/citation/usage preservation, and one opt-in authoritative
  live interoperability probe.

### Phase checkpoint

- [ ] **P4-T10 / AC-03–AC-05, AC-11:** Verify base-import isolation and clean installs
  for each extra; run automated review, focused tests, dependency/security
  audits, and the phase checkpoint protocol.

## Phase 5 — Prove cross-format conformance (#331)

- [ ] **P5-T1 / AC-06:** Define the canonical decision fixture and failing
  parity assertions across Croissant, Frictionless, Arrow IPC, Parquet, and
  direct NumPy/xarray representations.
- [ ] **P5-T2 / AC-06:** Add the deterministic fixture manifest with pinned
  descriptor, resource, schema, and content digests.
- [ ] **P5-T3 / AC-06:** Implement deterministic fixture generation and the
  schema, provenance, ordering, meaningful-change, and numerical-equivalence
  conformance matrix.
- [ ] **P5-T4 / AC-06, AC-10, AC-11:** Assert binding-profile, data-quality,
  governance-metadata, and materialization-receipt parity without requiring
  source formats to share irrelevant descriptive metadata.
- [ ] **P5-T5 / AC-06:** Add malformed/adversarial cases, property-based mapping
  tests, parser-differential checks, and fresh-process PyArrow/Polars checks.
- [ ] **P5-T6 / AC-06:** Add the conformance matrix to tox and hosted CI; run
  automated review, validation, and the phase checkpoint protocol.

## Phase 6 — Ship the user-facing product surface (#332)

- [ ] **P6-T1 / AC-07:** Write failing Python API, CLI help, exit-code,
  diagnostic-redaction, and clean-install tests.
- [ ] **P6-T2 / AC-07:** Add `croissant`, `frictionless`, and aggregate
  `ingestion` extras.
- [ ] **P6-T3 / AC-07:** Implement `ingest validate`, `ingest inspect`,
  `ingest normalize`, and `calculate-from-dataset` with explicit selection,
  binding, offline, and source-policy options.
- [ ] **P6-T4 / AC-07, AC-11:** Make inspection output include data-quality,
  provider-capability, binding-resolution, governance, and materialization
  receipt details in stable machine-readable form.
- [ ] **P6-T5 / AC-07:** Add Python, Croissant/ML, and
  Frictionless/operations-research examples.
- [ ] **P6-T6 / AC-07:** Update Astro data-structure, CLI, architecture, and
  security guidance plus README, changelog, roadmap, and todo.
- [ ] **P6-T7 / AC-07:** Run automated review, CLI/docs/Vale validation, clean
  install checks, and the phase checkpoint protocol.

## Phase 7 — Security, performance, compatibility, and release (#333)

- [ ] **P7-T1 / AC-08:** Write failing traversal, archive-bomb, SSRF,
  unauthorized-network, secret-leakage, unsafe-transform, and resource-limit
  tests.
- [ ] **P7-T2 / AC-08, AC-11:** Add DNS-rebinding, redirect-policy,
  cache-poisoning, checksum-mismatch, decompression-ratio, and mutable-live-data
  tests.
- [ ] **P7-T3 / AC-08, AC-11:** Complete source-policy enforcement,
  content-addressed verified caching, immutable materialization receipts,
  offline replay, and streaming or bounded-batch behavior.
- [ ] **P7-T4 / AC-08:** Benchmark parsing, normalization, Arrow conversion,
  memory use, and calculation separately; define representative
  non-regression thresholds.
- [ ] **P7-T5 / AC-08:** Verify Python 3.12–3.14, minimum/maximum dependencies,
  CPU fallback, numerical equivalence, Arrow round trips, base/extra wheels,
  license inventory, and SBOM changes.
- [ ] **P7-T6 / AC-08:** Run typing, Ruff, coverage, mutation targets,
  dependency audits, repository harness, full `tox`, and all hosted checks.
- [ ] **P7-T7 / AC-08:** Publish supported-standard compatibility and
  deprecation policy without claiming unsupported upstream coverage.
- [ ] **P7-T8 / AC-08:** Run automated review, resolve high-confidence
  findings, and complete the final implementation checkpoint.

## Planning review enhancements (2026-07-24)

- [x] **REV-T1:** Incorporate the pre-implementation architecture review into
  the specification, implementation phases, GitHub sub-issue checklists, and
  Project 28 records. (`67e079e`)
- [x] **REV-T2:** Validate the amended planning artifacts, record review
  evidence, and reconcile PR #334 without claiming functional implementation.
  (`67e079e`)

## Phase 8 — Reconcile Conductor and GitHub (#325)

- [ ] **P8-T1 / AC-09:** Reconcile every plan task with issues #326–#333,
  native parent/sub-issue links, Project 28 status/fields, pull requests, and
  evidence ledger entries.
- [ ] **P8-T2 / AC-09:** Confirm every issue acceptance criterion is supported
  by repository and hosted evidence or remains explicitly blocked.
- [ ] **P8-T3 / AC-09:** Run the complete Conductor validation and distinguish
  this track's state from pre-existing legacy archive-validation debt.
- [ ] **P8-T4 / AC-09:** Update metadata and registry status, perform the final
  automated Conductor review, and archive only when all track acceptance
  criteria are satisfied.

## Current execution boundary

- Track and issue/project initialization are authorized and complete.
- Functional implementation has not started.
- Publication, external submission, authenticated dataset access, and
  relaxation of security or quality gates are not authorized by this plan.
