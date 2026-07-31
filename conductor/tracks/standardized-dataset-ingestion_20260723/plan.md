# Implementation Plan: Standardized Dataset Ingestion

`plan.md` is the task source of truth. Tests precede implementation in every
functional phase. Each phase ends with automated review, focused validation,
and a Conductor checkpoint under `conductor/workflow.md`.

## Phase 1 — Freeze the normalized input contract (#326)

- [x] **P1-T1 / AC-01:** Write failing contract tests for strict validation,
  immutability, explicit VOI bindings, canonical JSON, schema/content
  fingerprints, secret redaction, and Arrow IPC/Parquet round trips. (#519)
- [x] **P1-T2 / AC-01:** Add the versioned dataset, resource, table, field,
  relationship, provenance, diagnostic, and binding models under
  `voiage/contracts/`. (`41e829c`)
- [x] **P1-T3 / AC-01:** Add
  `specs/core-api/schemas/v2/normalized-input-bundle.schema.json`, compatibility
  rules, exports, and deterministic golden fixtures. (`0af240c`)
- [x] **P1-T4 / AC-10:** Define the independently versioned VOI binding profile,
  its JSON Schema, canonical serialization, digest, precedence rules, units,
  perspectives, transformations, and method-applicability validation. (`fa91f90`)
- [x] **P1-T5 / AC-01, AC-10:** Add unsupported-version, conflicting-binding,
  stale-reference, unit-incompatibility, and namespaced-extension tests. (`a2e082f`)
- [x] **P1-T6 / AC-01:** Verify that the core contract imports no external
  ingestion dependency. (`b13a76c`)
- [x] **P1-T7 / AC-01, AC-10:** Run automated review, focused tests, schema validation,
  Arrow/Polars fresh-process round trips, and the phase checkpoint protocol. (#523)

## Phase 2 — Prepare existing VOI runtime inputs (#327)

- [x] **P2-T1 / AC-02:** Write failing tests for wide and long net-benefit
  tables, cost/outcome derivation, perspectives, parameter preparation, sample
  alignment, joins, strategies, nulls, dtypes, and cardinality. (#525)
- [x] **P2-T2 / AC-02:** Implement `PreparedAnalysisInputs` and the
  format-neutral preparation layer. (#525)
- [x] **P2-T3 / AC-02, AC-11:** Emit a machine-readable data-quality report for
  row counts, missingness, uniqueness, keys, join coverage, coercions,
  exclusions, and selected records or partitions. (`944feec`)
- [x] **P2-T4 / AC-02:** Propagate artifact IDs and normalized input digests
  through `AnalysisSpec`, `RunContext`, diagnostics, and provenance. (`5239de1`)
- [x] **P2-T5 / AC-02:** Add the method/input capability matrix and preserve
  existing direct Python and CSV entry points. (`ccd61b6`)
- [x] **P2-T6 / AC-02:** Verify that filtering, projection, sampling,
  aggregation, and exclusion cannot occur implicitly. (#525)
- [x] **P2-T7 / AC-02, AC-11:** Run direct-versus-normalized numerical-equivalence
  tests, automated review, validation, and the phase checkpoint protocol. (#525)

## Phase 3 — Establish the optional provider boundary (#328)

- [x] **P3-T1 / AC-03:** Write failing fake-provider, missing-extra,
  import-isolation, error-taxonomy, source-policy, and resolver-injection tests.
- [x] **P3-T2 / AC-03:** Implement the `IngestionProvider` protocol, provider
  registry, stable errors, and conservative provider probing.
- [x] **P3-T3 / AC-03:** Implement `SourceAccessPolicy` and the deterministic,
  dependency-injected resource-resolution boundary.
- [x] **P3-T4 / AC-03:** Add provider capability declarations for versions,
  media types, transforms, projection, filtering, streaming, and random access.
- [x] **P3-T5 / AC-03:** Add opt-in, allow-listed Python entry-point discovery;
  prohibit automatic third-party imports during package import and probing.
- [x] **P3-T6 / AC-03:** Document the third-party provider extension contract.
- [x] **P3-T7 / AC-03:** Verify clean base installation, run automated review,
  focused validation, and the phase checkpoint protocol.

## Phase 4 — Implement the two source-format providers (#329, #330)

### Croissant ML

- [x] **P4-T1 / AC-04:** Run `uv lock --upgrade` and
  `python scripts/dependency_frontier.py . --strict`; review and record the
  Croissant parser dependency and supported-profile decision before changing
  dependencies. (`e41591e`)
- [~] **P4-T2 / AC-04:** Write failing offline fixtures/tests for versioning,
  identities, resources, record sets, fields, keys, references, splits,
  supported transformations, integrity failures, archives, nesting, and
  ambiguous semantics. Expanded integrity-declaration, non-CSV-media-type,
  field-source, and malformed collection-entry corpus coverage. Focused corpus
  and provider validation pass; full tox remains blocked by the pre-existing
  missing source-provenance and JOSS contract files in this clean worktree.
  Non-object descriptor-root handling is covered by the shared provider guard
  (`333ee68f`, `d745d2d6`, `1e4e6bd7`). JSON-LD context-array and exact-version
  adversarial coverage: `5a3a7b04`. A combined context-array/governance fixture
  now proves descriptor-only inspection, retained non-semantic governance, and
  materialization receipt fields while rejecting unexpanded JSON-LD context
  objects (`d814e6e1`, partial).
- [~] **P4-T3 / AC-04, AC-11:** Add fixtures for Croissant 1.1 conformance,
  parser-feature gaps, live datasets, citations, PROV, usage information,
  ODRL, and RAI metadata preservation. Offline governance fixture added
  (`9a5b45b7`); the authoritative live probe remains an explicit external gate.
- [~] **P4-T4 / AC-04:** Implement the lazy optional Croissant provider and
  publish separate standard-conformance and parser-capability profiles. Public
  provider export is now lazy (`2ad0a24a`, partial); profile acceptance evidence
  remains active.
- [~] **P4-T5 / AC-04, AC-11:** Add Croissant inspection, diagnostics,
  provenance, governance metadata, and one opt-in authoritative live
  interoperability probe. Registry/CLI inspection now verifies stable provider
  capabilities without resource materialization; materializing validation
  verifies receipt, provenance, and retained governance for the offline
  Croissant 1.1 fixture. The opt-in authoritative live probe remains an
  explicit external gate.

### Frictionless Data

- [x] **P4-T6 / AC-05:** Review and record the Frictionless dependency and
  supported-profile decision through the same dependency-frontier evidence. (`frictionless==5.19.0` decision)
- [~] **P4-T7 / AC-05:** Write failing offline fixtures/tests for package and
  data validation, resources, schemas, dialects, types, constraints, missing
  values, keys, integrity, governance metadata, supported tabular formats, and
  ambiguous resources. File-backed baseline and fail-closed format/integrity
  coverage added (`d0c1b238`); malformed-resource coverage is also file-backed.
  Non-object descriptor-root handling is covered by the shared provider guard
  (`0c19fe1b`, `1e4e6bd7`); duplicate CSV-header and schema-field ambiguity
  now rejects through the stable ingestion error boundary (`8f60b184`,
  partial). Remaining acceptance evidence is tracked below.
- [~] **P4-T8 / AC-05:** Implement the lazy optional Frictionless provider and
  documented supported profile. Public provider export is now lazy; profile
  acceptance evidence remains active (`2ad0a24a`, partial).
- [~] **P4-T9 / AC-05, AC-11:** Add Frictionless inspection, diagnostics,
  provenance, licence/citation/usage preservation, and one opt-in authoritative
  live interoperability probe. Registry/CLI inspection now verifies stable
  provider capabilities without resource materialization; materializing
  validation verifies receipt, provenance, and retained governance for an
  offline Data Package fixture. The opt-in authoritative live probe remains an
  explicit external gate.

### Phase checkpoint

- [~] **P4-T10 / AC-03–AC-05, AC-11:** Verify base-import isolation and clean installs
  for each extra; run automated review, focused tests, dependency/security
  audits, and the phase checkpoint protocol. A wheel built from `46ef7873`
  installed and passed import-isolation/registry checks in fresh Python 3.14
  environments for base, `croissant`, `frictionless`, and `ingestion` extras
  (partial); the full phase checkpoint remains pending.

## Phase 5 — Prove cross-format conformance (#331)

- [~] **P5-T1 / AC-06:** Define the canonical decision fixture and failing
  parity assertions across Croissant, Frictionless, Arrow IPC, Parquet, and
  direct NumPy/xarray representations. The canonical corpus now asserts the
  same explicit preparation result across all listed source representations,
  including equal xarray and NumPy runtime views (partial: parser-differential
  acceptance remains active under P5-T5).
- [~] **P5-T2 / AC-06:** Add the deterministic fixture manifest with pinned
  descriptor, resource, schema, and content digests. Both checked-in fixture
  corpora are now verified through one deterministic digest-manifest validator
  (`scripts/validate_standardized_ingestion_fixtures.py`). It pins descriptor,
  resource, format-neutral schema, and direct normalized-content digests
  (partial: the corpus remains intentionally small).
- [~] **P5-T3 / AC-06:** Implement deterministic fixture generation and the
  schema, provenance, ordering, meaningful-change, and numerical-equivalence
  conformance matrix. The fixture validator now fails closed on stale artifact
  and normalized-identity digests and supports explicit deterministic
  ``--write`` regeneration. Provider declarations that disagree with CSV order
  fail rather than reorder data, and changed resource bytes change receipts and
  normalized content (partial: full parser-differential coverage remains
  active).
- [~] **P5-T4 / AC-06, AC-10, AC-11:** Assert binding-profile, data-quality,
  governance-metadata, and materialization-receipt parity without requiring
  source formats to share irrelevant descriptive metadata. Canonical Croissant
  and Frictionless fixtures now assert explicit binding, binding-profile,
  data-quality, and receipt parity while leaving format-specific descriptive
  provenance independent (`e9a22f0c`, partial).
- [~] **P5-T5 / AC-06:** Add malformed/adversarial cases, property-based mapping
  tests, parser-differential checks, and fresh-process PyArrow/Polars checks.
  A fresh-process IPC/Parquet round-trip now reads normalized bundles and
  converts their Arrow tables through Polars before asserting identical schema
  and rows (`b16b52df`). Generated Croissant/Frictionless CSV mappings now
  assert identical rows and explicit binding preparation, while malformed
  parser nodes and mismatched declared resource digests fail through the
  stable error taxonomy (partial: broader parser-differential coverage remains
  active; `0c6a2e7a`).
- [~] **P5-T6 / AC-06:** Add the conformance matrix to tox and hosted CI; run
  automated review, validation, and the phase checkpoint protocol. The explicit
  `ingestion-conformance` tox environment and named hosted job now run the
  canonical cross-format, fixture-digest, and reference-case matrix (partial:
  hosted result and phase checkpoint remain active).

## Phase 6 — Ship the user-facing product surface (#332)

- [~] **P6-T1 / AC-07:** Write failing Python API, CLI help, exit-code,
  diagnostic-redaction, and clean-install tests. The CLI now explicitly proves
  that a rejected credential-bearing descriptor URI is redacted at the
  user-facing boundary, and all four ingestion commands have executable help
  contracts. Explicit non-default source-root tests now cover all materializing
  commands and fail closed at the resource-byte limit. Provider/source-policy,
  binding, and output failures now have stable differentiated exit codes while
  Typer-owned usage errors retain exit code 2; explicit provider and binding
  profile selection are enforced, and unsupported resource projection remains
  absent (partial: complete Phase 6 acceptance reconciliation remains active).
- [x] **P6-T2 / AC-07:** Add `croissant`, `frictionless`, and aggregate
  `ingestion` extras. The declared extras remain dependency-neutral because
  built-in providers require only the base Arrow/JSON stack.
- [x] **P6-T3 / AC-07:** Implement `ingest validate`, `ingest inspect`,
  `ingest normalize`, and `calculate-from-dataset` with explicit selection,
  binding, offline, and source-policy options. Every materializing command now
  accepts an explicit `--source-root` in addition to the cache, offline, and
  resource-size policy controls.
- [~] **P6-T4 / AC-07, AC-11:** Keep inspection and materialization evidence
  distinct in stable machine-readable output. `ingest inspect` is now
  descriptor-only (provider capabilities and an explicit null binding
  resolution), so it cannot accidentally resolve resources; materializing
  validation/normalization output carries provenance, governance, receipts,
  and data-quality evidence. Broader receipt and live-source acceptance remains
  active.
- [x] **P6-T5 / AC-07:** Add Python, Croissant/ML, and
  Frictionless/operations-research examples.
- [~] **P6-T6 / AC-07:** Update Astro data-structure, CLI, architecture, and
  security guidance plus README, changelog, roadmap, and todo. The Astro guide,
  changelog, roadmap/todo, and README now describe the supported profile,
  explicit safety boundary, CLI, cross-domain examples, CLI exit taxonomy, and
  explicit provider/binding-profile boundary; final docs/links and
  phase-checkpoint evidence remains active.
- [ ] **P6-T7 / AC-07:** Run automated review, CLI/docs/Vale validation, clean
  install checks, and the phase checkpoint protocol.

## Phase 7 — Security, performance, compatibility, and release (#333)

- [~] **P7-T1 / AC-08:** Write failing traversal, archive-bomb, SSRF,
  unauthorized-network, secret-leakage, unsafe-transform, and resource-limit
  tests. Local policy coverage now rejects URI schemes (including `file:` and
  `data:` forms) and archive suffixes before any file, DNS, redirect, or archive
  operation; archive extraction remains explicitly unsupported (`749083d`,
  partial).
- [~] **P7-T2 / AC-08, AC-11:** Add DNS-rebinding, redirect-policy,
  cache-poisoning, checksum-mismatch, decompression-ratio, and mutable-live-data
  tests. Partial evidence: `d3550e9c` rejects a cache-namespace symlink whose
  resolved directory escapes the configured cache root before it can redirect a
  verified materialization; hard-linked cache entries are also rejected so a
  verified object cannot share a writable inode with an alternate path
  (partial). The remote/archive and mutable-live-source cases remain active;
  authoritative live probes stay externally gated.
- [~] **P7-T3 / AC-08, AC-11:** Complete source-policy enforcement,
  content-addressed verified caching, immutable materialization receipts,
  offline replay, and streaming or bounded-batch behavior. Built-in CSV/TSV
  parsing now streams Arrow record batches and rejects configured per-resource
  row ceilings and internal batch ceilings before retaining a batch or
  constructing a table; all materializing CLI commands expose the explicit
  `--max-resource-rows` policy option (partial: remote/archive streaming and
  their policy evidence remain active).
- [~] **P7-T4 / AC-08:** Benchmark parsing, normalization, Arrow conversion,
  memory use, and calculation separately; define representative
  non-regression thresholds.
- [~] **P7-T5 / AC-08:** Verify Python 3.12–3.14, minimum/maximum dependencies,
  CPU fallback, numerical equivalence, Arrow round trips, base/extra wheels,
  license inventory, and SBOM changes.
- [~] **P7-T6 / AC-08:** Run typing, Ruff, coverage, mutation targets,
  dependency audits, repository harness, full `tox`, and all hosted checks.
- [~] **P7-T7 / AC-08:** Publish supported-standard compatibility and
  deprecation policy without claiming unsupported upstream coverage. The Astro
  support matrix now distinguishes conservative supported Croissant,
  Frictionless, DataFrame, and normalized Arrow profiles from rejected
  remote/archive/transform and authoritative-live paths (`1dc545a7`, partial).
- [~] **P7-T8 / AC-08:** Run automated review, resolve high-confidence
  findings, and complete the final implementation checkpoint.

## Planning review enhancements (2026-07-24)

- [x] **REV-T1:** Incorporate the pre-implementation architecture review into
  the specification, implementation phases, GitHub sub-issue checklists, and
  Project 28 records. (`67e079e`)
- [x] **REV-T2:** Validate the amended planning artifacts, record review
  evidence, and reconcile PR #334 without claiming functional implementation.
  (`67e079e`)

## Phase 8 — Publish the provider SDK and DataFrame adapter (#467)

- [~] **P8-T1 / AC-12:** Freeze the supported provider-SDK surface only after
  phases 1–5 establish stable core contracts and conformance evidence.
- [~] **P8-T2 / AC-12:** Add typed protocol stubs, a minimal example provider,
  reusable contract tests, capability manifests, compatibility rules, and an
  opt-in entry-point publication checklist. The SDK v1 now has a versioned
  machine-readable consumer fixture, regression tests for its public protocol
  and capability fields, and registry rejection of empty provider identities
  (partial: a separately installed third-party provider remains future evidence).
- [~] **P8-T3 / AC-12:** Write failing DataFrame-interchange tests covering
  pandas, Polars, dtype/null/category/timezone/index handling, copy diagnostics,
  and clean optional environments. Partial diagnostics evidence: `fdda14a`;
  producer-specific nullable/category/timezone/index consumer evidence is added
  in this increment. Dependency-neutral `croissant`, `frictionless`, and
  aggregate `ingestion` extras now have subprocess import-isolation regression
  coverage while enhanced parser modules are absent (partial: actual enhanced
  parser extras remain intentionally unpromoted).
- [~] **P8-T4 / AC-12:** Implement the generic `__dataframe__` adapter through
  Arrow and `NormalizedInputBundle`, with no alternate preparation or numerical
  path. Partial conversion-diagnostics evidence: `fdda14a`.
- [~] **P8-T5 / AC-12:** Assess Hugging Face and OpenML Croissant support and
  create registry-specific providers only for documented, tested gaps.
- [~] **P8-T6 / AC-12:** Run SDK consumer tests, conformance, numerical
  equivalence, import isolation, security review, full tox, and hosted checks.

## Phase 9 — Ship cross-domain reference cases (#468)

- [~] **P9-T1 / AC-13:** Define rights-cleared or deterministic synthetic ML,
  engineering/operations, and business decision cases with explicit method
  applicability.
- [~] **P9-T2 / AC-13:** Represent every case as Croissant, Frictionless, and
  direct inputs using the same binding profile and pinned artifact digests. The
  executable synthetic cases now exercise Croissant, Frictionless, direct Arrow,
  and DataFrame representations for each documented domain (partial; hosted
  evidence remains required).
- [~] **P9-T3 / AC-13:** Add validation, inspection, data-quality, governance,
  materialization, Python API, and CLI walkthrough evidence. The canonical ML
  Croissant and engineering Frictionless descriptors now execute a separate
  metadata-only inspection path and materializing validation/calculation path;
  the latter carries governance, binding-quality, and receipt evidence. The
  direct DataFrame business walkthrough remains covered by the executable
  reference example (partial).
- [~] **P9-T4 / AC-13:** Assert normalized-object and numerical equivalence
  without adding domain-specific kernels or semantic inference. The reference
  runner now fails if any supported representation differs in EVPI (partial;
  broader hosted evidence remains required).
- [~] **P9-T5 / AC-13:** Publish the support matrix and run fixture
  regeneration, docs, links, Vale, conformance, and hosted regression checks.

## Follow-on recommendation incorporation (2026-07-24)

- [x] **REV2-T1:** Add native sub-issues #467 and #468, Project 28 records,
  Conductor requirements/phases, central cross-references, and PR dependency
  notes for the provider SDK and cross-domain reference cases. (`211e7aa`)
- [x] **REV2-T2:** Validate and record the follow-on planning amendment without
  claiming that SDK, adapters, examples, or dependency remediation are
  implemented. (`211e7aa`)

## Phase 10 — Reconcile Conductor and GitHub (#325)

- [x] **P10-T1 / AC-09:** Reconcile every plan task with issues #326–#333,
  #467, and #468,
  native parent/sub-issue links, Project 28 status/fields, pull requests, and
  evidence ledger entries. (`7549b66e`)
- [x] **P10-T2 / AC-09:** Confirm every issue acceptance criterion is supported
  by repository and hosted evidence or remains explicitly blocked. (`7549b66e`)
- [x] **P10-T3 / AC-09:** Run the complete Conductor validation and distinguish
  this track's state from pre-existing legacy archive-validation debt. (`7549b66e`)
- [~] **P10-T4 / AC-09:** Update metadata and registry status, perform the final
  automated Conductor review, and archive only when all track acceptance
  criteria are satisfied.

### Additive reconciliation update — 2026-07-31

`p10-reconciliation-20260731.md` maps the 30 merged ingestion increments in
#639–#690 to exact merge commits, representative changed artifacts, and their
hosted-check provenance. It records the absent, unmerged, open, and unrelated
PR numbers rather than guessing associations. Project 28 confirms that #325–#333,
#467, and #468 are present and `In Progress`; that field is not an
acceptance-criterion verdict. P10-T4 remains active: all current and external
acceptance boundaries still prohibit final review or archive.

## Current execution boundary

- Track and issue/project initialization are authorized and complete. PR #334
  merged on 2026-07-24 after PR #465 and its final rebase, with required CI and
  strict changed-line assurance passing.
- The merged baseline implements the normalized bundle, preparation boundary,
  optional Croissant/Frictionless CSV profiles, CLI, DataFrame interchange
  adapter, and reference documentation. It is not evidence for unchecked plan
  tasks or the entire issue acceptance checklists.
- Dependabot PR #324 is merged as a separate repository-security lane; it is
  not evidence that the remaining ingestion acceptance criteria are complete.
- The track, parent issue #325, child issues #326–#333 and #467–#468, and their
  Project 28 items remain active. Archive is prohibited until the unchecked
  plan tasks and issue acceptance criteria have supporting evidence.
- Phase 10 reconciliation on 2026-07-27 confirms that #325–#333, #467, and
  #468 are open; #627 is merged; the Phase 8 SDK and Phase 9 reference-case
  changes landed separately on `main` as `24e12977` and `c4106739`; and #634
  carries the remaining Phase 10 evidence migration. Project 28 was rechecked through GitHub Projects
  v2 after GraphQL access recovered; every linked item is now `In Progress`,
  consistent with its still-open issue and this active plan. This is status
  reconciliation, not completion evidence.
- Final Conductor review found that the pre-existing append-only evidence
  ledger fails schema validation at entries 9 and 10: both omit required
  artifact digests, and entry 10 records a non-zero validation command as
  `passed`. The immutable legacy ledger is preserved under its original SHA-256
  and a new valid ledger chain begins with a migration receipt; future evidence
  is now appendable without rewriting historical claims.
- Publication, external submission, authenticated dataset access, and
  relaxation of security or quality gates are not authorized by this plan.
