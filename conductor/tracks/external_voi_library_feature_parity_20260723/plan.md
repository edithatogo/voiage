# Track Implementation Plan: Comprehensive VOI Software Landscape And Improvement Review

## Phase 1: Existing reproducible baseline

- [x] Add failing registry, freshness, license, and traceability tests.
  (`2630e39`)
- [x] Define discovery, inclusion, feature, parity, and exclusion schemas.
  (`2630e39`)
- [x] Pin initial package/tool candidates and authoritative sources.
  (`2630e39`)
- [x] Commit, attach a git note, record the short commit SHA, and commit the
  plan update. (`2630e39`)
- [x] Automated review and validation checkpoint. (`c4c4fab`; registry,
  governance, Ruff, and live GitHub reconciliation passed)
- [x] Conductor - Analyst Manual Verification 'Phase 1: Existing reproducible baseline'
  (`87a7148`; analyst approved 2026-07-27)
  (Protocol in workflow.md).

## Phase 2: Comprehensive schema and inventory

- [x] Under [#569](https://github.com/edithatogo/voiage/issues/569), freeze
  `landscape-schema-review-protocol`: nested product/version/schema/feature/
  subfeature/option records, evidence strength, rights, duplicate resolution,
  inclusion/exclusion, freshness, review, and deterministic generation.
  (`24e0f54`)
- [x] Add failing schema, representative-record, evidence-ordering,
  observability, duplicate, rights, and freshness tests before expanding the
  current baseline.
  (`6d4deda`)
- [x] Under [#565](https://github.com/edithatogo/voiage/issues/565), complete
  `landscape-open-source-inventory` across registries, source hosts, archives,
  papers, supplements, HTA, decision analysis, Bayesian OED, active learning,
  causal policy, forecasting, optimization, and information economics.
  (`30ffda6`)
- [x] Include the exact #593--#600 families and their named submethods in
  reproducible search queries and capability extraction; map observed software
  to the residual planning register pending additive scientific review.
  (`b86e212`)
- [x] For each open-source product, inspect version-pinned source, API,
  schemas, functions/classes/commands, algorithms, estimators, features,
  subfeatures, options/defaults, diagnostics, errors, plots, reports, examples,
  tests, interoperability, dependencies, performance, license, and maintenance.
  (`e484829`, `7edbce0`)
- [x] Under [#568](https://github.com/edithatogo/voiage/issues/568), complete
  `landscape-commercial-hosted-inventory` using only observable evidence,
  with evidence strength, extraction limitations, closest VOIAGE workflow,
  user impact, and review due.
  (`885b4d9`)
- [x] Reconcile forks, renamed packages, inactive tools, unavailable
  supplements, commercial products, spreadsheet tools, hosted services, and
  adjacent systems without treating the list as universally exhaustive.
  (`f703551`)
- [x] Preserve the existing source/test/docs/example inventory, independent
  fixtures, exclusions, license normalizations, and archived-tool records
  unless the expanded schema invalidates them; reopen affected rows explicitly.
  (`3f8ead3`)
- [x] Commit, attach a git note, record the short commit SHA, and commit the
  plan update.
  (`3f8ead3`)
- [x] Automated review and validation checkpoint. (`2c728b9`; complete
  serialized matrix, generated-artifact, semantic inventory, Rust/polyglot,
  and diff checks passed)
- [x] Conductor - Analyst Manual Verification 'Phase 2: Comprehensive schema and inventory'
  (`user approval 2026-07-27`; Protocol in workflow.md).

## Phase 3: Capability map and reviewed improvement proposal

- [x] Under [#573](https://github.com/edithatogo/voiage/issues/573), generate
  `landscape-capability-adoption-map` for methods, schemas, options, workflows,
  UX, reporting, collaboration, governance, integrations, deployment,
  accessibility, and industry templates.
  (`7277151`)
- [ ] Map every capability to a canonical ID and `native`, `equivalent`,
  `adapter`, `planned`, `excluded`, or `not-reproducible`; require independent
  fixtures/tests for every positive parity claim.
- [ ] Keep residual candidate mappings separate from canonical parity rows and
  report whether products implement their estimand, an estimator, a diagnostic,
  an alias, an application, or an adjacent analysis.
- [ ] Generate deterministic views and summaries by product, ecosystem,
  capability, method, domain, parity, evidence, maintenance, license, adoption
  lesson, gap, MoSCoW, priority, risk, and review date.
- [ ] Under [#567](https://github.com/edithatogo/voiage/issues/567), generate
  `landscape-gap-review-roadmap-proposal` with user value, roles/domains,
  novelty, evidence, dependencies, design, licensing, MoSCoW, priority, effort,
  maturity, owner, proposed issue, alternatives, and decision state.
- [ ] Prove duplicate-resistant issue routing in dry-run mode. Do not create,
  close, or reparent implementation issues from the review generator.
- [ ] Run license, provenance, schemas, generated-artifact, docs, complete
  quality, competitor-absent, live hierarchy, and Project 28 gates.
- [ ] Present the checksum-bound proposal for named analyst review. Preserve
  approved, rejected, revised, and deferred decisions individually.
- [ ] After review, prepare a separate proposed roadmap change containing only
  approved recommendations; do not apply it in this track without renewed
  authorization.
- [ ] Commit, attach a git note, record the short commit SHA, and commit the
  plan update.
- [ ] Final review and validation checkpoint.
- [ ] Conductor - Analyst Manual Verification 'Phase 3: Capability map and reviewed improvement proposal'
  (Protocol in workflow.md).

## Preserved completed evidence

- [x] The original feature census inspected source, tests, documentation,
  examples, schemas, and releases for the 27-tool baseline. (`5dadf91`,
  `7443253`, `ae77dc1`)
- [x] Independent fixtures and feature-to-method mappings exist. (`6e3ebb1`)
- [x] Rust-authoritative expected opportunity loss was added. (`95beb20`)
- [x] Negative parity states retain reviewed exclusions and closest workflows.
  (`e1080ef`)
- [x] The generated comparison, quarterly freshness automation, ecosystem-drift
  proposal, and repository gates passed at the recorded revisions. (`2630e39`,
  `6e3ebb1`, `534d7a1`, `f32573d`)
- [x] Feature evidence remains distinct from method-level scientific evidence,
  and mutable upstream `HEAD` links were replaced with pinned revisions.
  (`c4c4fab`, `ae77dc1`)

## Approval and execution boundary

The user has selected this as the next review programme. Implementation of
recommendations discovered by the programme is not pre-approved: #567 must
present them for review before a later roadmap or runtime change.
