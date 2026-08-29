# Implementation Plan: Pre-Submission Comprehensive Hardening

## Phase 0 — Consolidation and immutable baseline

- [ ] **P0-T1 / AC-01:** Create the approved canonical track and validate its
  specification, plan, metadata, index, evidence ledger, and registry entry.
- [ ] **P0-T2 / AC-01:** Build a task-level migration manifest for all 21 active
  tracks, preserving completed evidence and classifying every remaining task.
- [ ] **P0-T3 / AC-01:** Mark the source tracks superseded, append migration
  receipts, archive them, update the registry, and prove Conductor integrity.
- [ ] **P0-T4 / AC-01:** Run automated phase review and the focused governance
  validation checkpoint.

## Phase 1 — Current requirements and whole-product gap analysis

- [ ] **P1-T1 / AC-02, AC-09:** Pin the current pyOpenSci, rOpenSci, JOSS,
  packaging, ecosystem, security, and research-software criteria to a dated
  source manifest.
- [ ] **P1-T2 / AC-02:** Refresh the external feature landscape and core/frontier
  method census from primary literature and authoritative software sources.
- [ ] **P1-T3 / AC-02:** Audit integration, data, worked-example, domain,
  interoperability, and executable research-workflow coverage.
- [ ] **P1-T4 / AC-02, AC-05:** Audit repository structure, Rust/Python ownership,
  public API, C ABI, R/Julia bindings, schemas, serialization, CLI, and installed
  artifacts.
- [ ] **P1-T5 / AC-02, AC-07:** Run `uv lock --upgrade`, the strict dependency
  frontier, licence/security audits, and a bounded stable/preview dependency
  alternatives analysis.
- [ ] **P1-T6 / AC-02, AC-08:** Capture reproducible baseline timings for local
  tox, pytest collection/execution, repository harness, Rust, R, Julia, docs,
  and representative hosted workflows.
- [ ] **P1-T7 / AC-02, AC-08:** Profile representative Python test lanes with
  Scalene, separating CPU, memory, import/collection, I/O, subprocess, and
  serialization costs; record tool limitations.
- [ ] **P1-T8 / AC-02:** Run automated phase review and validation checkpoint.

## Phase 2 — Disposition and architecture freeze

- [ ] **P2-T1 / AC-03:** Produce the canonical finding ledger and disposition
  every gap as must-fix, accepted limitation, preview, reviewed exclusion,
  external gate, or human gate.
- [ ] **P2-T2 / AC-03, AC-05:** Freeze the target stable architecture, package
  boundaries, API/ABI compatibility policy, and binding capability matrix.
- [ ] **P2-T3 / AC-03, AC-07:** Freeze dependency promotion and rollback rules,
  including named preview extras and CPU-fallback requirements.
- [ ] **P2-T4 / AC-03, AC-08:** Freeze the CI optimization design: deterministic
  shards, caches, reusable build artifacts, concurrency/cancellation, focused
  change lanes, and full release validation.
- [ ] **P2-T5 / AC-03:** Run automated phase review and validation checkpoint.

## Phase 3 — Core, analytical, data, and integration repairs

- [ ] **P3-T1 / AC-04, AC-05:** Add failing contract, reference, property, and
  compatibility tests for accepted core/API/ABI findings.
- [ ] **P3-T2 / AC-04, AC-05:** Implement the accepted Rust, Python, schema,
  serialization, CLI, and installed-artifact repairs.
- [ ] **P3-T3 / AC-04:** Add failing tests for accepted analytical, diagnostic,
  data, integration, and worked-example findings.
- [ ] **P3-T4 / AC-04:** Implement accepted analytical, diagnostic, data,
  integration, and worked-example repairs or record reviewed exclusions.
- [ ] **P3-T5 / AC-04:** Synchronize capability registries, public docs,
  changelog, roadmap, todo, and migration guidance.
- [ ] **P3-T6 / AC-04, AC-05:** Run automated phase review and validation
  checkpoint.

## Phase 4 — Standalone R and polyglot package hardening

- [ ] **P4-T1 / AC-06:** Add failing clean-install tests for a self-contained
  `voiageR` source package and reconcile `NeedsCompilation` and system
  requirements with the chosen Rust bridge.
- [ ] **P4-T2 / AC-06:** Implement the standalone Rust/R build architecture and
  installed runtime tests without an undeclared external shared library.
- [ ] **P4-T3 / AC-06:** Map every applicable rOpenSci statistical standard with
  item-level `@srrstats` or justified `@srrstatsNA` evidence.
- [ ] **P4-T4 / AC-06:** Repair the repository `pkgcheck` runner and pass
  `pkgcheck`, coverage, examples, vignettes, and `R CMD check --as-cran` on the
  supported platform matrix.
- [ ] **P4-T5 / AC-05, AC-06:** Validate and document Julia, C ABI, and Python
  installed-package compatibility against shared numerical fixtures.
- [ ] **P4-T6 / AC-05, AC-06:** Run automated phase review and validation
  checkpoint.

## Phase 5 — Dependency, CI/CD, and test-performance improvements

- [ ] **P5-T1 / AC-07:** Implement accepted stable dependency updates and keep
  experimental/preview dependencies in named non-blocking lanes.
- [ ] **P5-T2 / AC-07:** Prove numerical equivalence, Arrow round trips,
  CPU fallback, compatibility, provenance, and security for each promoted
  dependency or preview feature.
- [ ] **P5-T3 / AC-08:** Apply measured Python test optimizations, including
  fixture scope, import/collection, serialization, subprocess, and parallel
  execution improvements.
- [ ] **P5-T4 / AC-08:** Apply measured Rust, R, Julia, docs, and packaging test
  optimizations using their native profiling and caching evidence.
- [ ] **P5-T5 / AC-08:** Implement deterministic CI sharding, dependency/build
  caches, reusable artifacts, concurrency cancellation, and focused PR lanes
  while preserving the full release gate.
- [ ] **P5-T6 / AC-08:** Re-run the profiling and timing matrix, quantify the
  improvement, and demonstrate unchanged correctness, coverage, and required
  gate semantics.
- [ ] **P5-T7 / AC-07, AC-08:** Run automated phase review and validation
  checkpoint.

## Phase 6 — Venue, manuscript, governance, and release alignment

- [ ] **P6-T1 / AC-09:** Refresh and satisfy all repository-controlled
  pyOpenSci requirements and stage only unchecked human attestations.
- [ ] **P6-T2 / AC-06, AC-09:** Refresh and satisfy all repository-controlled
  rOpenSci requirements and stage the pre-submission inquiry without posting.
- [ ] **P6-T3 / AC-09:** Refresh the JOSS article contract, AI disclosure,
  citation/source audit, research-use boundary, and current review criteria.
- [ ] **P6-T4 / AC-09:** Bind the Python, Rust, R, Julia, docs, JOSS manuscript,
  arXiv source, SBOM, provenance, and release evidence to the exact final
  candidate version and revision.
- [ ] **P6-T5 / AC-04, AC-09:** Reconcile governance, sustainability, support,
  contribution, AI-transparency, badge, registry, and archival readiness
  without performing external actions.
- [ ] **P6-T6 / AC-09:** Run automated phase review and validation checkpoint.

## Phase 7 — Whole-programme assurance and submission freeze

- [ ] **P7-T1 / AC-10:** Run the repository harness, complete tox matrix,
  language-native checks, documentation builds, security/dependency audits,
  mutation/coverage gates, and full Conductor validation.
- [ ] **P7-T2 / AC-04, AC-10:** Run independent whole-programme review, repair
  all Critical/High/Medium findings, and rerun the complete gate.
- [ ] **P7-T3 / AC-08, AC-10:** Record final local and hosted performance and
  exact-head required-check evidence without weakening any required gate.
- [ ] **P7-T4 / AC-10:** Freeze the submission candidate, confirm a clean tree,
  record every remaining external/human gate, and explicitly attest that no
  submission was performed.
- [ ] **P7-T5 / AC-10:** Mark the repository programme complete and archive it
  only after every repository acceptance criterion passes.

