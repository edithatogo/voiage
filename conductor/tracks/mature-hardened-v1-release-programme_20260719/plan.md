# Mature Hardened v1.0 Architecture and Release Programme Plan

Execute phases sequentially. Existing child tracks are reconciled in Phase 1 and may be completed in place when their scope directly satisfies a later phase. Every implementation slice follows red, green and refactor, and every externally controlled result preserves the distinction between repository readiness and live approval or indexing.

## Phase 1: Programme Governance and Repository Reconciliation [checkpoint: 941691f]

- [x] Task: Establish the authoritative v1.0 programme baseline (76f9ba8)
    - [x] Reconcile local `main`, `origin/main`, open pull requests, remote branches, GitHub issues, releases and generated artifacts.
    - [x] Inventory every active and archived Conductor track.
    - [x] Classify tracks as required for v1.0, post-v1, externally blocked, superseded, duplicate or completed.
    - [x] Map valid existing tracks into this programme without reopening completed historical foundation work.
    - [x] Register the dependency-ordered execution queue.
- [x] Task: Reconcile roadmap and backlog sources (5495fef)
    - [x] Align `roadmap.md`, `todo.md`, GitHub issues and Conductor with the v1.0 specification.
    - [x] Remove contradictory, duplicate or stale completion claims.
    - [x] Establish one machine-verifiable source for v1.0 completion status.
- [x] Task: Implement programme integrity validation using TDD (1532b9d)
    - [x] Red: add failing tests for registry, track, roadmap and issue-state drift.
    - [x] Green: implement validation until the tests pass.
    - [x] Refactor: consolidate the validation into maintainable repository tooling and rerun the tests.
- [x] Task: Apply Phase 1 automated review fixes (4192632)
    - [x] Point the active registry entry to the track context index.
    - [x] Validate GitHub snapshot counts and blocked-pull-request evidence structure.
- [x] Review fix: refresh the machine-readable baseline after post-merge synchronization (1d0f31e)
    - [x] Update the authoritative commit, GitHub snapshot timestamp, open-issue count,
      and blocked-pull-request classification from the live remote state.
- [x] Task: Conductor - Automated Review and Checkpoint 'Programme Governance and Repository Reconciliation' (Protocol in workflow.md) (941691f)

## Phase 2: Stable-Core Contract and Compatibility Freeze [checkpoint: 62f38f7]

- [x] Task: Define the normative v1.0 API contract (958b217)
    - [x] Classify public symbols as stable, provisional, experimental, deprecated or removed.
    - [x] Specify core numerical, schema, diagnostic, reporting, provenance, plotting and CLI contracts.
    - [x] Define shapes, types, missing-value behavior, tolerances, seeds, determinism, warnings, errors and supported platforms.
- [x] Task: Build executable compatibility fixtures using TDD (81c5ca7)
    - [x] Red: add failing golden-fixture and contract tests for normal, edge and invalid inputs.
    - [x] Green: implement canonical datasets and language-neutral expected results until tests pass.
    - [x] Refactor: consolidate fixtures into a reusable conformance suite and rerun all tests.
- [x] Task: Establish semantic-versioning and deprecation policy (aeab996)
    - [x] Define compatibility guarantees for Rust, Python, C ABI, bindings, CLI, schemas and serialized outputs.
    - [x] Document deprecation periods and experimental exclusions.
- [x] Task: Apply Phase 2 automated review fixes (eef8f09)
    - [x] Reject negative and non-finite compatibility tolerances.
    - [x] Validate malformed expected outcomes and policy structure fail-closed.
    - [x] Clarify pre-v1 warning migration and replace message-derived error normalization.
    - [x] Add CEAF and dominance result schemas and canonical examples.
    - [x] Enforce normative Python exception types without adapter masking.
    - [x] Cover non-finite values, determinism, provenance and serialization evidence.
    - [x] Close policy-schema, metadata and contract-documentation findings.
    - [x] Align CEAF and dominance schemas with stable public runtime serialization.
    - [x] Enforce non-finite rejection across every stable method.
    - [x] Make seed, determinism and result provenance evidence executable.
    - [x] Reconcile the downstream compatibility runner contract.
- [x] Task: Conductor - Automated Review and Checkpoint 'Stable-Core Contract and Compatibility Freeze' (Protocol in workflow.md) (62f38f7)

## Phase 3: Rust Workspace, Domain Model and ABI Stabilization [checkpoint: 99b0a8c]

- [x] Task: Establish the production Rust workspace using TDD (570b2ad)
    - [x] Red: add failing workspace, feature-boundary and package-layout checks.
    - [x] Green: separate domain, numerics, diagnostics, serialization and test-support crates, with FFI and WASM as leaf adapter crates.
    - [x] Refactor: declare the MSRV and supported targets, remove cyclic and binding-specific core dependencies and rerun checks.
    - [x] Provenance deviation: the initial workspace skeleton commit `563982f` preceded the executable contract-test commit `568f5f6`; subsequent integration and review were test-gated, but this historical slice did not strictly follow Red-first ordering.
- [x] Task: Stabilize Rust domain contracts using TDD (0351ec3)
    - [x] Red: add failing tests for validated costs, effects, probabilities, samples, strategies, thresholds, outputs, diagnostics and provenance.
    - [x] Green: implement domain types, fail-closed serialization and explicit machine-readable errors.
    - [x] Refactor: prove canonical-schema, compatibility-fixture and unknown-field parity, then optimize ownership and allocation while preserving property and round-trip tests.
- [x] Task: Stabilize the narrow C ABI using TDD (1eaf562)
    - [x] Red: add failing layout, namespaced-symbol, ownership, error-transport, panic-containment, pointer/length/overflow and compatibility tests.
    - [x] Green: implement portable ABI version negotiation, structure size/version fields, opaque lifecycle primitives, bounded error transport, versioned headers and Rust exports.
    - [x] Refactor: freeze ABI infrastructure only, audit unsafe code and verify allocator pairing, symbol visibility, thread-safety, sanitizers, leaks and cross-compilers; defer operation-level ABI freezing until each Phase 5 kernel passes parity and profiling.
- [x] Task: Apply Phase 3 automated review fixes (d170ab1)
    - [x] Make portable C/C++ consumers fail on lifecycle or error-query failures.
    - [x] Derive cross-platform export checks from the canonical symbol baseline.
    - [x] Enforce every frozen ABI size, alignment and field offset from the canonical layout baseline.
    - [x] Use fixed-width stable wire counts and expose validated typed construction for every canonical result DTO.
    - [x] Add Cargo dependency updates, advisory/license/source policy, Miri unsafe-boundary analysis and measured Rust coverage.
    - [x] Reconcile TDD provenance, ABI evidence range and release-note wording.
- [x] Task: Conductor - Automated Review and Checkpoint 'Rust Workspace, Domain Model and ABI Stabilization' (Protocol in workflow.md) (99b0a8c)

## Phase 4: PyO3 and Maturin Runtime Bridge [checkpoint: daf72af]

- [x] Task: Introduce the production Python-to-Rust bridge using TDD (1c696b4)
    - [x] Red: add Python contract tests requiring stable operations to execute through Rust.
    - [x] Green: configure PyO3 and maturin and expose the narrow internal extension API.
    - [x] Refactor: preserve the public Python surface while minimizing conversion and allocation overhead.
- [x] Task: Harden Python packaging using TDD (5979799)
    - [x] Red: add failing sdist, wheel, clean-install and import tests.
    - [x] Green: implement dynamic versioning and supported platform and Python ABI builds.
    - [x] Refactor: minimize base dependencies and rerun the artifact matrix.
- [x] Task: Prove runtime provenance (ed459ee)
    - [x] Add tests that fail if stable operations silently use duplicate Python kernels.
    - [x] Expose diagnostic build and runtime information without expanding the public contract.
- [x] Task: Conductor - Automated Review and Checkpoint 'PyO3 and Maturin Runtime Bridge' (Protocol in workflow.md) (daf72af)

## Phase 5: Stable Numerical Kernel Migration

- [x] Task: Migrate deterministic foundational kernels using TDD (a6f2d7d)
    - [x] Red: add parity, edge, property and performance tests for EVPI, ENBS, dominance, frontier and CEAF behavior.
    - [x] Green: implement authoritative Rust kernels until shared fixtures pass.
    - [x] Refactor: consolidate numerical primitives and optimize allocation paths.
    - [x] Review fix: pin all benchmark checkout actions after hosted zizmor validation (c02de92).
    - [x] Review fix: add benchmark concurrency and verified current Rust toolchain pin (246c573).
    - [x] Review fix: add CEAF property coverage and committed benchmark workload contract (bb4afea).
    - [x] Review fix: reject malformed CEAF strategy indices without panicking (2936bed).
- [x] Task: Migrate supported EVPPI methods using TDD (b1b488c, 61b5595)
    - [x] Red: add fixture and parity tests for each retained EVPPI method (5d8e9e1).
    - [x] Green: implement the stable method set in Rust with deterministic diagnostics and errors (01e5ba4, 2a3abb5).
    - [x] Refactor: remove Python numerical-policy dependencies and rerun parity tests (b1b488c, 61b5595).
- [~] Task: Migrate supported EVSI methods using TDD
    - [x] Red: define the bounded seeded-bootstrap fixture and error contract (8ee1ecc).
    - [x] Green: implement the seeded-bootstrap numerical kernel and typed PyO3 bridge (cf5d3e2, 4ed2c43).
    - [x] Refactor: add reproducibility properties, one-strategy behavior, benchmark evidence, and focused validation (a9fb96b).
    - [x] Review fix: reject perfect-information overflow and repair the zero-state seed edge (a5a1222).
    - [x] Define and expose the deterministic efficient-linear kernel through the private PyO3/runtime bridge while retaining public callback dispatch (94c8d6d, a1ef46a, 392e926).
    - [x] Review fix: use scaled/online accumulation, active-pivot rank scaling, and a versioned efficient-linear result envelope (0131208).
    - [x] Document native EVSI ownership boundaries and retain fixture-backed migration status (f30ea5e).
    - [x] Define and expose the fail-closed deterministic moment-based kernel through the private PyO3/runtime bridge while retaining public callback dispatch (b5b80a5, 84b0b81).
    - [x] Harden and version the seeded-bootstrap result envelope with online finite-value accumulation (b9dc7ae).
    - [x] Route the public efficient + linear callback estimator through the native deterministic kernel with bridge/parity coverage (a8ac9d9).
    - [x] Harden the public route with native-envelope validation, Python fallback, rank-deficiency compatibility, and scaling coverage (a48285b).
    - [x] Resolve CodeQL mixed-import finding in public EVSI integration tests (6507642).
    - [x] Normalize the full sample-information test module to one import style for CodeQL (3b917d0).
    - [x] Freeze an optional seeded Python two-loop contract with local RNG isolation while retaining Python ownership (75df3f4, 9ecc3ef).
    - [x] Resolve redundant-import CodeQL findings from the seeded EVSI tests (7d1a4fb).
    - [x] Implement meaningful ``n_inner_loops`` semantics for the retained Python two-loop estimator (cc63e9b).
    - [x] Apply formatter review fix to the two-loop implementation (5c0ca57).
    - [x] Apply repository-gate review fix for Python 3.12 generic and type-alias syntax (772a4f8).
    - [x] Resolve the follow-on `ty` generic-parameter diagnostics in distributed helpers (bd8f70d).
    - [x] Restore compatibility-ordered public exports after Ruff sorting review (fabd89d).
    - [~] Extend native ownership to the Python model-callback estimator family after explicit seed and estimator contracts are frozen; retain two-loop, regression, random-forest efficient, adaptive, and NMA paths in Python meanwhile.
        - [x] Route the public moment-based estimator through the versioned Rust kernel with a rank-deficiency compatibility fallback (ac1b8e8).
        - [x] Add public moment-based envelope and compatibility-fallback coverage (cec2c7f).
- [~] Task: Harden numerical correctness
    - [x] Add finite-value, reproducibility, rank, and bounded-result property coverage for the native EVSI kernels (a5a1222, 0131208, b9dc7ae).
    - [x] Add committed benchmark workloads for seeded-bootstrap, efficient-linear, and moment-based EVSI (79de00d).
    - [x] Add versioned native-EVSI benchmark regression budgets and CI enforcement (1c5bfe4, 1a168df).
    - [x] Add value-shift and positive-scale metamorphic invariants for all native EVSI kernels (5c2638d).
    - [x] Add multi-seed seeded-bootstrap reference differential coverage (a21a6e6, 797b35f).
    - [x] Add the centered two-parameter moment interaction-order contract fixture (738e634, 797b35f).
    - [x] Add panic-free malformed-input and stable-error proptest coverage (d31fbe2, 1af20c8).
    - [x] Add concurrent thread-safety and repeatability coverage for all native EVSI kernels (c8c96a9).
    - [x] Add bounded generated-input fuzz coverage for stable EVSI kernels (cc735ee).
    - [x] Validate generated-input fuzz coverage across hosted language and platform gates (c7c8937).
    - [x] Scope the scheduled mutation gate to the stable Python numerical facade (5babf7b).
    - [x] Add executable validation for the versioned native benchmark baseline contract (1840ebb).
    - [x] Add differential, metamorphic, fuzz and mutation tests. (validated 2026-07-20; existing evidence: `evsi_differential`, `evsi_metamorphic`, `evsi_fuzz`, `evsi_thread_safety`, and scheduled mutation gate)
    - [x] Establish benchmark baselines, regression budgets, thread-safety checks and promised determinism. (validated 2026-07-20; existing evidence: committed baseline contract, CI regression gate, thread-safety suite, and deterministic-kernel tests)
- [ ] Task: Conductor - Automated Review and Checkpoint 'Stable Numerical Kernel Migration' (Protocol in workflow.md)

## Phase 6: Python Legacy-Core Deprecation and Removal

- [ ] Task: Inventory and classify non-Rust Python code
    - [x] Classify every runtime Python module into an explicit v1 boundary category (e9eac92).
    - [x] Enforce the inventory as an executable unclassified-module failure gate (e9eac92).
    - [x] Identify duplicate kernels, facade code, schemas, I/O, orchestration, CLI, plotting, reporting, wrappers and unrelated extensions (735948a).
    - [x] Produce an executable allowlist for Python code permitted at v1.0 (c818b8c).
    - [x] Enforce Rust authority and compatibility-only Python role for transitional kernels (b4c23b3).
- [ ] Task: Complete the 0.x compatibility bridge using TDD
    - [x] Make transitional Python efficient-linear and moment-based fallbacks observable with deprecation warnings (fb5baae).
    - [x] Red: add tests for deprecation warnings and warning-free native migration routing (5a44349).
    - [x] Define the Rust/PyO3 EVPI runtime contract and adapter forwarding test (fa7ad2b).
    - [x] Route the NumPy-backed public EVPI facade through Rust with a transitional fallback (c724c84).
    - [x] Define the Rust/PyO3 dominance execution contract and adapter forwarding test (cc1172b).
    - [x] Route the public dominance facade through Rust with a transitional fallback (211b41f).
    - [x] Define the Rust/PyO3 CEAF execution contract and adapter forwarding test (0055bf0).
    - [x] Route the public CEAF facade through Rust with a transitional fallback (6e08c67).
    - [x] Green: route stable public APIs to Rust and implement controlled shims. (3882f42; EVPI, EVPPI default, seeded/efficient-linear/moment-based EVSI, dominance, and CEAF routes verified)
    - [x] Refactor: simplify wrappers and publish migration documentation. (3882f42; Astro Rust-core handoff and compatibility boundary)
- [ ] Task: Remove the duplicate Python numerical core using TDD
    - [ ] Red: add tests that fail when retired kernels or fallback paths remain reachable.
    - [ ] Green: remove duplicate implementations, exports and obsolete dependencies.
    - [ ] Refactor: reduce retained Python to a fully typed facade and rerun the suite.
- [ ] Task: Validate the minimal Python distribution
    - [x] Prove stable operation without JAX, GPU, web, widget, distributed or experimental dependencies. (packaging probes validated 2026-07-20)
    - [x] Enforce at least 90 percent coverage for retained production Python. (validated by `tests/test_ci_cd_quality_gates.py` and hosted Coverage Report, 2026-07-20)
- [ ] Task: Conductor - Automated Review and Checkpoint 'Python Legacy-Core Deprecation and Removal' (Protocol in workflow.md)

## Phase 7: Cross-Language Binding Consolidation

- [ ] Task: Define the retained binding matrix
    - [ ] Confirm supported R, Julia, TypeScript, Go and .NET surfaces and registries.
    - [ ] Remove bindings that cannot meet stable contract and maintenance requirements.
- [ ] Task: Convert retained bindings into thin Rust adapters using TDD
    - [ ] Red: add language-specific conformance tests against shared fixtures.
    - [ ] Green: use the C ABI for R, Julia, Go and .NET and WASM or N-API for TypeScript as justified.
    - [ ] Refactor: eliminate independent numerical policy and duplicate conversion logic.
- [ ] Task: Harden binding lifecycle and ABI compatibility
    - [ ] Add build, package, install, unload, memory, concurrency and error-propagation tests.
    - [ ] Prove every retained binding executes Rust across supported version combinations.
- [ ] Task: Conductor - Automated Review and Checkpoint 'Cross-Language Binding Consolidation' (Protocol in workflow.md)

## Phase 8: Supported Extensions and Experimental Isolation

- [ ] Task: Classify all non-core functionality
    - [ ] Evaluate domain modules, web applications, widgets, accelerators, distributed execution, frontier methods and research prototypes.
    - [ ] Record retain, extract, remove or experimental decisions with evidence.
- [ ] Task: Enforce supported-extension boundaries using TDD
    - [ ] Red: add failing dependency-direction, export and packaging boundary tests.
    - [ ] Green: require retained extensions to use Rust execution, shared contracts and independent optional packaging.
    - [ ] Refactor: remove duplicated policy and unnecessary dependencies.
- [ ] Task: Isolate experimental functionality
    - [ ] Move experimental APIs into an explicit namespace or package with maturity metadata and warnings.
    - [ ] Ensure experimental dependencies and failures cannot block the stable core.
- [ ] Task: Remove or extract unsupported code
    - [ ] Preserve migration history where necessary and remove dead dependencies, exports, tests and docs.
- [ ] Task: Conductor - Automated Review and Checkpoint 'Supported Extensions and Experimental Isolation' (Protocol in workflow.md)

## Phase 9: Astro-Only Documentation Consolidation

- [ ] Task: Establish Astro as the sole documentation system using TDD
    - [ ] Red: add failing checks for Sphinx configuration, duplicate generated references and RST-only content.
    - [ ] Green: migrate unique content into Astro and remove Sphinx builds, dependencies and configuration.
    - [ ] Refactor: consolidate navigation, generation and validation tooling.
- [ ] Task: Generate trustworthy API and binding references
    - [ ] Generate references from stable Rust, Python, ABI and binding contracts.
    - [ ] Add drift checks between source contracts and published documentation.
- [ ] Task: Complete and validate v1.0 user documentation
    - [ ] Cover installation, concepts, tutorials, examples, migration, compatibility, security, support and extension maturity.
    - [ ] Validate examples, links, accessibility, spelling and production builds in clean environments.
- [ ] Task: Conductor - Automated Review and Checkpoint 'Astro-Only Documentation Consolidation' (Protocol in workflow.md)

## Phase 10: Quality, Security, Performance and Reproducibility Gates

- [ ] Task: Harden solo-maintainer GitHub governance
    - [ ] Preserve merge autonomy without mandatory external review.
    - [ ] Require reliable CI, attributable changes, protected release workflows and live check-name validation.
- [ ] Task: Maximize continuous-integration coverage using TDD
    - [ ] Red: add failing harness tests for missing Python, Rust, binding, packaging, docs and clean-install gates.
    - [ ] Green: implement blocking lint, type, dead-code, contract, ABI and generated-drift matrices.
    - [ ] Refactor: remove flaky or routinely bypassed checks and minimize redundant work.
- [ ] Task: Establish security release gates
    - [ ] Add dependency, secret, static-analysis, supply-chain, license and artifact scanning.
    - [ ] Generate SBOMs, provenance, checksums and signatures and resolve critical or high findings.
    - [x] Raise the optional deep-learning torch floor to the first patched release and refresh uv.lock (8a054ef).
- [ ] Task: Establish quantitative quality and performance gates
    - [ ] Enforce Python and Rust coverage targets plus property, fuzz, sanitizer, mutation and binding memory-safety checks.
    - [ ] Define benchmark budgets and fail material regressions.
- [ ] Task: Prove reproducible release inputs
    - [ ] Validate tool constraints, lockfiles, generated outputs, fixtures and clean-builder artifact reproduction.
- [ ] Task: Conductor - Automated Review and Checkpoint 'Quality, Security, Performance and Reproducibility Gates' (Protocol in workflow.md)

## Phase 11: Registry Publication and Installability

- [ ] Task: Establish synchronized dynamic release metadata using TDD
    - [ ] Red: add failing tests for version and metadata drift across all artifacts.
    - [ ] Green: implement one authoritative version source and propagation.
    - [ ] Refactor: simplify release metadata generation and rerun package validation.
- [ ] Task: Publish and verify Rust and Python artifacts
    - [ ] Validate TestPyPI and PyPI trusted publishing and provenance.
    - [ ] Publish retained crates on crates.io.
    - [ ] Complete the existing conda-forge publication track and verify indexing.
- [ ] Task: Publish and verify retained bindings
    - [ ] Complete existing R, Julia, TypeScript, Go and .NET registry tracks for CRAN or the approved R target, Julia General, npm, Go proxy and NuGet.
- [ ] Task: Prove registry installability
    - [ ] Test every registry in clean registry-only environments on supported platforms and architectures.
    - [ ] Record linkage, checksums, provenance, smoke results and precise external blockers without overclaiming completion.
- [ ] Task: Conductor - Automated Review and Checkpoint 'Registry Publication and Installability' (Protocol in workflow.md)

## Phase 12: Release Candidate, Migration and Final v1.0 Release

- [ ] Task: Cut and validate a release candidate
    - [ ] Freeze features and regenerate all release evidence.
    - [ ] Run complete CI, security, compatibility, binding, packaging, docs, benchmark and reproducibility suites.
    - [ ] Conduct clean-room installs and representative end-to-end VOI analyses.
- [ ] Task: Complete migration and operational readiness
    - [ ] Finalize 0.x migration guidance, deprecations, removals, support, security, compatibility and rollback procedures.
    - [ ] Prove no stable operation depends on retired Python kernels.
- [ ] Task: Close the programme backlog
    - [ ] Resolve or explicitly defer every v1.0 GitHub issue.
    - [ ] Merge or close every programme pull request and reconcile remaining branches.
    - [ ] Archive completed child tracks and separate post-v1 and externally blocked work.
- [ ] Task: Publish and verify the signed v1.0 release
    - [ ] Publish artifacts, checksums, SBOMs, provenance, signatures, release notes, migration guidance and reproducibility report.
    - [ ] Verify registries, Astro deployment and Rust-backed smoke analyses for every retained ecosystem.
- [ ] Task: Archive the v1.0 programme
    - [ ] Record final evidence against every acceptance criterion and update roadmap, backlog, registry and maturity status.
    - [ ] Archive this umbrella only after all required child tracks and external publication gates are complete.
- [ ] Task: Conductor - Automated Review and Checkpoint 'Release Candidate, Migration and Final v1.0 Release' (Protocol in workflow.md)
