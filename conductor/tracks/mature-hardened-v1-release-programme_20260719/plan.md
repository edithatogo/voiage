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

## Phase 5: Stable Numerical Kernel Migration [checkpoint: 5c9006a]

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
- [x] Task: Migrate supported EVSI methods using TDD (83bc772)
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
    - [x] Scope the C15 exact Python 3.14 runner assertion to its approved interpreter lane (8fac8b6).
    - [x] Align tox minimum and maximum dependency bounds with the declared runtime contract (97a1ce7).
    - [x] Reconcile the Rust migration matrix with the implemented two-loop EVSI contract (ae74b21).
    - [x] Extend native ownership to the supported Python model-callback estimator family after explicit contracts were frozen; retain two-loop, random-forest efficient, adaptive, and NMA paths in Python where their contracts remain intentionally outside the stable native kernel boundary. (83bc772)
        - [x] Define and implement the versioned regression prediction envelope: Python owns callback simulation; Rust owns finite OLS fit/predict aggregation. (83bc772)
        - [x] Review fix: document errors, check count conversion, and satisfy strict Rust clippy for regression elimination. (3d1fd4f)
        - [x] Route the public moment-based estimator through the versioned Rust kernel with a rank-deficiency compatibility fallback (ac1b8e8).
        - [x] Add public moment-based envelope and compatibility-fallback coverage (cec2c7f).
- [x] Task: Harden numerical correctness (validated 2026-07-21 by full `CI=true uv run tox -q` and native Rust regression/clippy gates)
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
- [x] Task: Conductor - Automated Review and Checkpoint 'Stable Numerical Kernel Migration' (Protocol in workflow.md) (5c9006a)

## Phase 6: Python Legacy-Core Deprecation and Removal [checkpoint: c07a729]

- [x] Task: Inventory and classify non-Rust Python code (validated 2026-07-21; `python scripts/validate_python_runtime_inventory.py .`, `python scripts/validate_v1_programme.py --repo-root .`)
    - [x] Classify every runtime Python module into an explicit v1 boundary category (e9eac92).
    - [x] Enforce the inventory as an executable unclassified-module failure gate (e9eac92).
    - [x] Identify duplicate kernels, facade code, schemas, I/O, orchestration, CLI, plotting, reporting, wrappers and unrelated extensions (735948a).
    - [x] Produce an executable allowlist for Python code permitted at v1.0 (c818b8c).
    - [x] Enforce Rust authority and compatibility-only Python role for transitional kernels (b4c23b3).
- [x] Task: Complete the 0.x compatibility bridge using TDD (validated 2026-07-21; existing route, warning, parity, and compatibility evidence)
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
- [x] Task: Remove the duplicate Python numerical core using TDD (2a47eab; retired stable duplicate kernels and removed rank-only EVSI shims; remaining Python methods are subject to Phase 8 extension classification)
    - [x] Red: add tests that fail when retired kernels or fallback paths remain reachable. (working tree test update)
    - [x] Green: remove duplicate EVPI, CEAF, and dominance fallback implementations and obsolete imports. (working tree implementation)
    - [x] Refactor: reduce retained public paths to validated Rust-backed facades and rerun focused tests. (focused validation passed)
    - [x] Remove EVSI efficient-linear, moment-based, and regression numerical fallbacks; missing-native paths fail closed and rank-deficient designs are handled natively. (validated by native and Python parity tests)
        - [x] Red: add shared rank-deficient fixtures and differential tests for efficient-linear, moment-based, and callback-regression kernels.
        - [x] Green: implement the specified native rank-aware solver and expose the versioned contract through PyO3.
        - [x] Refactor: remove the rank-only Python compatibility shims and rerun native/package validation.
    - [x] Review/validation: full `CI=true uv run tox -q` passed across lint, harness, typecheck, Astro, frontier/version, Python 3.12-3.14, min/max dependency, and coverage environments (91.11% coverage, 2026-07-21).
    - [x] Review fix: apply Ruff formatting to the numerical-core retirement slice. (7b219d4)
- [x] Task: Validate the minimal Python distribution (full tox matrix and runtime inventory validation passed 2026-07-21)
    - [x] Prove stable operation without JAX, GPU, web, widget, distributed or experimental dependencies. (packaging probes validated 2026-07-20)
    - [x] Enforce at least 90 percent coverage for retained production Python. (validated by `tests/test_ci_cd_quality_gates.py` and hosted Coverage Report, 2026-07-20)
- [x] Task: Conductor - Automated Review and Checkpoint 'Python Legacy-Core Deprecation and Removal' (c07a729; review and full validation passed 2026-07-21)

## Phase 7: Cross-Language Binding Consolidation

- [~] Task: Define the retained binding matrix (799fd23; machine-readable matrix and drift tests passed 2026-07-21)
    - [~] Confirm supported Rust, Python/Mojo, R and Julia surfaces, adapters, registries and external gates. (Rust/Python/R/Julia are repository-owned; Mojo is recorded as an external boundary because `command -v mojo` returns no executable and the repository contains no Mojo binding or release workflow)
    - [x] Confirm every retained surface has a repository path, shared fixture root, CI workflow and version tag contract.
    - [x] Remove bindings that cannot meet stable contract and maintenance requirements. (Go, TypeScript and .NET implementations, duplicate standalone Rust binding, WASM crate, workflows and active publication references removed; retained matrix is Rust, Python, R and Julia, with Mojo explicitly external)
- [~] Task: Convert retained bindings into thin Rust adapters using TDD
    - [x] Red/green: add and pass a C ABI EVPI conformance test against the Rust numerical kernel (f7cc0b7).
    - [x] Shared EVPI conformance: Python, R, Julia and Rust surfaces pass the canonical/reference fixture or ABI smoke gate (3f22b1a, 7647b81, and existing Rust evidence).
    - [~] Green: use the stable Rust C ABI for R and Julia, and the native Rust/PyO3 boundary for Python; Mojo remains an explicit upstream integration boundary.
    - [x] Refactor: eliminate independent EVPI numerical policy and duplicate conversion logic across all seven retained surfaces (2a05984, 3f22b1a, 7647b81, d475e64).
    - [x] Explicitly isolate advanced binding methods under the Phase 8 extension policy until Rust-backed contracts exist (`specs/v1/extension-policy.json`).
- [~] Task: Harden binding lifecycle and ABI compatibility
    - [x] Add executable matrix drift coverage for build, test, package and Rust ABI lifecycle/error gates (working tree `tests/test_binding_lifecycle_contract.py`).
    - [~] Add binding-specific install, unload, memory, concurrency and error-propagation tests. (Rust ABI and contract coverage complete; R Rust ABI smoke and dependency-boundary checks pass in c8cec5d; Julia passes when pointed at the built FFI library; Python/PyO3 and hosted runtime evidence remain)
    - [x] Prove the non-PyO3 Rust workspace and ABI lifecycle gates pass with all features; PyO3 full-workspace execution remains runner-bound because the local environment lacks `libpython3.13.dylib`.
    - [~] Prove every retained binding executes Rust across supported version combinations. (R Rust ABI smoke passes; Julia Pkg.test passes with the built FFI library; Rust non-PyO3 workspace/ABI tests pass; local PyO3 requires libpython3.13.dylib; Python/Mojo hosted evidence remains)
- [ ] Task: Conductor - Automated Review and Checkpoint 'Cross-Language Binding Consolidation' (Protocol in workflow.md)

## Phase 8: Supported Extensions and Experimental Isolation

- [~] Task: Classify all non-core functionality
    - [x] Classify every `voiage/methods/` module as stable Rust facade, optional extension, or experimental using `specs/v1/extension-policy.json` and executable coverage (`tests/test_extension_policy.py`).
    - [x] Classify every remaining Python runtime file across facade, assurance, optional-extension and experimental surfaces using `specs/v1/extension-surface-policy.json` with exactly-one-disposition coverage.
    - [~] Evaluate domain modules, web applications, widgets, accelerators, distributed execution, frontier methods and research prototypes for final retain/extract/remove decisions. (runtime disposition policy is recorded; final package-level decisions remain)
    - [~] Record retain, extract, remove or experimental decisions with evidence. (machine-readable policies and executable coverage added)
- [~] Task: Enforce supported-extension boundaries using TDD
    - [x] Red: add failing dependency-direction, export and packaging boundary tests. (lazy import/export isolation and exactly-one-disposition tests)
    - [~] Green: require retained extensions to use Rust execution, shared contracts and independent optional packaging. (normative stable/provisional __all__, stable import/export, base-dependency gates, and the machine-readable single-wheel optional-extra boundary pass; Rust execution and independent packaging remain; e9144ef)
    - [ ] Refactor: remove duplicated policy and unnecessary dependencies.
- [x] Task: Isolate experimental functionality (1e1eca4, e770dfe)
    - [x] Move experimental APIs into an explicit namespace or package with maturity metadata and warnings. (voiage.experimental; experimental functions resolve lazily)
    - [x] Ensure experimental dependencies and failures cannot block the stable core. (clean-import and lazy-namespace tests)
- [ ] Task: Remove or extract unsupported code
    - [ ] Preserve migration history where necessary and remove dead dependencies, exports, tests and docs.
- [ ] Task: Conductor - Automated Review and Checkpoint 'Supported Extensions and Experimental Isolation' (Protocol in workflow.md)

## Phase 9: Astro-Only Documentation Consolidation

- [~] Task: Establish Astro as the sole documentation system using TDD
    - [x] Red: add failing checks for Sphinx configuration, duplicate generated references and RST-only content. (existing repository harness and Astro contract tests)
    - [~] Green: migrate unique content into Astro and remove Sphinx builds, dependencies and configuration. (active public links migrated; legacy source migration remains)
    - [ ] Refactor: consolidate navigation, generation and validation tooling.
- [x] Task: Generate trustworthy API and binding references (0554bdc, c48233d; stable API, C ABI, and binding references published)
    - [x] Generate references from stable Rust, Python, ABI and binding contracts. (normative stable API, C ABI manifests, and binding matrix references published)
    - [x] Add drift checks between source contracts and published documentation. (tests/test_binding_reference_docs.py, tests/test_api_reference_docs.py)
- [ ] Task: Complete and validate v1.0 user documentation
    - [x] Cover installation, concepts, tutorials, examples, migration, compatibility, security, support and extension maturity. (2883d32; Astro readiness guide and topic/link contract tests)
    - [~] Validate examples, links, accessibility, spelling and production builds in clean environments. (51a5832; repository-owned Astro route/GitHub-backed link validation, Astro check, and 68-page production build pass; clean temporary execution passed for the complete non-optional notebook matrix, including getting_started, EVPI, EVPPI, EVSI, adaptive, advanced, calibration, fluent, interactive, engineering, environmental, financial, observational, portfolio, metamodeling, NMA, structural, benchmarking, voiage validation, and visualization; optional JAX/widget/Colab examples remain dependency-gated; Vale reports zero alerts across 1,086 Markdown/MDX files and generated pages have no missing image alt text; clean-builder evidence remains)
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
- [~] Task: Prove reproducible release inputs
    - [~] Validate tool constraints, lockfiles, generated outputs, fixtures and clean-builder artifact reproduction. (tool/version and source-identity binding pass; `f554d9f` reconciles the Maturin-rewritten sdist workspace with a generated lockfile before `--locked` extraction builds; `ddf6bcb` packages normative JSON fixtures; local extracted-sdist Rust all-target tests pass through non-PyO3 targets, and release `cargo build --locked` plus Maturin wheel build pass; local PyO3 execution remains blocked by missing libpython3.13.dylib, while cross-runner artifact and hosted clean-builder evidence remain)
- [ ] Task: Conductor - Automated Review and Checkpoint 'Quality, Security, Performance and Reproducibility Gates' (Protocol in workflow.md)

## Phase 11: Registry Publication and Installability

- [x] Task: Establish synchronized dynamic release metadata using TDD (b51fb0f, 0d834d4, 58d5ddd; version-sync tests and CI lane pass 2026-07-21)
    - [x] Red: add failing tests for version and metadata drift across all artifacts.
    - [x] Green: implement one authoritative Rust workspace version source and propagation.
    - [x] Refactor: simplify release metadata generation and rerun package validation.
- [ ] Task: Publish and verify Rust and Python artifacts
    - [ ] Validate TestPyPI and PyPI trusted publishing and provenance.
    - [ ] Publish retained crates on crates.io.
    - [ ] Complete the existing conda-forge publication track and verify indexing.
- [ ] Task: Publish and verify retained bindings
    - [ ] Complete existing R and Julia registry tracks for CRAN or the approved R target and Julia General; retain Python/Mojo publication as its own release boundary.
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
