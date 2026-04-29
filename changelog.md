# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- Restored the CI and tox lint path for Bandit by adding it to the dev tool
  set and wiring the security check into the local lint env, and committed the
  Vale configuration files that the prose job expects at the repository root.
- Relaxed two CLI test assertions so they accept the rich help and error
  rendering used by Typer across local and CI environments.
- Fixed the coverage-report workflow by installing tox through the locked dev
  environment and relaxed the main CLI help assertion so it no longer depends
  on a specific rich-rendered option line in GitHub Actions.
- Marked the CLI E2E suite as `e2e` in pytest collection and removed the
  coverage gate from the integration and E2E CI jobs so those jobs exercise
  their own scope instead of failing on the global unit-test threshold.
- Split the coverage gate into its own CI job so the unit matrix can collect
  coverage for Codecov without enforcing the repo-wide threshold twice.

### Added
- Aligned the GitHub Actions Ruff gate with the repo's tox lint scope so the
  workflow checks `voiage` and `tests` instead of legacy root-level scratch
  scripts that are outside the supported lint surface, and corrected the Vale
  install target to the command package path used by the CI job.
- Added the planned threshold, tipping-point, and robust VOI contract
  scaffold, including versioned input/output schemas, illustrative example
  payloads, and a frontier doc note that frames the surface as
  sample-by-strategy-by-threshold-profile analysis.
- Added the planned model-validation VOI contract scaffold, including versioned
  input/output schemas, illustrative example payloads, and a frontier doc note
  that frames the surface as sample-by-strategy-by-validation-profile analysis.
- Added the planned preference heterogeneity and individualized care contract
  scaffold, including versioned input/output schemas, illustrative example
  payloads, and a frontier doc note that frames the surface as
  sample-by-strategy-by-preference-profile analysis.
- Added a SOTA VOI frontier track and research note, led by Value of
  Perspective and including planned distributional/equity VOI,
  implementation-adjusted VOI, preference-information VOI, validation VOI,
  threshold/tipping-point VOI, robust VOI, dynamic real-options VOI, and
  adjacent causal, data-quality, computational, and elicitation VOI extensions.
- Added an experimental Value of Perspective API, high-level
  `DecisionAnalysis` wrapper, CLI command, regret-matrix plot helper, and v1
  contract scaffold for comparing decisions across multiple perspectives.
- Added deterministic fixture manifests and exact input/output payloads for
  Value of Perspective, distributional/equity VOI, and implementation-adjusted
  VOI.
- Added a repository-level frontier fixture manifest validation test so the
  experimental contract directories are checked against real artifacts.
- Added a top-level frontier fixture registry manifest so the committed
  experimental contracts can be discovered from one place.
- Added a registry schema for the frontier fixture discovery layer and a test
  that validates the registry schema alongside the fixture manifests.
- Added a reusable `scripts/validate_frontier_contract.py` entrypoint for the
  frontier fixture registry and family manifests.
- Updated the top-level README and frontier research note to describe the
  registry-backed frontier contract layer.
- Wired the frontier contract validator into tox and CI so registry-backed
  frontier fixtures are checked in the normal quality gates.
- Updated the migration guide so the frontier table and roadmap context reflect
  the registry-backed experimental contracts.
- Added frontier examples to the canonical advanced VOI guide for Value of
  Perspective and distributional/equity VOI.
- Added explicit experimental API warnings to the frontier research note and
  advanced VOI guide.
- Added experimental distributional/equity VOI and implementation-adjusted VOI
  APIs, together with `DecisionAnalysis` wrappers, curated exports, and
  regression tests.
- Added deterministic experimental frontier contract files for distributional
  and implementation-adjusted VOI, including versioned schemas, README
  scaffolds, and example result payloads.
- Added shared CHEERS-VOI reporting payload helpers and attached them to the
  experimental Value of Perspective, distributional/equity VOI, and
  implementation-adjusted VOI results.
- Added the same CHEERS-VOI reporting payload helper to Value of Heterogeneity
  so the distributional frontier retains a consistent reporting baseline.
- Added the same CHEERS-VOI reporting payload helper to CEAF and dominance
  outputs so the core cost-effectiveness summary layer also carries the shared
  reporting baseline.
- Added shared CHEERS-VOI reporting payloads to the core scalar CLI outputs for
  EVPI, EVPPI, EVSI, and ENBS.
- Updated the tox coverage job to run the repository's verified pytest
  coverage command directly, so tox and the local suite now use the same
  verification path.
- Added Vale prose linting for Markdown and reStructuredText docs, with the CI
  job pointed at the real documentation paths and contributor guidance for the
  local command.
- Added ISPOR VOI and CHEERS-VOI references to the SOTA frontier track so the
  remaining frontier methods are anchored to method guidance and reporting
  standards.
- Added the HEOR module naming brainstorm track for `calibrate`, `evidence`, `process`, `report`, `registry`, `workflow`, `quality`, `engines`, and `heoml`, with PM4Py kept as an ecosystem-only process-mining capability.
- Linked the roadmap phases to their corresponding Conductor tracks so the
  remaining missing features are visible in one place.
- Refreshed the migration guide feature comparison so it reflects the current
  implemented method set, cross-language scaffolds, and roadmap context.
- Refreshed the top-level README feature matrix and roadmap summary to match
  the current implementation state and the active Phase 5/6 roadmap.
- Added the ecosystem incubation contract outline under `specs/ecosystem/`,
  including the HEOML `voiage` extension scaffold and planned fixture
  families for `lifecourse`, `innovate`, and `mars`.
- Added a compact deterministic `lifecourse` compatibility fixture bundle
  with net-benefit, parameter-sample, EVPI, and EVPPI reference payloads under
  `specs/integrations/lifecourse/v1/fixtures/normative/`.
- Added the `specs/integrations/lifecourse/v1/` scaffold with examples,
  fixtures, and schema overlay placeholders for the future `lifecourse`
  compatibility contract.
- Added a planned `lifecourse` integration contract track and strategy document
  for consuming `lifecourse` PSA outputs through stable VOI artifacts rather
  than package internals.
- Documented the `lifecourse` integration profile as portable-artifact based,
  with pickle excluded from the shared compatibility contract.
- Identified HEOML as the candidate shared health-economic interchange profile
  for the future `lifecourse` and `voiage` handoff.
- Added an ecosystem-module incubation track and strategy for positioning
  `voiage` alongside `lifecourse`, `innovate`, `mars`, HEOML, and future
  sibling modules through optional artifact-first HEOR contracts.
- Expanded the core public API docstrings for EVPI, EVPPI, EVSI, ENBS, CEAF, CEAC, and the main schema/analysis types with NumPy-style sections and examples.
- Expanded the dominance, heterogeneity, portfolio, and dominance-plot public docs with full parameter, return, and notes sections.
- Expanded the adaptive-trial and structural-VOI public docs with full NumPy-style parameter and return sections.
- Expanded the calibration, observational, NMA, and VOI-curve public docs with fuller API-style sections and examples.
- Expanded the `DecisionAnalysis` wrapper docs so the top-level analysis methods now describe parameters, returns, and formulas in the public API surface.
- Expanded the network-meta-analysis implementation docs so the NMA entrypoints and helpers now have fuller NumPy-style sections.
- Documented the polyglot binding release matrix with tooling parity, registry/versioning expectations, and the logging policy for CLI/library output.
- Added a CLI-wide `--quiet` option that suppresses confirmation chatter while keeping the result output intact.
- Added a CLI-wide `--verbose` option that emits debug diagnostics to stderr without changing stdout result formatting.
- Added `voiage generate-config` so the CLI can emit JSON templates for common analysis inputs, starting with EVSI.
- Added working examples to the CLI help output for the result, plotting, and config-generation commands.
- Added a dedicated CLI e2e smoke-test file that exercises the full command surface through `CliRunner`.
- Added a CLI-wide `--format` option for text, JSON, and CSV output, with formatter coverage across the main result commands and plot summaries.
- Added a `--parameters-of-interest` alias for `voiage calculate-structural-evppi`, plus CLI wrappers for the existing CEAC, CEAF, VOI-curve, and dominance plotting APIs.
- Added `voiage plot-ceac`, `voiage plot-ceaf`, `voiage plot-voi-curves`, and `voiage plot-dominance` CLI commands for the existing plotting APIs.
- Added `voiage calculate-adaptive-evsi`, `voiage calculate-portfolio-voi`, and `voiage calculate-sequential-voi` as thin CLI wrappers over the adaptive, portfolio, and sequential VOI methods.
- Added `voiage calculate-enbs` with direct or file-backed EVSI input parsing and optional result-file output.
- Added curated top-level `voiage` package exports for `DecisionAnalysis`, schema types, and core methods, plus a `__version__` attribute from package metadata.
- Added `DecisionAnalysis` wrappers for CEAF, dominance, Value of Heterogeneity, and portfolio VOI so the new methods are reachable through the high-level analysis surface.
- Added exact dynamic-programming portfolio VOI optimization with budget-constrained selection, dependency-group value discounting, and regression coverage against greedy misses.
- Added Value of Heterogeneity calculation for subgroup-specific decisions, numeric subgroup binning, and subgroup VOH plotting.
- Added strong/extended dominance analysis, ICER calculation helpers, cost-effectiveness frontier extraction, and a dominance plot helper.
- Added Cost-Effectiveness Acceptability Frontier (CEAF) calculation and plotting helpers with uncertainty bands and export coverage.
- Added efficient PSA-regression and moment-based EVSI approximation methods, including a `voiage calculate-evsi` CLI command with method selection.
- Added a built-in observational study VOI modeler for explicit net-benefit or cost/effect PSA samples, including sample-size and bias-strength uncertainty adjustment.
- Added targeted branch coverage for schema, backend, structural VOI, config, financial-risk, healthcare, memory-optimization, and network meta-analysis paths, and enabled branch-aware coverage gating at 90%.
- Added initial TypeScript, Go, Rust, Julia, and .NET 11 binding package scaffolds, plus GitHub Actions CI/release workflows for npm, Go modules, crates.io, Julia package validation, NuGet, and the existing R package.
- Archived and marked complete the Python cleanup against spec track.
- Added regression coverage confirming chunked EVPPI evaluation matches the unchunked path.
- Added regression coverage for the TreeAge invalid XML fail-soft path and its warning emission.
- Added regression coverage for the curated `__all__` exports on `voiage.backends` and `voiage.methods`.
- Added regression coverage for the full versioned core API schema/example contract matrix.
- Removed an invalid EVPPI regression that assumed raw dictionary parameter samples were supported.
- Added deterministic provenance metadata validation for normative core API fixtures in the versioned manifest.
- Added normative and illustrative CEAC conformance fixtures under `specs/core-api/fixtures/v1/` and registered them in the versioned manifest.
- Added a stable core API CEAC result contract with a versioned schema, example payload, and validator coverage.
- Synchronized the core API v1 README indexes with the CEAC result schema and example contract.
- Synchronized the core API v1 schema README index with the full contract matrix.
- Seeded the illustrative EVSI conformance fixture under `specs/core-api/fixtures/v1/illustrative/` and registered it in the versioned manifest.
- Seeded the normative and illustrative ENBS conformance fixtures under `specs/core-api/fixtures/v1/` and registered them in the versioned manifest.
- Seeded the illustrative EVPPI conformance fixture under `specs/core-api/fixtures/v1/illustrative/` and registered it in the versioned manifest.
- Seeded the first illustrative EVPI conformance fixture under `specs/core-api/fixtures/v1/illustrative/` and registered it in the versioned manifest.
- Seeded the first normative EVPI conformance fixture under `specs/core-api/fixtures/v1/normative/` and registered it in the versioned manifest.
- Seeded the normative EVPPI conformance fixture under `specs/core-api/fixtures/v1/normative/` and registered it in the versioned manifest.
- Seeded the normative EVSI conformance fixture under `specs/core-api/fixtures/v1/normative/` and registered it in the versioned manifest.
- Added the versioned core API fixture scaffold under `specs/core-api/fixtures/v1/`, including the initial manifest contract and validator coverage for the normative and illustrative subtrees.

### Changed
- Made the calibration VOI modeler optional by defaulting to the built-in calibration modeler when no custom modeler is supplied.
- Replaced the JAX two-loop EVSI placeholder/fallback path with a real JAX-assisted posterior update and resampling implementation.
- Replaced the sequential VOI step-level EVPI variance heuristic with the standard `E[max NB] - max(E[NB])` calculation for explicit strategy payoff samples.
- Replaced stale legacy type-checker references with `ty` across the active tooling, contributor docs, and Conductor infrastructure plan.
- Consolidated security linting into Ruff's selected `S` rules and removed the standalone Bandit gate from active tooling.
- Stabilized the curated package exports for `voiage.core`, `voiage.methods`, and `voiage.plot`, and added regression coverage for package-level imports.
- Stabilized the top-level `voiage` package facade so importing `voiage` now exposes the main submodules directly, with regression coverage for the export surface.
- Added focused regression coverage for NMA CLI config validation and error branches.
- Added focused regression coverage for NICE HTA scoring and decision thresholds.
- Added central pytest test categorization in `tests/conftest.py`, automatically marking collected tests as `unit`, `integration`, or `benchmark` based on file naming conventions.
- Declared the `unit` pytest marker in project configuration to support marker-based collection and selection.
- Updated the legacy pytest section in `setup.cfg` to `tool:pytest` so modern pytest versions invoked via `tox` can parse repository configuration.
- Declared missing runtime dependencies for `psutil` and `typing_extensions` so tox-installed environments can import the shipped package successfully.
- Added `pytest-benchmark` to the tox test environment so benchmark-marked tests have their required fixture during suite execution.
- Updated `ValueArray.values` to return the underlying `xarray.DataArray` and added `numpy_values`, copy, subset, and equality helpers for schema-level interoperability.
- Updated decision-analysis and downstream method code paths to use raw NumPy access where numerical kernels require ndarray semantics.
- Fixed structural JAX EVPI aggregation, GPU backend detection/mockability, and memory-budget handling so the shipped test environment runs cleanly under `tox`.
- Added regression coverage for deterministic GPU helper paths in the backend layer, including GPU detection, memory-info reporting, batch flushing, and advanced-backend delegation.
- Normalized health-economic trial outputs to Python floats and added lightweight optional-dependency fallbacks for GAM and BART metamodels when the heavy native stacks are unavailable.
- Removed the temporary GPU-test xfail and hardened metamodel diagnostics and cross-validation edge cases so the `tox` suite completes without warning noise.
- Hardened the core analysis and clinical-trial kernels for JAX tracing, removed a NumPy alias warning, and kept the full `tox` suite green without warning output.
- Reorganized the Conductor track layout into spec-first tracks to support the planned core API, fixtures, and future language bindings.
- Clarified the EVPI/EVPPI validation notebook and marked the benchmark notebook TODO complete.
- Expanded regression coverage across deterministic public modules, including backend helpers, ecosystem-import/export paths, health-economics utilities, CLI flows, schema validation, and the fluent API.

### Added
- **Structural Uncertainty VOI Methods**:
  - `structural_evpi()`: Calculate Expected Value of Perfect Information for Model Structure
  - `structural_evppi()`: Calculate Expected Value of Partial Perfect Information for Model Structure
  - `structural_evpi_jit()`: JAX-accelerated version with JIT compilation
  - `structural_evppi_jit()`: JAX-accelerated version with JIT compilation
  - CLI commands: `voiage calculate-structural-evpi` and `voiage calculate-structural-evppi`
  - JSON config file support for defining multiple model structures

- **Network Meta-Analysis VOI Methods**:
  - `NetworkMetaAnalysisData`: Data structure for NMA inputs with validation
  - `calculate_nma_evpi()`: Calculate EVPI for Network Meta-Analysis
  - `calculate_nma_evppi()`: Calculate EVPPI for Network Meta-Analysis
  - CLI command: `voiage calculate-nma-voi`
  - Support for willingness-to-pay thresholds
  - Dictionary-to-NMA data conversion for ease of use

### Changed
- Migrated from pip/tox to uv for 10-100x faster dependency resolution
- Expanded Ruff configuration with comprehensive rule sets
- Standardized static type checking on ty
- Enhanced pre-commit hooks with ty, commitlint, shellcheck, vulture
- Added integration and E2E test structure with pytest markers
- Modernized CI/CD with uv caching, CodeQL, benchmark tracking
- Added Renovate configuration for automated dependency updates

## [0.2.0] - 2025-08-02

### Changed
- **Major Refactoring of Core API and Data Structures**:
  - Introduced a new object-oriented interface (`voiage.analysis.DecisionAnalysis`) as the primary entry point for VOI calculations (EVPI, EVPPI). This encapsulates the state of a decision problem.
  - Implemented a new computational backend system (`voiage.backends`) with NumPy as the default backend. This will allow for future extensions to support other backends like JAX or PyTorch.
  - Centralized and standardized all core data structures (e.g., `ValueArray`, `ParameterSet`, `TrialDesign`) in `voiage.schema.py`.
  - Provided backward-compatible wrappers and aliases in `voiage.core.data_structures` to ensure that existing code continues to work.
  - The functional API in `voiage.methods.basic` now uses the new `DecisionAnalysis` class, providing a consistent implementation.

## [Unreleased]

### Added
- Initial project structure and placeholder files.
- Core dependencies in `pyproject.toml`.
- Pre-commit hooks for `black`, `flake8`, and `ty`.
- `NetBenefitArray`, `PSASample`, `TrialArm`, `TrialDesign`, `PortfolioStudy`, `PortfolioSpec`, and `DynamicSpec` data structures.
- `evpi()`, `evppi()`, `evsi()`, and `enbs()` methods.
- Unit tests for `evpi()`, `evppi()`, `evsi()`, and `enbs()`.
- Initial documentation in the `docs/` directory.
- Synchronized the core API README indexes with the current schema, example, fixture, and validator layout.
- Added regression coverage for callable import resolution and schema round-trips.

### Changed
- Consolidated changes from `pleides/update`.

### Fixed
- N/A
