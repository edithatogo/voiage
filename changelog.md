# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Added regression coverage for the full versioned core API schema/example contract matrix.
- Added regression coverage for EVPPI handling of raw dictionary parameter samples.
- Added deterministic provenance metadata validation for normative core API fixtures in the versioned manifest.
- Added normative and illustrative CEAC conformance fixtures under `specs/core-api/fixtures/v1/` and registered them in the versioned manifest.
- Added a stable core API CEAC result contract with a versioned schema, example payload, and validator coverage.
- Synchronized the core API v1 README indexes with the CEAC result schema and example contract.
- Seeded the illustrative EVSI conformance fixture under `specs/core-api/fixtures/v1/illustrative/` and registered it in the versioned manifest.
- Seeded the normative and illustrative ENBS conformance fixtures under `specs/core-api/fixtures/v1/` and registered them in the versioned manifest.
- Seeded the illustrative EVPPI conformance fixture under `specs/core-api/fixtures/v1/illustrative/` and registered it in the versioned manifest.
- Seeded the first illustrative EVPI conformance fixture under `specs/core-api/fixtures/v1/illustrative/` and registered it in the versioned manifest.
- Seeded the first normative EVPI conformance fixture under `specs/core-api/fixtures/v1/normative/` and registered it in the versioned manifest.
- Seeded the normative EVPPI conformance fixture under `specs/core-api/fixtures/v1/normative/` and registered it in the versioned manifest.
- Seeded the normative EVSI conformance fixture under `specs/core-api/fixtures/v1/normative/` and registered it in the versioned manifest.
- Added the versioned core API fixture scaffold under `specs/core-api/fixtures/v1/`, including the initial manifest contract and validator coverage for the normative and illustrative subtrees.

### Changed
- Stabilized the curated package exports for `voiage.core`, `voiage.methods`, and `voiage.plot`, and added regression coverage for package-level imports.
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
- Added ty type checker alongside MyPy
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
- Pre-commit hooks for `black`, `flake8`, and `mypy`.
- `NetBenefitArray`, `PSASample`, `TrialArm`, `TrialDesign`, `PortfolioStudy`, `PortfolioSpec`, and `DynamicSpec` data structures.
- `evpi()`, `evppi()`, `evsi()`, and `enbs()` methods.
- Unit tests for `evpi()`, `evppi()`, `evsi()`, and `enbs()`.
- Initial documentation in the `docs/` directory.
- Synchronized the core API README indexes with the current schema, example, fixture, and validator layout.

### Changed
- Consolidated changes from `pleides/update`.

### Fixed
- N/A
