# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

### Changed
- Consolidated changes from `pleides/update`.

### Fixed
- N/A
