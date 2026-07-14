# CI/CD Quality Gates Documentation

This document describes the strict CI/CD quality gates implemented for the voiage project, including fast PR gates, expensive scheduled gates, and the overall governance framework.

## Overview

The voiage project implements a comprehensive CI/CD quality gate framework to ensure code quality, security, and reliability across all supported languages and platforms. The gates are designed to provide fast feedback for日常 development while maintaining thorough validation for releases.

## Gate Matrix

### Fast PR Gates (Run on every pull request)

These gates provide quick feedback during development and must pass before merging:

- **Lint & Type Check**: Ruff linting, formatting checks, Bandit security scanning, and `ty` static type checking
- **Unit Tests**: pytest unit tests across Python 3.10-3.14 with coverage reporting
- **Integration Tests**: Integration test suite for component interactions
- **E2E Tests**: End-to-end CLI tests for complete user workflows
- **Coverage Report**: Enforces 90% minimum coverage threshold
- **Frontier Contract Validation**: Validates frontier VOI registry and family manifests
- **Version Synchronization**: Ensures version consistency across package manifests
- **Documentation Build**: Builds documentation to catch doc errors
- **Prose Lint**: Vale prose linting for documentation quality

### Expensive Scheduled Gates (Run weekly on Mondays)

These gates are computationally expensive and run on a scheduled basis:

- **Mutation Testing**: Uses mutmut to validate test effectiveness by mutating code
- **Performance Profiling**: Scalene profiler to identify performance bottlenecks
- **Benchmark Regression**: Ensures performance doesn't degrade over time
- **CodeQL Security Analysis**: Deep security analysis using GitHub CodeQL

### Release Gates (Run on releases)

Additional validation for release builds:

- **Full Test Suite**: Complete test execution including all test categories
- **Security Scanning**: Comprehensive security analysis including SBOM generation
- **Binding Package Checks**: Validates all polyglot binding packages
- **Documentation Deployment**: Builds and deploys documentation to GitHub Pages

## Quality Gate Categories

### 1. Linting and Formatting

**Tools**: Ruff, Bandit, Vulture

**Configuration**: 
- Ruff: Comprehensive rule set covering style, complexity, security, and best practices
- Bandit: Security-focused linting with project-specific configuration
- Vulture: Dead code detection (informational)

**Gate**: Must pass in CI and pre-commit hooks

### 2. Type Checking And Typing

**Tool**: `ty` static type checker

**Configuration**: Python 3.10 baseline with selective ignores for compatibility

**Gate**: Must pass in CI, informational warnings allowed

**Typing Requirements**: All public functions must include full type hints. The type checker enforces typing consistency across the codebase.

### 3. Docstrings

**Style**: NumPy-style docstrings for all public functions and classes

**Coverage**:
- All public API functions and methods must have docstrings
- Module-level docstrings required for every module
- Usage examples required for complex methods

**Enforcement**: CI gate checks for missing docstrings on public interfaces

### 4. Coverage

**Tool**: pytest-cov

**Threshold**: 90% minimum coverage

**Gate**: Hard failure if coverage falls below 90%

### 4. Property-Based Testing

**Tool**: Hypothesis

**Coverage**: Mathematical invariants and property-based tests for core VOI methods

**Files**: 
- `test_property_based.py`: General property tests
- `test_property_invariants.py`: Mathematical invariant tests  
- `test_cli_invariants.py`: CLI property tests

**Gate**: Must pass in CI

### 5. Mutation Testing

**Tool**: mutmut

**Configuration**: 90% mutation score threshold

**Schedule**: Weekly on Mondays

**Gate**: Informational, tracks test effectiveness over time

### 6. Security

**Tools**: Bandit, CodeQL, SBOM generation

**Coverage**: 
- Bandit: Python security issues
- CodeQL: Deep semantic analysis
- SBOM: Software Bill of Materials for supply chain security

**Gate**: Must pass in CI (Bandit), weekly (CodeQL)

### 7. Documentation

**Tools**: Astro/Starlight, Vale

**Coverage**: 
- API documentation completeness
- Tutorial and guide quality
- Prose style and consistency

**Gate**: Must build successfully, Vale linting must pass

### 8. Binding Language-Native Gates

Each binding maintains language-native quality gates:

**TypeScript**: npm check, pack dry-run
**Go**: go test, go vet
**Rust**: cargo fmt, cargo clippy, cargo test, cargo doc
**Julia**: Pkg.test()
**.NET**: dotnet build, dotnet test, dotnet pack
**R**: R CMD build, rcmdcheck

**Gate**: Must pass in binding-specific CI

## Python Version Support

**Supported Versions**: 3.10, 3.11, 3.12, 3.13, 3.14

**Testing**: Full test matrix across all supported versions

**Policy**: Minimum version 3.10, maximum version <3.15

## Dependency Policy

### Base Dependencies

Core dependencies required for basic functionality:
- NumPy, SciPy, pandas, xarray (scientific computing)
- JAX, NumPyro (high-performance computing)
- scikit-learn, statsmodels (machine learning)
- matplotlib, seaborn (visualization)
- typer (CLI framework)

### Optional Dependencies

Heavy or experimental dependencies are optional to avoid base-install conflicts:
- **Development Tools**: mutmut, scalene, vulture (testing/profiling)
- **Deep Learning**: PyTorch (optional backend)
- **Documentation**: Astro/Starlight (checking and static-site building)

**Policy**: Heavy dependencies in `dev` extra, never in base install

## Governance

### Coverage Floor

**Policy**: 90% coverage floor is non-negotiable

**Rationale**: Ensures code reliability and refactoring safety

**Enforcement**: Hard failure in CI if coverage falls below threshold

### Public API Compatibility

**Policy**: Preserve public API compatibility unless explicitly approved

**Validation**: API contract tests in `test_core_api_contract_validator.py`

**Process**: Breaking changes require explicit approval and version bump

### External Gates

**Hardware-Dependent**: FPGA, ASIC, TPU, GPU runtime evidence is external-gated

**Cloud-Dependent**: Paid cloud workflows are external-gated

**Registry-Dependent**: External registry submissions are external-gated

**Policy**: Mark external gates explicitly, never silently omit

## Blocked or Skipped Gates

### Currently Blocked Gates

- **Full Mutation Testing**: Too expensive for PR, runs weekly only
- **Deep Profiling**: Too expensive for PR, runs weekly only
- **Hardware Runtime**: FPGA/ASIC/TPU evidence requires physical hardware
- **Cloud Actions**: Paid cloud workflows require quota and billing setup

### Documentation

All blocked or skipped gates are documented with:
- Reason for blocking
- Expected trigger conditions
- Alternative validation approaches

## Verification Commands

### Local Development

```bash
# Run all quality gates
uv run tox

# Run specific gates
uv run pytest tests/ --cov=voiage --cov-report=term-missing
uv run ruff check voiage tests
uv run ruff format voiage tests --check
tox -e typecheck
tox -e lint
tox -e docs
```

### CI Simulation

```bash
# Simulate PR gates
uv run pytest tests/ -m "not integration and not e2e and not benchmark" --cov=voiage
uv run ruff check voiage tests --output-format=github
uv run ruff format voiage tests --check
uv run bandit -r voiage -s B101,B110,B405,B314 -c pyproject.toml

# Simulate scheduled gates
uv run mutmut run --paths-to-mutate voiage/
uv run scalene --cli --outfile profile_results.txt voiage/analysis.py
```

## Continuous Improvement

### Metrics Tracked

- Coverage percentage over time
- Mutation testing effectiveness
- Performance benchmark trends
- Security scan results
- Test execution time

### Review Cadence

- **Weekly**: Review expensive gate results
- **Monthly**: Evaluate gate effectiveness and adjust thresholds
- **Quarterly**: Review and update quality gate policy

## References

- **CI Configuration**: `.github/workflows/ci.yml`
- **Tox Configuration**: `tox.ini`
- **Project Configuration**: `pyproject.toml`
- **Binding CI**: `.github/workflows/bindings-ci.yml`
- **Security**: `.github/workflows/codeql.yml`, `.github/workflows/sbom.yml`
- **Documentation**: `.github/workflows/docs.yml`

## Compliance

This quality gate framework ensures compliance with:

- **Repository-wide 90% coverage gate**
- **Language-native binding quality gates**
- **Security best practices**
- **Documentation standards**
- **Cross-language parity requirements**

## Contact and Support

For questions about quality gates or to suggest improvements, please open an issue or contact the maintainers.
