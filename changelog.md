# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Automated publishing workflow to TestPyPI and GitHub Releases
- Comprehensive release process with version tagging
- Enhanced security scanning with Bandit and Safety
- Improved code quality checks with Ruff
- Better documentation deployment automation
- Enhanced testing infrastructure with tox environments

### Changed
- Updated dependency management with pip-tools
- Improved code coverage reporting with Codecov
- Enhanced CI/CD pipeline with additional checks

### Fixed
- Various code quality issues identified by linters
- Security vulnerabilities detected by safety scans
- Documentation inconsistencies

## [0.2.0] - 2025-11-02

### Added
- Complete implementation of core VOI methods (EVPI, EVPPI, EVSI, ENBS)
- Advanced VOI methods (Structural Uncertainty VOI, Network Meta-Analysis VOI)
- Adaptive Design VOI implementation
- Portfolio Optimization for research prioritization
- Value of Heterogeneity analysis
- Web API interface for remote access
- Command-line interface for automation
- Docker deployment support
- GPU acceleration capabilities
- Memory optimization for large datasets
- Streaming data processing
- Result caching mechanisms
- Parallel processing support
- Interactive plotting widgets
- Comprehensive test suite with pytest
- Property-based testing with Hypothesis
- Performance benchmarks
- Detailed documentation and examples
- Type hints throughout the codebase
- Pre-commit hooks for code quality
- CI/CD pipeline with GitHub Actions
- Security scanning and vulnerability detection
- Code coverage reporting
- Release automation to PyPI and TestPyPI

### Changed
- Restructured project for better modularity
- Improved performance with optimized algorithms
- Enhanced usability with fluent API
- Better error handling and validation
- Updated dependencies to latest stable versions

### Fixed
- Various bugs in VOI calculations
- Performance issues with large datasets
- Compatibility problems with different Python versions
- Documentation errors and inconsistencies

## [0.1.0] - 2025-09-15

### Added
- Initial release with basic VOI functionality
- Core data structures (ValueArray, ParameterSet)
- Basic VOI methods (EVPI, EVPPI)
- Simple CLI interface
- Basic documentation
- Initial test suite

[Unreleased]: https://github.com/doughnut/voiage/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/doughnut/voiage/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/doughnut/voiage/releases/tag/v0.1.0