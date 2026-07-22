# voiage - Product Guidelines

## Code Style & Conventions

### Naming Conventions
- **Modules**: lowercase with underscores (e.g., `health_economics.py`)
- **Classes**: PascalCase (e.g., `VOIAnalyzer`, `JAXBackend`)
- **Functions/Methods**: snake_case (e.g., `calculate_evpi`, `run_sampling`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `DEFAULT_POPULATION`)
- **Private Members**: prefix with underscore (e.g., `_internal_method`)

### Documentation Standards
- **Docstrings**: NumPy-style docstrings for all public functions
- **Type Hints**: Required for all function signatures
- **Module Docstrings**: Each module must have a top-level docstring explaining its purpose
- **Examples**: Include usage examples in docstrings for complex methods

### Error Handling
- **Custom Exceptions**: Use domain-specific exceptions from `voiage.exceptions`
- **Graceful Degradation**: Provide meaningful error messages with actionable guidance
- **Input Validation**: Validate all inputs at method boundaries
- **Logging**: Use Python's logging module for diagnostic information

## API Design Principles

### Consistency
- All VOI methods follow the same signature pattern: `method(inputs, outputs, **kwargs)`
- Return versioned Rust result envelopes through each retained binding; Python
  adapters may expose NumPy or xarray views without owning numerical policy.
- Provide both functional and object-oriented interfaces

### CLI Guidelines
- Use Typer for CLI implementation
- All core methods must have CLI commands
- Support CSV input/output for batch processing
- Provide `--help` with comprehensive examples

### Runtime Authority and Binding Boundaries
- Rust is the sole stable numerical execution authority.
- Python/PyO3, R and Julia are thin bindings over the Rust core; Mojo remains
  an explicitly tracked upstream interop boundary until its toolchain and
  distribution contract are available.
- Python may retain schemas, I/O, orchestration, CLI, plotting and reporting,
  but must not silently reintroduce stable numerical policy or fallbacks.
- Accelerator, distributed and research implementations remain optional or
  experimental and cannot block or bloat the stable installation.

## Testing Standards

### Coverage Requirements
- Target: >90% code coverage
- Critical methods: >95% coverage
- Use pytest-cov for coverage reporting

### Test Organization
- Mirror source directory structure in tests/
- Use fixtures for common test data
- Property-based testing with Hypothesis for numerical methods

### Performance Testing
- Include benchmarks for core methods
- Track regression in performance-critical paths
- Test JAX acceleration separately

## Multi-Domain Support

### Domain Modules
- `voiage.healthcare` - Health economics and HTA
- `voiage.financial` - Financial VOI applications
- `voiage.environmental` - Environmental policy VOI

### Domain-Agnostic Core
- Stable numerical algorithms live in the Rust workspace.
- `voiage.methods` exposes Rust-backed facades and explicitly classified
  optional or experimental extensions.
- Domain adapters remain outside the Rust core and may not duplicate stable
  numerical policy.

## Release Process

### Versioning
- Follow semantic versioning (MAJOR.MINOR.PATCH)
- Update `changelog.md` with each release
- Tag releases in Git with version number

### Quality Gates
- All tests must pass before release
- Coverage must not decrease
- Documentation must be updated
- Pre-commit hooks must pass

## Branding & UX

### Documentation
- Comprehensive Astro/Starlight documentation
- Jupyter notebook examples for each feature
- Clear migration guides between versions

### Visual Design
- Consistent plotting styles with matplotlib/seaborn
- Accessible color palettes for charts
- Clear labels and legends in all visualizations
