# voiage - Product Definition

## Vision
To be the premier, cross-domain, high-performance Value of Information (VOI) library, built around a Rust execution core with Python as the reference façade and thin bindings for other languages, empowering researchers and decision-makers across healthcare, finance, environmental policy, and beyond.

## Mission
Fill the critical gap in the VOI ecosystem by providing a comprehensive, open-source analysis toolkit that matches or exceeds the capabilities of commercial tools and R packages while remaining contract-first, polyglot, and reproducible.

## Target Users
- **Health Economists & HTA Agencies**: Researchers performing cost-effectiveness analysis for health technology assessment
- **Clinical Researchers**: Designing adaptive trials and optimizing sample sizes
- **Policy Analystysts**: Evaluating the value of reducing uncertainty in environmental and public policy decisions
- **Financial Analystysts**: Assessing research portfolio optimization and investment decisions under uncertainty
- **Academic Researchers**: Teaching and advancing VOI methodology

## Core Capabilities

### Implemented (v0.2-v0.3)
- **EVPI** (Expected Value of Perfect Information)
- **EVPPI** (Expected Value of Partial Perfect Information) - regression-based methods
- **EVSI** (Expected Value of Sample Information) - two-loop Monte Carlo and regression-based
- **ENBS** (Expected Net Benefit of Sampling)
- **Plotting Suite**: CEAC, EVSI curves, EVPI visualization
- **CLI Interface**: Full command-line tools for batch processing
- **Multi-Domain Support**: Core framework for healthcare, financial, environmental domains
- **JAX Backend**: High-performance computing with JAX/NumPyro integration
- **Contract-First Frontier Methods**: Value of Perspective, preference/individualized-care, validation, threshold, distributional/equity, and implementation-adjusted surfaces with deterministic fixtures
- **Polyglot Binding Scaffolds**: Rust core with Python/Mojo, R and Julia release and contract scaffolds
- **Docs Platform**: Astro/Starlight is the authoritative versioned docs site
  and documentation validation path
- **Community and HPC Readiness**: packaging and review readiness for
  pyOpenSci, rOpenSci, JOSS, scikit-learn-contrib, and NumFOCUS, plus HPC
  distribution strategy for Spack, EasyBuild, HPSF, and E4S

### In Development
- Rust-core migration and ABI shaping
- Dynamic real-options VOI
- Adjacent frontier extensions for causal, data-quality, computational, and elicitation VOI
- Ecosystem integrations and external registry handoff refinement

## Architecture
- **Modular Core**: Domain, numerics, reporting, contracts, CLI, and bindings are split into dedicated modules
- **Backend Abstraction**: Pluggable backends for NumPy, JAX, and the Rust execution core
- **Contract-First Development**: Explicit schemas, deterministic fixtures, and reviewable examples define compatibility
- **CLI-First**: Full command-line interface for all core and frontier methods
- **Polyglot Bindings**: Thin adapters for Python/Mojo, R and Julia over Rust
- **Plotting**: Matplotlib/Seaborn integration for visualization

## Technology Stack
- **Languages**: Rust core, Python 3.12-3.14, and supported binding runtimes for R and Julia; Mojo remains an upstream integration boundary
- **Core Libraries**: NumPy, SciPy, pandas, xarray
- **High-Performance**: JAX, NumPyro, Rust
- **Machine Learning**: scikit-learn, statsmodels
- **CLI**: Typer
- **Visualization**: Matplotlib, Seaborn
- **Testing**: pytest, pytest-cov, hypothesis
- **Linting/Security/Type Checking**: Ruff, ty
- **CI/CD**: GitHub Actions, tox, nox

## Roadmap
- **Next**: Finish the Rust-core migration and narrow ABI boundary
- **Then**: Complete dynamic real-options VOI and adjacent frontier extensions
- **Then**: Maintain and extend contract fixtures, docs, and bindings across languages
- **Ongoing**: Release and registry automation for all supported binding targets
- **Ongoing**: Packaging review readiness, HPC distribution strategy, ABI
  policy, and polyglot repo/docs architecture for the Rust-core future

## Success Metrics
- Feature parity with BCEA, dampack, voi (R packages)
- Surpass commercial tools in method coverage
- Active community contributions
- Publication in Journal of Statistical Software
