# voiage - Product Definition

## Vision
To be the premier, cross-domain, high-performance Python library for Value of Information (VOI) analysis, empowering researchers and decision-makers across healthcare, finance, environmental policy, and beyond.

## Mission
Fill the critical gap in the Python ecosystem by providing a comprehensive, open-source VOI analysis toolkit that matches or exceeds the capabilities of commercial tools and R packages.

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

### In Development
- Structural Uncertainty VOI
- Network Meta-Analysis VOI
- Adaptive Trial VOI
- Portfolio Optimization
- Value of Heterogeneity

## Architecture
- **Modular Design**: Domain-specific modules (healthcare, financial, environmental)
- **Backend Abstraction**: Pluggable backends (NumPy, JAX)
- **Factory Pattern**: Extensible method instantiation
- **CLI-First**: Full command-line interface for all core methods
- **Plotting**: Matplotlib/Seaborn integration for visualization

## Technology Stack
- **Language**: Python >=3.8
- **Core Libraries**: NumPy, SciPy, pandas, xarray
- **High-Performance**: JAX, NumPyro
- **Machine Learning**: scikit-learn, statsmodels
- **CLI**: Typer
- **Visualization**: Matplotlib, Seaborn
- **Testing**: pytest, pytest-cov, hypothesis
- **Linting/Type Checking**: Ruff, MyPy
- **CI/CD**: GitHub Actions, tox

## Roadmap
- **v0.4**: Complete advanced methods (structural uncertainty, network NMA)
- **v0.5**: Portfolio optimization and sequential decisions
- **v0.6**: Enhanced plotting and visualization capabilities
- **v1.0**: Full feature parity with commercial tools

## Success Metrics
- Feature parity with BCEA, dampack, voi (R packages)
- Surpass commercial tools in method coverage
- Active community contributions
- Publication in Journal of Statistical Software
