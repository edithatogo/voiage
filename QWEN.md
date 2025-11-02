# voiage: Python Library for Value of Information Analysis

## Project Overview

voiage is a comprehensive Python library for Value of Information (VOI) analysis, designed to provide a complete, open-source toolkit for researchers and decision-makers. The library addresses key gaps in the Python ecosystem by offering a unified, easy-to-use solution for various VOI methods that are typically fragmented across different tools or unavailable in Python.

### Key Features

- **Core VOI Methods:**
  - Expected Value of Perfect Information (EVPI)
  - Expected Value of Partial Perfect Information (EVPPI)
  - Expected Value of Sample Information (EVSI)
  - Expected Net Benefit of Sampling (ENBS)

- **Advanced VOI Methods:**
  - Structural Uncertainty VOI
  - Network Meta-Analysis VOI
  - Adaptive Design VOI
  - Portfolio Optimization
  - Value of Heterogeneity

- **Advanced Capabilities:**
  - Web API interface for remote access
  - Command-line interface for automation
  - Docker deployment support
  - GPU acceleration
  - Memory optimization for large datasets
  - Streaming data processing
  - Result caching
  - Parallel processing
  - Interactive plotting widgets

## Project Structure

```
voiage/
├── voiage/                 # Main library source code
│   ├── core/               # Core utilities and infrastructure
│   ├── methods/            # VOI calculation implementations
│   ├── web/                # FastAPI web API
│   ├── plot/               # Visualization capabilities
│   ├── widgets/            # Interactive analysis widgets
│   ├── healthcare/         # Healthcare-specific VOI methods
│   ├── financial/          # Financial risk VOI methods
│   ├── environmental/      # Environmental impact VOI methods
│   ├── analysis.py         # Main analysis interface
│   ├── cli.py              # Command-line interface
│   ├── schema.py           # Data structure definitions
│   ├── exceptions.py       # Custom exception classes
│   └── ...                 # Other modules
├── tests/                  # Comprehensive test suite
├── docs/                   # Documentation
├── examples/               # Usage examples
├── pyproject.toml          # Python build and dependency configuration
├── Dockerfile              # Docker container configuration
├── docker-compose.yml      # Docker orchestration
└── README.md               # Project overview and usage
```

## Key Dependencies

- `numpy>=1.20,<2.0`
- `scipy>=1.7,<1.15`
- `pandas>=1.3,<3.0`
- `xarray>=0.19,<2025.0`
- `numpyro>=0.13,<0.20`
- `jax>=0.4,<0.5`
- `scikit-learn>=1.0,<2.0`
- `statsmodels>=0.13,<1.0`
- `matplotlib>=3.4,<4.0`
- `seaborn>=0.11,<1.0`
- `typer[all]>=0.9,<1.0`

## Building and Running

### Installation

Install the library directly with pip:
```bash
pip install voiage
```

### From Source

1. Clone the repository and navigate to the project directory
2. Install in development mode:
```bash
pip install -e .
```

### Docker Deployment

The project includes Docker support for containerized deployment:
```bash
# Build the container
docker build -t voiage .

# Run the web API
docker run -p 8000:8000 voiage
```

### Web API

The library includes a FastAPI-based web API that can be run with:
```bash
python -m voiage.web.main
```
The API provides endpoints for all core VOI calculations accessible via HTTP requests.

### Command Line Interface

The CLI provides direct access to VOI calculations:
```bash
# Calculate EVPI
voiage calculate-evpi data.csv

# Calculate EVPPI
voiage calculate-evppi net_benefit.csv parameter_file.csv
```

## Development Conventions

### Testing

The project uses pytest for testing with extensive coverage:
- Unit tests for individual methods
- Integration tests for complex workflows
- Property-based testing using Hypothesis
- Performance benchmarks

Run tests with:
```bash
pytest
```

### Code Quality

- Type hints are used throughout the codebase
- Ruff is used for linting and formatting
- MyPy for type checking
- Pre-commit hooks for code quality enforcement

### Data Structures

The library uses well-defined data structures:
- `ValueArray`: For net benefit values from PSA
- `ParameterSet`: For parameter samples from PSA
- `TrialDesign`: For clinical trial specifications
- `PortfolioSpec`: For research portfolio optimization

## Usage Examples

### Basic Usage
```python
import numpy as np
from voiage.analysis import DecisionAnalysis

# Create sample data (1000 PSA samples, 2 strategies)
psa_outputs = np.random.rand(1000, 2)

# Perform VOI analysis
analysis = DecisionAnalysis(nb_array=psa_outputs)
evpi_value = analysis.evpi()
print(f"EVPI: {evpi_value}")
```

### With Parameter Samples
```python
from voiage.schema import ParameterSet

# Create parameter samples
param_samples = {
    'param1': np.random.rand(1000),
    'param2': np.random.rand(1000)
}

# Perform EVPPI analysis
parameter_set = ParameterSet.from_numpy_or_dict(param_samples)
analysis = DecisionAnalysis(nb_array=psa_outputs, parameter_samples=parameter_set)
evppi_value = analysis.evppi()
print(f"EVPPI: {evppi_value}")
```

## Documentation and Resources

- [Documentation](https://voiage.readthedocs.io) (placeholder link)
- [API Reference](https://voiage.readthedocs.io/api) (placeholder link)
- [GitHub Repository](https://github.com/doughnut/voiage)

## Contributing

The project welcomes contributions. See the [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines on how to contribute code, documentation, or bug reports.