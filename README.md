# voiage: A Python Library for Value of Information Analysis

[![PyPI version](https://badge.fury.io/py/voiage.svg)](https://badge.fury.io/py/voiage)
[![Build Status](https://github.com/search?q=repo%3Aedithatogo%2Fvoiage+workflow%3ACI&type=code)](https://github.com/edithatogo/voiage/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

`voiage` is a Python library for Value of Information (VOI) analysis, designed to provide a comprehensive, open-source toolkit for researchers and decision-makers.

Current development state:

- Core VOI methods are implemented and validated.
- Advanced methods such as structural VOI, NMA VOI, adaptive EVSI, portfolio VOI, sequential VOI, calibration VOI, observational VOI, CEAF, dominance, and heterogeneity analysis are implemented.
- CLI polish is in place, including `--format`, `--quiet`, `--verbose`, and `generate-config`.
- Cross-language bindings, HEOML-aligned ecosystem contracts, and fixture-first integration work are scaffolded and tracked in `roadmap.md`.
- The SOTA roadmap now includes frontier VOI methods, led by an experimental Value of Perspective surface for comparing payer, societal, patient, provider, regulator, equity-weighted, and custom stakeholder perspectives side by side, plus fixture-backed manifests, a registry schema, and a reusable frontier contract validator.
- Frontier contract validation now runs through the shared registry manifest, schema, and validator.

## Branch Architecture

This repository uses a separated branch architecture:
- **Main branch**: Contains only the core software library code
- **Paper branch**: Contains the academic paper and related documentation

This separation ensures a clean development environment for software changes while allowing focused development of the academic paper.

## Background: The Need for a Comprehensive VOI Tool in Python

Value of Information (VOI) analysis is a powerful set of techniques used to estimate the value of collecting additional data to reduce uncertainty in decision-making. While several tools for VOI analysis exist, the current landscape has some significant gaps:

*   **Limited Python Support:** The Python ecosystem lacks a mature, comprehensive VOI library. Most existing tools are written in R or are commercial, closed-source products.
*   **Fragmented Features:** Existing tools, even in the R ecosystem, are fragmented. Different packages support different VOI methods, and none of them offer a complete toolkit.
*   **Lack of Advanced Methods:** Many advanced and specialized VOI methods, such as those for adaptive trial designs, network meta-analyses, or structural uncertainty, are not available in any off-the-shelf tool.

`voiage` aims to fill these gaps by providing a single, powerful, and easy-to-use library for a wide range of VOI analyses in Python.

## Feature Matrix

The table below summarizes the current `voiage` capability surface and how it
maps to the active roadmap.

| Capability | State | Notes |
| :-- | :--: | :-- |
| EVPI, EVPPI, EVSI, ENBS | ✅ | Core VOI methods are implemented, tested, and exposed through the API and CLI. |
| CEAF, dominance, heterogeneity | ✅ | Analysis and plotting helpers are available for frontier, dominance, and subgroup workflows. |
| Structural VOI, NMA VOI | ✅ | Structural uncertainty and network meta-analysis methods are implemented. |
| Adaptive, calibration, observational, sequential VOI | ✅ | Trial and study-design oriented workflows are available. |
| Portfolio VOI | ✅ | Budget-constrained portfolio optimization is implemented. |
| CLI developer experience | ✅ | `--format`, `--quiet`, `--verbose`, help examples, and config generation are available. |
| Cross-language binding scaffolds | 🚧 | TypeScript, Go, Rust, Julia, .NET 11, and R scaffolds exist with language-specific CI/release checks. |
| HEOML / ecosystem contracts | 🚧 | `lifecourse` and ecosystem-incubation contract scaffolds exist; deterministic fixtures are being expanded. |
| Numerics, diagnostics, extension model | 📋 | Next planned track for explicit numerical equivalence, diagnostics, and extension rules. |
| Value of Perspective | 🚧 | Experimental Python API, CLI, plot helper, fixture-backed contract scaffold, and registry-backed deterministic fixtures for comparing multiple decision perspectives, regret, switching value, consensus strategies, and Pareto strategies. |
| Frontier VOI methods | 📋 | Planned distributional/equity VOI, implementation-adjusted VOI, preference-information VOI, validation VOI, threshold/tipping-point VOI, robust VOI, and dynamic real-options VOI. |
| Adjacent frontier extensions | 📋 | Planned triage for causal/transportability VOI, data-quality and privacy VOI, computational/model-refinement VOI, expert-elicitation VOI, and evidence-synthesis design VOI. |

**Legend:**
*   ✅: Implemented
*   🚧: Scaffolded or in progress
*   📋: Planned

## Academic Paper

The academic paper describing the `voiage` library is maintained in the `paper` branch of this repository. For detailed methodological information, mathematical foundations, and comprehensive validation, please refer to:

- Paper source files in the `paper` branch 
- Published version in the Journal of Statistical Software (forthcoming)

## Installation

You can install `voiage` via pip:

```bash
pip install voiage
```

Supported Python versions: 3.10-3.14.

## Getting Started

Here's a simple example of how to use `voiage` to calculate the EVPI:

```python
import numpy as np
from voiage.analysis import evpi

# Your model inputs and outputs
psa_inputs = {
    'param1': np.random.rand(1000),
    'param2': np.random.rand(1000),
}
psa_outputs = np.random.rand(1000, 2) # 1000 simulations, 2 strategies

# Calculate the EVPI
evpi_value = evpi(psa_inputs, psa_outputs)
print(f"EVPI: {evpi_value}")
```

## Visual Examples

`voiage` provides comprehensive visualization capabilities for VOI analysis:

### Cost-Effectiveness Acceptability Curve (CEAC)
![CEAC Example](docs/images/ceac_example.png)
*CEAC showing the probability each treatment strategy is cost-effective across different willingness-to-pay thresholds*

### Expected Value of Sample Information (EVSI)
![EVSI Example](docs/images/evsi_example.png)
*EVSI analysis showing how the value of additional data varies with sample size, including Expected Net Benefit of Sampling (ENBS) and research costs. `voiage` provides both two-loop Monte Carlo and regression-based methods for EVSI calculation.*

### Expected Value of Perfect Information (EVPI)
![EVPI vs WTP Example](docs/images/evpi_wtp_example.png)
*EVPI analysis showing how the value of perfect information changes with willingness-to-pay thresholds*

## Command-Line Interface

`voiage` provides a powerful CLI for batch analysis and integration into workflows:

```bash
# Calculate EVPI from CSV data
voiage calculate-evpi net_benefits.csv --population 100000 --time_horizon 10 --discount-rate 0.03

# Calculate EVPI and save to file
voiage calculate-evpi example_net_benefits.csv --output evpi_result.txt

# Calculate EVPPI for specific parameters
voiage calculate-evppi example_net_benefits.csv example_parameters.csv --population 100000

# Full EVPPI analysis with all options
voiage calculate-evppi example_net_benefits.csv example_parameters.csv \
    --population 100000 --time_horizon 15 --discount-rate 0.035 --output results.txt
```

### Example Data Format

**Net Benefits CSV** (`example_net_benefits.csv`):
```csv
Standard_Care,Treatment_A,Treatment_B
95.23,108.45,102.67
87.91,115.23,98.34
...
```

**Parameters CSV** (`example_parameters.csv`):
```csv
effectiveness,cost_multiplier
0.234,1.123
0.187,0.987
...
```

### Sample CLI Output
```bash
$ voiage calculate-evpi example_net_benefits.csv
EVPI: 5.457500

$ voiage calculate-evppi example_net_benefits.csv example_parameters.csv
EVPPI: 0.020708
```

## Current Development State

The active work is split across three layers:

1. **Core library maintenance**: keep the implemented VOI methods stable and
   documented.
2. **Spec-first expansion**: continue the numerics/diagnostics extension model
   and the cross-language conformance fixture tracks.
3. **Ecosystem integration**: finish the HEOML-aligned `lifecourse` contract,
   then mature the optional `innovate` and `mars` integration policies.

The most visible repository-level roadmap items are already reflected in
[`roadmap.md`](roadmap.md):

- **Phase 5**: Spec, fixtures, and polyglot bindings
- **Phase 6**: Ecosystem integrations
- **Phase 7**: SOTA VOI frontier methods, starting with Value of Perspective

That means the remaining work is mostly contract hardening, fixture parity,
language-binding maturation, and documentation around the new ecosystem
boundaries rather than fundamental method implementation.

For more detailed examples and tutorials, please see the [documentation](https://edithatogo.github.io/voiage).

## Contributing

`voiage` is an open-source project, and we welcome contributions from the community. If you'd like to contribute, please see our [contributing guidelines](CONTRIBUTING.md).

## License

`voiage` is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
