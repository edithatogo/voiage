# voiage: A Python Library for Value of Information Analysis

[![PyPI version](https://badge.fury.io/py/voiage.svg)](https://badge.fury.io/py/voiage)
[![Build Status](https://github.com/search?q=repo%3Adoughnut%2Fvoiage+workflow%3ACI&type=code)](https://github.com/doughnut/voiage/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

`voiage` is a Python library for Value of Information (VOI) analysis, designed to provide a comprehensive, open-source toolkit for researchers and decision-makers.

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

## Feature Comparison

The following table compares the features of `voiage` with other common VOI software.

| VOI Analysis                                  | `voiage` (Python) | `BCEA` (R) | `dampack` (R) | `voi` (R) | Commercial Tools | Notes                                                                                              |
| :-------------------------------------------- | :---------------: | :--------: | :-----------: | :-------: | :--------------: | :------------------------------------------------------------------------------------------------- |
| **Core Methods**                              |                   |            |               |           |                  |                                                                                                    |
| Expected Value of Perfect Information (EVPI)  |         ✔️         |     ✔️      |       ✔️       |     ✔️     |        ✔️         | The most fundamental VOI metric.                                                                   |
| Expected Value of Partial Perfect Info (EVPPI) |         ✔️         |     ✔️      |       ✔️       |     ✔️     |        ✔️         | `voiage` supports modern, efficient algorithms.                                                    |
| Expected Value of Sample Information (EVSI)   |         ✔️         |     ❌      |       ✔️       |     ✔️     |        ✔️         | `voiage` provides a flexible framework for various data-generating processes.                      |
| Expected Net Benefit of Sampling (ENBS)       |         ✔️         |     ❌      |       ❌       |     ✔️     |        ❌         | Crucial for optimizing research design.                                                            |
| **Advanced & Specialized Methods**            |                   |            |               |           |                  |                                                                                                    |
| Structural Uncertainty VOI                    |         🚧         |     ❌      |       ❌       |     ❌     |        ❌         | For comparing different model structures.                                                          |
| Network Meta-Analysis VOI                     |         🚧         |     ❌      |       ❌       |     ❌     |        ❌         | For synthesizing evidence from multiple studies.                                                   |
| Adaptive Design VOI                           |         🚧         |     ❌      |       ❌       |     ❌     |        ❌         | For trials with pre-planned adaptations.                                                           |
| Portfolio Optimization                        |         🚧         |     ❌      |       ❌       |     ❌     |        ❌         | For prioritizing multiple research opportunities.                                                  |
| Value of Heterogeneity                        |         🚧         |     ❌      |       ❌       |     ❌     |        ❌         | For understanding the value of learning about subgroup effects.                                    |

**Legend:**
*   ✔️: Implemented
*   🚧: In Progress / Planned
*   ❌: Not Supported

## Academic Paper

The academic paper describing the `voiage` library is maintained in the `paper` branch of this repository. For detailed methodological information, mathematical foundations, and comprehensive validation, please refer to:

- Paper source files in the `paper` branch 
- Published version in the Journal of Statistical Software (forthcoming)

## Installation

You can install `voiage` via pip:

```bash
pip install voiage
```

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

For more detailed examples and tutorials, please see the [documentation](https://edithatogo.github.io/voiage).

## Contributing

`voiage` is an open-source project, and we welcome contributions from the community. If you'd like to contribute, please see our [contributing guidelines](CONTRIBUTING.md).

## License

`voiage` is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
