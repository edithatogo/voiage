# voiage: A Python Library for Value of Information Analysis

[![PyPI version](https://badge.fury.io/py/voiage.svg)](https://badge.fury.io/py/voiage)
[![CI](https://github.com/edithatogo/voiage/actions/workflows/ci.yml/badge.svg)](https://github.com/edithatogo/voiage/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/badge/coverage-90%25-brightgreen)](https://github.com/edithatogo/voiage/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.12--3.14-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

`voiage` is a Python library for Value of Information (VOI) analysis, designed to provide a comprehensive, open-source toolkit for researchers and decision-makers.

Current development state:

- Core VOI methods are implemented and validated.
- Advanced methods such as structural VOI, NMA VOI, adaptive EVSI, portfolio VOI, sequential VOI, calibration VOI, observational VOI, CEAF, dominance, and heterogeneity analysis are implemented.
- CLI polish is in place, including `--format`, `--quiet`, `--verbose`, and `generate-config`.
- Cross-language bindings, HEOML-aligned ecosystem contracts, and fixture-first integration work are scaffolded and tracked in `roadmap.md`; the Rust-core migration is now a distinct roadmap program with the Rust domain model and deterministic EVPI/ENBS slices in place, Rust as the authoritative execution core, Python as the primary façade, and the non-Python packages as thin bindings/adapters over the same contract once that migration lands. The R binding currently releases source archives through GitHub Releases, with CRAN and r-universe still handled through external registry processes.
- The R documentation/manual track is complete and archived: package help pages, a narrative vignette, a deterministic PDF manual, and the verification/release-handoff guidance are all in place.
- The polyglot tutorial surface track is complete and archived: the Python notebooks, the R vignette/manual, and the non-Python binding walkthrough READMEs now share the same canonical VOI examples, and the repo includes smoke checks for the binding walkthroughs.
- The SOTA roadmap now includes frontier VOI methods, led by Value of Perspective plus preference, validation, and threshold surfaces for comparing payer, societal, patient, provider, regulator, equity-weighted, and custom stakeholder perspectives and preference profiles side by side, along with fixture-backed manifests, a registry schema, and a reusable frontier contract validator.
- Frontier contract validation now runs through the shared registry manifest, schema, and validator.
- Community support and governance docs are now explicit: `SUPPORT.md`, `CODE_OF_CONDUCT.md`, and `SECURITY.md` provide the first stop for help, conduct, and vulnerability reporting.

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
| Cross-language bindings | 🚧 | Rust is the core; Python, Mojo, Julia, and R are the supported binding surfaces. |
| HEOML / ecosystem contracts | 🚧 | `lifecourse` and ecosystem-incubation contract scaffolds exist; deterministic fixtures are being expanded. |
| Numerics, diagnostics, extension model | 📋 | Next planned track for explicit numerical equivalence, diagnostics, and extension rules. |
| Value of Perspective | 🚧 | Experimental Python API, CLI, plot helper, fixture-backed contract scaffold, and registry-backed deterministic fixtures for comparing multiple decision perspectives, regret, switching value, consensus strategies, and Pareto strategies. |
| Frontier VOI methods | 🚧 | Value of Perspective, validation, threshold, and preference/individualized-care runtime surfaces are implemented with CLI entrypoints and fixture-backed conformance contracts; distributional/equity VOI and implementation-adjusted VOI have experimental runtime surfaces with deterministic fixtures, while robust VOI and dynamic real-options VOI remain planned. |
| Adjacent frontier extensions | 📋 | Planned triage for causal/transportability VOI, data-quality and privacy VOI, computational/model-refinement VOI, expert-elicitation VOI, and evidence-synthesis design VOI. |

**Legend:**
*   ✅: Implemented
*   🚧: Scaffolded or in progress
*   📋: Planned

## Comparison With R Packages

The table below is intentionally high level. It highlights where `voiage`
already offers a broader or more explicit Python surface, not a full feature
parity audit of the R ecosystem.

| Capability | voiage | BCEA | dampack | voi |
| :-- | :--: | :--: | :--: | :--: |
| Core EVPI / EVPPI / EVSI | ✅ | ✅ | ✅ | ✅ |
| CEAF / dominance / subgroup analysis | ✅ | ✅ | ⚪ | ✅ |
| Adaptive / sequential / portfolio VOI | ✅ | ⚪ | ⚪ | ⚪ |
| Structural / NMA / cross-domain VOI | ✅ | ⚪ | ⚪ | ⚪ |
| CLI-first workflows | ✅ | ⚪ | ⚪ | ⚪ |
| Frontier / perspective analysis | 🚧 | ⚪ | ⚪ | ⚪ |

Legend: ✅ supported, ⚪ not a primary focus or not exposed as a first-class
workflow in the package documentation.

## Documentation

The main user and developer references are:

- [Getting started](https://edithatogo.github.io/voiage/getting-started/)
- [Notebook tutorials and examples](https://edithatogo.github.io/voiage/examples/)
- [R vignette and manual source](r-package/voiageR/vignettes/voiageR-getting-started.Rmd)
- [Julia walkthrough](bindings/julia/README.md)
- [R package](r-package/voiageR/README.md)
- [CLI reference](https://edithatogo.github.io/voiage/cli-reference/)
- [Method reference](https://edithatogo.github.io/voiage/methods/)
- [Plotting reference](https://edithatogo.github.io/voiage/user-guide/plotting/)
- [Data structures](https://edithatogo.github.io/voiage/data-structures/)
- [Backends](https://edithatogo.github.io/voiage/backends/)
- [Developer guide](https://edithatogo.github.io/voiage/developer-guide/)
- [Community support](SUPPORT.md)
- [Code of conduct](CODE_OF_CONDUCT.md)
- [Security policy](SECURITY.md)
- [Frontier VOI roadmap](https://edithatogo.github.io/voiage/sota-voi-frontier/)

## Academic Paper

The academic paper describing the `voiage` library is maintained in the `paper` branch of this repository. For detailed methodological information, mathematical foundations, and comprehensive validation, please refer to:

- Paper source files in the `paper` branch
- Published version in the Journal of Statistical Software (forthcoming)

## Installation

You can install `voiage` via pip:

```bash
pip install voiage
```

Supported Python versions: 3.12-3.14.

## Getting Started

Here's a small example that works out of the box:

```python
import numpy as np
from voiage.analysis import DecisionAnalysis
from voiage.schema import ValueArray

net_benefit = ValueArray.from_numpy(
    np.array(
        [
            [10.0, 12.0],
            [11.0, 9.0],
            [13.0, 14.0],
        ]
    ),
    strategy_names=["Standard care", "New treatment"],
)

analysis = DecisionAnalysis(net_benefit)
print(f"EVPI: {analysis.evpi():.3f}")
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
voiage calculate-evpi net_benefits.csv --population 100000 --time-horizon 10 --discount-rate 0.03

# Calculate EVPI and save to file
voiage calculate-evpi examples/cli_samples/evpi_net_benefit.csv --output evpi_result.txt

# Calculate EVPPI for specific parameters
voiage calculate-evppi examples/cli_samples/evpi_net_benefit.csv examples/cli_samples/evppi_parameters.csv --population 100000

# Full EVPPI analysis with all options
voiage calculate-evppi examples/cli_samples/evpi_net_benefit.csv examples/cli_samples/evppi_parameters.csv \
    --population 100000 --time-horizon 15 --discount-rate 0.035 --output results.txt
```

### Example Data Format

**Net Benefits CSV** (`examples/cli_samples/evpi_net_benefit.csv`):
```csv
standard care,new treatment
20000,25000
21000,24800
20500,25250
```

**Parameters CSV** (`examples/cli_samples/evppi_parameters.csv`):
```csv
treatment_effect,cost_shift
0.1,-0.2
0.4,0.0
0.9,0.2
```

### Sample CLI Output
```bash
$ voiage calculate-evpi examples/cli_samples/evpi_net_benefit.csv
EVPI: 5.457500

$ voiage calculate-evppi examples/cli_samples/evpi_net_benefit.csv examples/cli_samples/evppi_parameters.csv
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

## Why voiage?

`voiage` exists to make VOI analysis practical in Python without forcing users
to stitch together separate packages for core methods, advanced methods, plots,
fixtures, and cross-domain workflows. The project aims to combine:

- a single `DecisionAnalysis`-centric API
- explicit CLI support for reproducible batch workflows
- fixture-backed contracts for stable testing and interoperability
- a growing frontier surface that includes perspective, equity, implementation,
  and adjacent VOI families

The design goal is to keep the core library easy to script while still leaving
room for specialized methods, binding generation, and registry-backed release
automation.

For more detailed examples and tutorials, please see the [documentation](https://edithatogo.github.io/voiage).

## Contributing

`voiage` is an open-source project, and we welcome contributions from the community. If you'd like to contribute, please see our [contributing guidelines](CONTRIBUTING.md).

## Getting Help

- Open an issue for bugs, doc fixes, or feature requests.
- Use GitHub Discussions for design questions or roadmap feedback.
- For implementation work, start from the relevant Conductor track and add the
  smallest test that demonstrates the issue or desired behavior.

## License

`voiage` is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
