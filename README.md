# voiage: A Python Library for Value of Information Analysis

[![PyPI version](https://badge.fury.io/py/voiage.svg)](https://badge.fury.io/py/voiage)
[![CI](https://github.com/edithatogo/voiage/actions/workflows/ci.yml/badge.svg)](https://github.com/edithatogo/voiage/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/edithatogo/voiage/branch/main/graph/badge.svg)](https://codecov.io/gh/edithatogo/voiage)
[![Python](https://img.shields.io/badge/python-3.12--3.14-blue)](https://www.python.org/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-yellow.svg)](https://opensource.org/licenses/Apache-2.0)

`voiage` is a v1.0.0 open-source library for Value of Information (VOI)
analysis, combining a Python interface with an authoritative Rust numerical
core and shared R and Julia binding contracts.

Current status:

- Stable VOI methods are implemented and Rust-backed, with Python façade,
  schema, orchestration, CLI, plotting, and reporting layers.
- R and Julia use the shared Rust C ABI; Mojo remains an explicitly external
  upstream boundary.
- The signed v1.0.0 release is public on GitHub and PyPI.
- Registry review/indexing, JOSS/RRID submission, Software Heritage snapshot
  verification, and experimental extensions remain separately tracked.

## Branch Architecture

The default `main` branch contains the maintained software and its submission
metadata. The JOSS draft is in `paper.md` with references in `paper.bib`;
`docs/joss-submission-readiness.md` records the remaining author and impact
checks before submission.

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
| Cross-language bindings | ✅/external | Rust, Python, R, and Julia are retained; Mojo is an upstream boundary. |
| HEOML / ecosystem contracts | 🚧 | Optional ecosystem contracts remain follow-on work. |
| Numerics, diagnostics, extension model | ✅ | Stable contracts and extension policies are enforced by tests and CI. |
| Value of Perspective | 🚧 | Experimental Python API, CLI, plot helper, fixture-backed contract scaffold, and registry-backed deterministic fixtures for comparing multiple decision perspectives, regret, switching value, consensus strategies, and Pareto strategies. |
| Frontier VOI methods | 🚧 | Several fixture-backed experimental surfaces exist; stable promotion remains governed by the extension policy. |
| Adjacent frontier extensions | 📋 | Planned triage for causal/transportability VOI, data-quality and privacy VOI, computational/model-refinement VOI, expert-elicitation VOI, and evidence-synthesis design VOI. |

**Legend:**
*   ✅: Implemented
*   🚧: Optional, experimental, or in progress
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

## Quality and security

The repository uses a layered, fail-closed verification model:

| Layer | Evidence |
| :-- | :-- |
| Formatting and lint | Ruff, Ruff format, Vale, Bandit, Vulture, and repository harness |
| Static typing | `ty` across the retained Python runtime |
| Unit and contract tests | Pytest suites with schema, API, provenance, and version-sync contracts |
| Integration and E2E | Marked integration suites, CLI E2E workflows, clean-install and package checks |
| Coverage | Branch coverage, changed-line and critical-module policy, 90% Python threshold, Codecov upload |
| Property and differential testing | Hypothesis, Rust proptest, cross-language fixtures, metamorphic and parity tests |
| Mutation testing | Broad ratchet, critical-kernel threshold, and externally anchored mutation cohort |
| Rust safety | Cargo tests/clippy, MSRV, Miri, fuzzing, FFI sanitizers, advisories, licenses, and source policy |
| Supply chain | Pinned GitHub Actions, CodeQL, Scorecard, zizmor, dependency review, SBOM, provenance, checksums, and signatures |
| Dependency updates | Weekly Dependabot updates for Python, Cargo, and GitHub Actions with cooldown windows |

Renovate is not enabled because Dependabot already owns the three maintained
dependency ecosystems; enabling both would duplicate update PRs. The remaining
quality debt is tracked explicitly, including broader Python annotation
coverage and external registry evidence.

## Academic paper

The JOSS draft is maintained in [`paper.md`](paper.md), with references in
[`paper.bib`](paper.bib), software metadata in [`codemeta.json`](codemeta.json),
and citation metadata in [`CITATION.cff`](CITATION.cff). The paper has not yet
been submitted to JOSS or arXiv.

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

## Current development state

The mature-v1 repository programme is complete and archived. Remaining work is
deliberately external or optional: registry review/indexing, JOSS and RRID
submission, Software Heritage snapshot verification, and experimental or
ecosystem extensions. See [`roadmap.md`](roadmap.md), [`todo.md`](todo.md),
and the [Conductor registry](conductor/tracks.md) for evidence-backed status.

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

`voiage` is licensed under the Apache License 2.0. See the [LICENSE](LICENSE)
file for details.
