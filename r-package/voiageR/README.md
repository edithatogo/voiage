# voiageR: R Interface to the voiage Python Library

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

`voiageR` provides an R interface to the [`voiage`](https://github.com/doughnut/voiage) Python library for Value of Information (VOI) analysis.

## Overview

Value of Information analysis is a powerful set of techniques used to estimate the value of collecting additional data to reduce uncertainty in decision-making. The `voiageR` package allows R users to leverage the comprehensive VOI methods implemented in the `voiage` Python library.

## Installation

### Prerequisites

1. Install the Python `voiage` package:
   ```bash
   pip install voiage
   ```

2. Install the R `reticulate` package:
   ```r
   install.packages("reticulate")
   ```

### Installing voiageR

You can install the development version of `voiageR` from GitHub:

```r
# install.packages("devtools")
devtools::install_github("doughnut/voiage/r-package/voiageR")
```

## Usage

```r
library(voiageR)

# Create sample net benefit data
net_benefits <- matrix(rnorm(2000), nrow = 1000, ncol = 2)

# Calculate EVPI
evpi_value <- evpi(net_benefits)
print(evpi_value)
```

For more detailed examples, see the vignette:

```r
vignette("voiageR-intro")
```

## Functions

- `evpi()`: Calculate Expected Value of Perfect Information
- `evppi()`: Calculate Expected Value of Partial Perfect Information
- `evsi()`: Calculate Expected Value of Sample Information
- `init_voiage()`: Initialize the voiage Python module
- `is_voiage_available()`: Check if the voiage Python package is available
- `set_voiage_env()`: Set the Python environment for voiage

## License

`voiageR` is licensed under the MIT License.