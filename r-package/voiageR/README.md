# voiageR: R Interface to the Rust-backed voiage Library

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

`voiageR` provides an R interface to the Rust-backed
[`voiage`](https://github.com/edithatogo/voiage) library for Value of
Information (VOI) analysis.

## Overview

Value of Information analysis estimates the value of collecting additional
data to reduce uncertainty in decision-making. `voiageR` calls the shared Rust
implementation directly for EVPI. Advanced EVPPI and EVSI methods currently use
the Python package through `reticulate`.

## Installation

### Prerequisites

The `evpi()` function calls the Rust `voiage_v1_evpi_i32_r` C ABI directly.
Set `VOIAGE_FFI_LIBRARY` to the built `voiage-ffi` library when it is not on
the system library path.

For advanced `evppi()` and `evsi()` methods, install the Python `voiage`
package and the R `reticulate` package:

```bash
pip install voiage
```

```r
install.packages("reticulate")
```

The direct `evpi()` path does not require Python or `reticulate`.

### Installing voiageR

You can install the development version of `voiageR` from GitHub:

```r
# install.packages("devtools")
devtools::install_github("edithatogo/voiage", subdir = "r-package/voiageR")
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

### Built-in two-loop EVSI

The R facade supports the built-in joint-normal two-loop study model through
the Python bridge. Each trial arm needs a matching prior parameter named
`mean_<normalised arm>`: arm names are lowercased and spaces become
underscores. The known outcome standard deviation is supplied as a finite,
strictly positive `sd_outcome` fixed across all prior draws.

```r
set.seed(17)
draws <- 1000L
prior_samples <- list(
  mean_treatment = rnorm(draws, mean = 0.06, sd = 0.03),
  mean_control = rnorm(draws, mean = 0.00, sd = 0.02),
  sd_outcome = rep(1.0, draws)
)
trial_design <- list(
  arms = list(
    list(name = "Treatment", sample_size = 100L),
    list(name = "Control", sample_size = 100L)
  )
)
model_func <- function(params) {
  cbind(
    Treatment = 50000 * params$mean_treatment - 3000,
    Control = 50000 * params$mean_control
  )
}

evsi(
  model_func,
  prior_samples,
  trial_design,
  method = "two_loop",
  n_outer_loops = 100L,
  n_inner_loops = 1000L,
  seed = 20260724L
)
```

Custom `trial_simulator` and `posterior_sampler` callbacks remain Python-only.
The R facade does not convert or execute them and does not claim custom-model
parity.

## Functions

- `evpi()`: Calculate Expected Value of Perfect Information
- `evppi()`: Calculate Expected Value of Partial Perfect Information
- `evsi()`: Calculate Expected Value of Sample Information
- `init_voiage()`: Initialize the voiage Python module
- `is_voiage_available()`: Check if the voiage Python package is available
- `set_voiage_env()`: Set the Python environment for voiage

## License

`voiageR` is licensed under the Apache License 2.0.
