# voiage Examples

This directory contains example notebooks demonstrating the usage of the voiage library.

## Validation Notebook

- [voiage_validation.ipynb](voiage_validation.ipynb): A comprehensive validation notebook that demonstrates all core methods of the voiage library:
  - EVPI (Expected Value of Perfect Information) validation against the benchmark case
  - EVPPI (Expected Value of Partial Perfect Information) validation for parameter subsets
  - EVSI (Expected Value of Sample Information) calculation using both two-loop and regression methods
  - Visualization of results with CEACs and VOI curves

## Additional Notebooks

- Validation notebooks: `evpi_validation.ipynb`, `evppi_validation.ipynb`, `evsi_validation.ipynb`, `nma_validation.ipynb`, `structural_voi_validation.ipynb`
- Tutorials: `getting_started.ipynb`, `advanced_methods.ipynb`, `financial_voi.ipynb`, `environmental_voi.ipynb`, `engineering_voi.ipynb`, `jax_performance.ipynb`
- CLI samples: `cli_samples/coss_study_design.json` provides a synthetic,
  provenance-labelled input for the experimental COSS calculation and plot.

## Runnable Python examples

- [`expected_utility_information.py`](expected_utility_information.py): runs
  the experimental Rust-backed EUI, CEI, BPI, SPI, anchored PPI, and VoC
  presentation contract using a deterministic nonlinear fixture. VoC remains a
  presentation of the shared result; the example does not claim a nonlinear
  EVPI alias or R, Julia, or Mojo support.

Run it from the repository root:

```bash
uv run python examples/expected_utility_information.py
```

## Usage

To run the examples, you'll need to have Jupyter installed:

```bash
pip install jupyter
```

Then you can start Jupyter and open the notebooks:

```bash
jupyter notebook
```

## Dependencies

The examples require the following packages in addition to voiage:
- jupyter
- matplotlib
- numpy

These can be installed with:

```bash
pip install jupyter matplotlib numpy
```
