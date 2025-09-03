# voiage Examples

This directory contains example notebooks demonstrating the usage of the voiage library.

## Validation Notebook

- [voiage_validation.ipynb](voiage_validation.ipynb): A comprehensive validation notebook that demonstrates all core methods of the voiage library:
  - EVPI (Expected Value of Perfect Information) calculation
  - EVPPI (Expected Value of Partial Perfect Information) calculation
  - EVSI (Expected Value of Sample Information) calculation using both two-loop and regression methods
  - Visualization of results with CEACs and VOI curves

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