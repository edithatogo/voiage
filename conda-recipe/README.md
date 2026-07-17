# voiage conda-forge Recipe

This directory contains the conda recipe for publishing voiage to conda-forge.

## Building the conda package

To build the conda package locally:

```bash
# Install conda-build
conda install -n base conda-build

# Build the package
conda build conda-recipe

# Install the built package
conda install -c local voiage
```

## Publishing to conda-forge

To publish to conda-forge:

1. Fork the [conda-forge/staged-recipes](https://github.com/conda-forge/staged-recipes) repository
2. Copy this recipe directory to `recipes/voiage` in the forked repository
3. Submit a pull request to `conda-forge/staged-recipes`; the recipe is pinned
   to the published `voiage 0.2.1` sdist and its verified SHA256 hash.

## Recipe Structure

- `meta.yaml`: Main recipe file with package metadata and dependencies
- `conda_build_config.yaml`: Configuration for building against multiple Python versions
- `conda_build_config.yaml`: Python-version matrix used for local and staged builds

## Dependencies

The recipe specifies the following dependencies:

### Host Dependencies
- python >=3.10
- pip
- setuptools >=69
- setuptools_scm >=8
- wheel

### Run Dependencies
- python >=3.10
- defusedxml >=0.7.1
- numpy, scipy, pandas, xarray, numpyro, jax
- scikit-learn, statsmodels, matplotlib, seaborn
- psutil, typing_extensions, typer

## Testing

The recipe includes tests to verify:
- Package imports correctly
- Command-line interface works
- All required dependencies are properly specified
