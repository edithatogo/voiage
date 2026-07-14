# voiage conda-forge Recipe

This directory contains the conda recipe for publishing voiage to conda-forge.

## Building the conda package

To build the conda package locally:

```bash
# Install conda-build
conda install conda-build

# Build the package
conda build .

# Install the built package
conda install -c local voiage
```

## Publishing to conda-forge

To publish to conda-forge:

1. Fork the [conda-forge/staged-recipes](https://github.com/conda-forge/staged-recipes) repository
2. Copy this recipe directory to `recipes/voiage` in the forked repository
3. Update the SHA256 hash in `meta.yaml` with the actual hash of the PyPI release
4. Submit a pull request to conda-forge/staged-recipes

## Recipe Structure

- `meta.yaml`: Main recipe file with package metadata and dependencies
- `conda_build_config.yaml`: Configuration for building against multiple Python versions
- `build.sh`: Build script for Unix-like systems (automatically generated)
- `bld.bat`: Build script for Windows (automatically generated)

## Dependencies

The recipe specifies the following dependencies:

### Host Dependencies
- python >=3.8
- pip
- setuptools >=45
- wheel

### Run Dependencies
- python >=3.8
- numpy >=1.20.0
- scipy >=1.7.0
- pandas >=1.3.0
- xarray >=0.18.0
- matplotlib >=3.4.0
- seaborn >=0.11.0
- scikit-learn >=1.0.0
- jax >=0.2.0
- jaxlib >=0.1.65

## Testing

The recipe includes tests to verify:
- Package imports correctly
- Command-line interface works
- All required dependencies are properly specified