# voiage conda-forge Recipe

This is the sole Conda recipe for the native Maturin package and the handoff
source for publishing voiage to conda-forge. The release workflow replaces its
version and source hash only after the immutable PyPI sdist is published.

## Qualification and source boundary

The runtime dependency block follows `specs/dependency-runtime-policy.json`:
Python >=3.15.1,<3.16, Polars >=2.1.0,<3 and Arrow >=26.0.0,<27. It declares
neither legacy tabular package. NumPy, SciPy and scikit-learn use the unchanged
current Python package bounds. The drift checker compares every runtime root.

The recipe's published version and source SHA256 remain the immutable 2.1.0
handoff anchor. That older sdist is not the migrated source. Do not build or
publish the updated dependency block against that source and call it qualified.
The release workflow must first bind a published migrated sdist/version/hash;
then a native Conda build and its import/CLI/numerical tests must pass. No Conda
build, feedstock publication or release is claimed by local wheel qualification.

After that source handoff, use `conda build conda-recipe`, followed by installation
and the recipe's native tests. Conda solver availability of the qualified Python
and scientific versions remains an external build gate.

## Publishing to conda-forge

To publish to conda-forge:

1. Fork the [conda-forge/staged-recipes](https://github.com/conda-forge/staged-recipes) repository
2. Copy this recipe directory to `recipes/voiage` in the forked repository
3. Submit a pull request to `conda-forge/staged-recipes`; the recipe is pinned
   to its recorded published sdist and verified SHA256; replace them through the
   release handoff before qualifying the migrated dependency set.

## Recipe Structure

- `meta.yaml`: Main recipe file with package metadata and dependencies

## Dependencies

The recipe specifies the following dependencies:

### Host Dependencies
- python >=3.15.1,<3.16
- pip
- maturin >=1.9,<2.0
- a conda-forge Rust compiler

### Run Dependencies
- python >=3.15.1,<3.16
- click, numpy, scipy, scikit-learn
- pyarrow, polars, pydantic, jsonschema, typing_extensions, typer

## Testing

The recipe includes tests to verify:
- Package imports correctly
- Command-line interface works
- All required dependencies are properly specified
