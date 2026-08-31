# EasyBuild 2024a foundation candidate

This local overlay prepares Python and scientific dependencies for Voiage
2.2.0 on the requested foss 2024a hierarchy. It does not yet provide the
complete Voiage dependency graph or the separate foss 2023a backport.

The six easyconfigs use GCCcore 13.3.0 or gfbf 2024a, which are components of
the same foss generation. A real EasyBuild 5.4.0 dependency dry-run passed on
macOS arm64 using an isolated Environment Modules 5.6.1 installation. The
retained module list is dependency-resolution evidence, not a native build.

## Source and provider boundaries

`source-manifest.json` records 44 downloaded and SHA-256-verified source
archives. `providers.json` maps actual extension names to their supplying
modules. The recipe, patch, license and evidence bytes are bound by
`manifest.json`. Copied upstream patches and license text remain unchanged;
see `NOTICE` for attribution and the pinned catalogue revision.

The Python 3.12.3 override updates flit-core and typing-extensions to support
the selected sources. Its setuptools-scm 7.1 satisfies python-dateutil's
explicit build requirement below version 8. This is a private site override
of an existing module identity: use a fresh installation prefix and put this
robot directory first. Do not overwrite a site's existing Python module.

The scientific bundle supplies NumPy 2.2.6, SciPy 1.16.3 and pandas 2.3.3.
Specialized EasyBuild NumPy and SciPy blocks retain their BLAS configuration.
SciPy slow tests are enabled and both scientific test failure gates are explicit.
It needs Meson 1.5.2 and pybind11 2.13.6; the older 2024a defaults do not meet
SciPy's source requirements. Cython remains 3.0.10, within SciPy's upper
bound below 3.2. The meson-python override uses Python's packaging provider,
so it does not load a second unrelated Python bundle just for packaging.

The support bundle supplies the CLI and date/time dependencies. A separate
Hypothesis test-provider override avoids loading the older broad Python bundle
into the scientific test environment. Bundles install their own selected
providers even when a build-only module exposes the same version, so unloading
build tools cannot remove a required runtime provider. Sixteen
selected support distributions were built from their verified source
archives in a separate macOS Python 3.12 environment. Dependency consistency
and Typer/Click help checks passed. The receipt lists the exact installed
versions and distinguishes this consumer test from EasyBuild installation.
Four further test-command helpers were source-built, and the actual SciPy
source archive's developer CLI help commands passed. This verifies test-command
initialization with Click 8.5, not execution of the scientific tests.

## Reproduce dependency resolution

Use the complete easyconfigs catalogue at the revision in `manifest.json`,
EasyBuild framework/easyblocks 5.4.0 and a functioning Modules installation.
Set `EASYBUILD_CATALOGUE` to that checkout's `easybuild/easyconfigs` directory
and `EASYBUILD_PREFIX` to a new private prefix. From the repository root:

```sh
eb packaging/easybuild-overlay/2024a/SciPy-bundle-2026.09-gfbf-2024a-voiage-2.2.0.eb \
  --robot --dry-run \
  --robot-paths=packaging/easybuild-overlay/2024a:"$EASYBUILD_CATALOGUE" \
  --prefix="$EASYBUILD_PREFIX" \
  --modules-tool=EnvironmentModules --module-syntax=Tcl
```

The earlier module-syntax and absent-module-tool failures are retained beside
the successful dry-run. No mock module implementation was used. A successful
dry-run does not prove all sources, patches or native packages in the larger
catalogue closure have been downloaded or built; the source manifest binds
only this overlay's 44 selected archives. The referenced hatchling build-tool
bundle still uses an upstream pure-Python trove-classifiers wheel; a claim
that the entire catalogue closure is source-only would require a separate
override or documented exception.

## Remaining work

Backport xarray, scikit-learn, Arrow/PyArrow, Polars, Pydantic and JSON Schema
with their complete native and source-build dependencies. JSON Schema needs
a newer rpds-py than the older catalogue support bundle provides. Preserve
stable versus dated-nightly Rust selection and vendor exact Cargo sources.
Then adapt the verified provider families to foss 2023a separately.

A native ARM64 Linux VM can provide real build and module-load evidence for
that architecture. It does not establish x86-64 compatibility, scheduler
behavior, or production cluster performance. Both requested toolchain
builds, installed numerical and Arrow smoke tests, and upstream review remain
pending. No native Python/scientific build or upstream submission is claimed.
