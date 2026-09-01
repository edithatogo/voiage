# EasyBuild 2024a foundation candidate

This local overlay prepares Python and scientific dependencies for Voiage
2.2.0 on the requested foss 2024a hierarchy. It does not yet provide the
complete Voiage dependency graph or the separate foss 2023a backport.

The seventeen easyconfigs use GCCcore/GCC 13.3.0, gfbf 2024a, or the system
OpenSSL wrapper. The compiler toolchains belong to the same foss generation. A real EasyBuild 5.4.0 dependency dry-run passed on
macOS arm64 using an isolated Environment Modules 5.6.1 installation. The
retained module list is dependency-resolution evidence, not a native build.

## Source and provider boundaries

`source-manifest-python31214.json` records 63 downloaded and SHA-256-verified
source archives. `providers.json` maps actual extension names to their supplying
modules, including matching build-only and runtime copies. The original
44-source manifest and execution receipts remain unchanged.
`history/manifest-0b43545c.json` describes that immutable commit, not the current
candidate; its original paths must be interpreted at that commit. The recipe, patch, license and evidence bytes are bound by
`manifest.json`. Copied upstream patches and license text remain unchanged;
see `NOTICE` for attribution and the pinned catalogue revision.

The Python 3.12.14 override updates flit-core and typing-extensions to support
the selected sources. Its setuptools-scm 7.1 satisfies python-dateutil's
explicit build requirement below version 8. This is a private site override
of an existing module identity: use a fresh installation prefix and put this
robot directory first. Do not overwrite a site's existing Python module.

Python 3.12.14 is the [12 August security release](https://www.python.org/downloads/release/python-31214/).
Its source contains Expat 2.8.3; neither the recipe nor the pinned Python
easyblock enables system Expat. The derived ctypes patch refreshes two context
lines only. Strict `patch --fuzz=0` application passed, and its resulting files
match the original patch applied with default fuzz. Raw historical patch bytes
remain available alongside the derived patch.

Cython, hatchling, pybind11 and Ninja now depend on Python 3.12.14.
BLIS, OpenBLAS, ICU and FlexiBLAS have matching build/test interpreter overrides;
their other recipe bytes, numerical versions and test settings are unchanged.
The real current robot log contains exactly one Python module: 3.12.14.
Hatchling uses nine verified source distributions, including calver before
trove-classifiers. The pybind11 easyblock installs through both CMake and pip;
its source backend needs setuptools, CMake and Ninja. Explicit binary tool
providers meet those bounds under EasyBuild's build-isolation policy. Its
pytest dependency comes from the controlled Hypothesis provider.

The system OpenSSL wrapper retains its vendor-library path and uses verified
OpenSSL 3.5.8 for its source fallback, following the
[25 August security release](https://openssl-library.org/news/openssl-3.5-notes/).
A native run must record the actual vendor package and security backports if
system libraries are selected; a version string alone does not establish that
status. No native selection or vendor-patch verification is claimed here.

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
only the selected archives listed in the current manifest. The broader
compiler and numerical-library catalogue closure still needs actual download,
patch and native-build verification. The earlier Python 3.12.3 robot receipt
is historical; use `evidence/scientific-robot-python31214.log` for this candidate.

## Remaining work

Backport Arrow/PyArrow with its complete native and source-build dependencies.
The adjacent `easybuild-2024a-polars-overlay` resolves Polars and its dated-nightly
Rust source boundary; native build and installed-module qualification remain pending.
The adjacent `easybuild-2024a-rust-overlay` supplies source-bound
Pydantic and JSON Schema providers, including the required newer rpds-py. Preserve
stable versus dated-nightly Rust selection and vendor exact Cargo sources.
The xarray and scikit-learn provider bundle is resolved for both requested
generations, but still needs native build and installed-module qualification.

A native ARM64 Linux VM can provide real build and module-load evidence for
that architecture. It does not establish x86-64 compatibility, scheduler
behavior, or production cluster performance. Both requested toolchain
builds, installed numerical and Arrow smoke tests, and upstream review remain
pending. No native Python/scientific build or upstream submission is claimed.

## Scientific consumer providers

`Voiage-scientific-consumers/2.2.0-gfbf-2024a` adds checksum-pinned
xarray 2024.11.0 and scikit-learn 1.7.2, together with their selected joblib
1.5.3 and threadpoolctl 3.6.0 runtime providers. Joblib 1.5.3 uses an isolated setuptools 84 build-only provider; the
foundation setuptools 70 provider remains unchanged. The source inventory records
the upstream build and runtime requirements. The EasyBuild 5.4 robot dry run
in `evidence/scientific-consumers-robot.log` resolves the recipe against the
pinned catalogue revision and this overlay. It parses and resolves the graph;
it does not compile sources, load the resulting module, or qualify a full
Voiage installation. `native_python_or_scientific_builds_executed` and
`full_voiage_ready` therefore remain false.
