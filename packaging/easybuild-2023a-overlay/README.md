# EasyBuild foss 2023a foundation

This candidate supplies a Python 3.12.14 scientific and support foundation for
Voiage 2.2.0. It preserves GCC 12.3.0 and the gfbf 2023a subset of foss 2023a.
It does not establish a complete Voiage installation or a native build.
The separate [2024a evidence](../easybuild-overlay/README.md) is unchanged.

Twenty local recipes cover Python, Meson, meson-python, Ninja, Cython, hatchling,
pybind11, scientific providers, CLI/date support and scientific test helpers.
The source manifests bind 71 downloaded archives. Backend requirements are
checked in installation order against the actual module providers. NumPy
2.2.6, SciPy 1.16.3 and pandas 2.3.3 remain within the release constraints.
SciPy slow tests and the NumPy/SciPy failure gates remain enabled.

The pinned catalogue has Python 3.11 bindings and cannot supply this foundation
through a toolchain-name substitution. Cython, hatchling and pybind11 bind the
new Python explicitly. Ninja also uses it for its build script. The pybind11
2.13.6 EasyBuild block runs both Python and CMake install paths. The Python
backend requires setuptools 42, CMake 3.18 and Ninja; native tests additionally
require Boost 1.56, Catch 2.13.9 and pytest 3.1. The selected CMake 3.26.3 and
explicit Ninja 1.11.1 build providers meet the stronger Python backend bounds. Its test
provider uses the controlled Hypothesis bundle instead of the older broad
Python package bundle.

Hatchling's trove-classifiers wheel is replaced with its verified source
archive and a preceding calver provider. Nine selected backend distributions
were built from these sources in an isolated macOS Python consumer; pip check
passed. That consumer is separate from the unbuilt EasyBuild modules.

## Reproduce the dependency inspection

Use EasyBuild 5.4.0 and Environment Modules with Tcl module syntax. Set
`EASYBUILD_CATALOGUE` to the easyconfigs directory from catalogue commit
`58e8b5a48767cbed1bf5669675d9638580d7259f` and `EASYBUILD_PREFIX` to a new
private installation directory. Local same-identity overrides must not replace
existing site modules.

```sh
eb packaging/easybuild-2023a-overlay/2023a/SciPy-bundle-2026.09-gfbf-2023a-voiage-2.2.0.eb \
  --robot --dry-run \
  --robot-paths="packaging/easybuild-2023a-overlay/2023a:$EASYBUILD_CATALOGUE" \
  --prefix="$EASYBUILD_PREFIX" \
  --modules-tool=EnvironmentModules --module-syntax=Tcl
```

The retained dependency dry run passes for this foundation. The initial
incomplete generation substitutions failed and are retained separately.
An intermediate run used the superseded Python 3.12.3 source; it is historical
only. A dry run does not apply source patches, compile libraries, execute their
tests or validate installed modules. Separate zero-fuzz checks pass for both
Python patches and the pybind11 Catch patch. The ctypes patch refresh changes
context only; its added and removed lines match the original patch.

## Remaining boundaries

Four native recipes (ICU, BLIS, OpenBLAS and FlexiBLAS) now use Python 3.12.14
for their build/test tasks. All other recipe bytes match the retained catalogue
references. The final dry run contains only the Python 3.12.14 module; the
intermediate older interpreter graph remains historical evidence. Source
archives and copied patch hashes are verified, but native builds remain pending.
The inherited OpenBLAS recipe also tolerates up to
150 LAPACK errors; this candidate neither changes that policy nor claims a
strict native numerical qualification. Native source/test closure needs its
own reviewed evidence. Python 3.12.14 bundles Expat 2.8.3 and defaults to that
copy; the selected EasyBuild block does not request system Expat. Actual
installed linkage still requires verification. The OpenSSL system wrapper
now has the verified 3.5.8 source fallback. Wrapping a system library still
needs vendor package and security-patch provenance; its version alone cannot
establish that status. No system wrapper has been installed by this packet.
The cURL, libarchive and CMake overrides change only their OpenSSL dependency
from the 1.1 wrapper to the 3 wrapper. Their downloaded sources contain OpenSSL
3 compatibility branches; actual compilation and linkage remain unverified.
This removes the old 1.1.1w fallback from the foundation graph without claiming
that every inherited native package is security-qualified.

The remaining Pydantic and JSON Schema families remain
pending. The xarray and scikit-learn recipes are resolved but have not been
natively built or loaded. Neither foss generation has an installed
Voiage stack established by this packet. Native Linux ARM64 builds, installed
module/CLI/numerical/Arrow checks and upstream review remain separate gates.
No x86-64, scheduler, production cluster or upstream acceptance is claimed.

## Scientific consumer providers

`Voiage-scientific-consumers/2.2.0-gfbf-2023a` supplies the same checksum-pinned
xarray 2024.11.0, scikit-learn 1.7.2, joblib 1.5.3 and threadpoolctl 3.6.0
provider set on the 2023a foundation. Joblib uses an isolated setuptools 84
build-only provider while the foundation setuptools 70 provider remains intact.
The build requirements are recorded in
`scientific-consumer-sources.json`; the corresponding EasyBuild 5.4 robot dry
run is retained in `evidence/scientific-consumers-robot.log`. This evidence is
dependency resolution only. No native package build, installed module smoke,
full Voiage graph, cluster qualification, or upstream submission is claimed.

The adjacent `easybuild-2023a-polars-overlay` supplies checksum-bound Polars 1.42.1 and `polars-runtime-32` recipes with the exact dated nightly Rust compiler. Its robot result is source-resolution evidence only; native and full-graph qualification remain false.
