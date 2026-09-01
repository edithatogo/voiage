# Arrow 25.0.1 provider candidate for foss 2024a

This isolated overlay adds Arrow and PyArrow 25.0.1 to the Python 3.12.14
foundation. Its seven easyconfigs resolve a 78-module dependency graph with
EasyBuild 5.4.0 and the pinned catalogue. Resolution is not a native build.
The complete Voiage graph and both requested Linux toolchain builds remain
unfinished.

Apache's [25.0.1 release](https://arrow.apache.org/release/25.0.1.html) fixes
incorrect Parquet double values on ARM64 SVE and an allocator crash in 25.0.0.
The original Apache source archive matches its published SHA-512. The selected
Thrift 0.22.0 and mandatory xsimd 14.2.0 archives also match the hashes in that
Arrow source. No Arrow or NumPy requirement is patched.

## Required providers

The actual C++ and Python CMake projects require CMake 3.25 and C++20.
GCC 13.3.0 and CMake 3.29.3 meet those requirements. Explicit dependencies
supply compression libraries, Boost, RE2 with Abseil, UTF-8 processing,
RapidJSON, Thrift and xsimd. The existing scientific module provides NumPy
2.2.6, within PyArrow's unchanged requirement of at least 1.25.

PyArrow now uses its own wrapper around scikit-build-core. Its build requires
Cython at least 3.1, LibCST at least 1.8.6 and setuptools-scm at least 8.
Cython 3.1.8 and a separate build support module satisfy those requirements.
The scientific foundation retains Cython 3.0.10 for its own builds and
setuptools-scm 7.1 for dateutil. The newer build support module is not an
Arrow runtime dependency.

A separate macOS Python 3.12.13 environment source-built the backend
providers and LibCST, imported the actual PyArrow backend, and passed pip
dependency checks. Bootstrap wheel/setuptools/flit-core wheels were used
only to initialize that environment, then replaced from verified sources.
This is not the selected Python 3.12.14 Linux installation. A real Modules
fixture restored setuptools-scm 7.1 after loading and unloading the build-only
9.2 provider; generated EasyBuild module loading still awaits native builds.

The support module includes scikit-build-core 0.12.2,
setuptools-scm 9.2.2, setuptools-rust 1.9.0, semantic-version, PyYAML,
pathspec and Trove classifiers. The current classifier list is required:
the older foundation list rejects scikit-build-core's free-threading
classifier. Validation remains enabled.

LibCST uses `CargoPythonPackage`, the actual EasyBuild block that prepares
vendored Cargo sources before pip installs the Python extension. All 95
registry archives match its source `Cargo.lock`; offline locked metadata
resolution includes those archives and two workspace packages. That local
check used the host Cargo 1.98, not a built instance of the selected Rust 1.96.
The stable Rust source provider preserves the separate compiler identity and
its versioned stage-zero bootstrap boundary. It does not select the dated
nightly compiler required by Polars.

## Features and source controls

The candidate enables Compute, Acero, Dataset, Parquet, CSV, JSON, local
filesystem and IPC. It uses system-provided dependencies instead of Arrow's
automatic source fallback. Pip cannot download dependencies or create an
isolated build environment. Optional cloud connectors, Flight, Gandiva,
Substrait, ORC and alternative allocators are disabled for this local
interchange profile. They are not claimed as supported features.

The installed sanity command checks 257 double values across four Parquet
codecs, NumPy equality, CSV and IPC round trips, Dataset filtering, Acero and
local filesystem access. A successful run against the existing public wheel
checks that command itself; it does not verify installation by EasyBuild.
Arrow's full native upstream suite and native performance remain unverified.
The xsimd header provider does not enable its optional test, benchmark,
example or xtl-complex dependencies. Thrift retains its C++ unit-test target;
other language bindings and optional transports are outside this profile.

`source-manifest.json` binds 116 downloaded archives, including the selected
native sources, Python backends and LibCST crates. It does not claim that
every source in the shared compiler and scientific foundation has been built
or downloaded. `evidence/source-members.json` binds extracted upstream
requirements. `manifest.json` binds the candidate's recipe and evidence bytes.
The older foundation manifests and historical build records are unchanged.

## Reproduce resolution

Use a fresh installation prefix, a functioning Environment Modules 5.6.1
installation and EasyBuild framework/easyblocks 5.4.0. Set
`EASYBUILD_CATALOGUE` to the catalogue revision recorded in `manifest.json`.
Place the Arrow overlay before the foundation and catalogue:

```sh
eb packaging/easybuild-2024a-arrow-overlay/2024a/Arrow-25.0.1-gfbf-2024a.eb \
  --robot --dry-run \
  --robot-paths=packaging/easybuild-2024a-arrow-overlay/2024a:packaging/easybuild-overlay/2024a:"$EASYBUILD_CATALOGUE" \
  --prefix="$EASYBUILD_PREFIX" \
  --modules-tool=EnvironmentModules --module-syntax=Tcl
```

Real native compilation, module loading, the installed sanity command and
upstream acceptance still need separate evidence. No Linux VM execution or
upstream submission is recorded by this candidate.
