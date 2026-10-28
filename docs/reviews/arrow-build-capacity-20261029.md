# Arrow source-build capacity experiment — 29 October 2026

The successful Polars pilot established a measured source-with-SDK workload
budget. The next independent workload is Arrow C++ and PyArrow 26.0.0, retaining
the current provider profile's compute, CSV, dataset, filesystem, IPC, Parquet,
mimalloc and six compression libraries. The frozen input packet copies the
provider manifest at `fded766fe2fc863bdeb128bdb5b4d173bafa6c35`, while the active
Voiage agent retains ownership of those providers and their integration.

The experiment uses the hosted GCC/CMake/Ninja SDK and Python 3.15.1. It builds
the exact Arrow C++ Git archive, stages the checksum-bound mimalloc archive,
and builds PyArrow from its exact source distribution. Upstream bundled C++
dependencies are built where needed, using the pinned source tree's own
dependency declarations and checksums. This differs from Spack's SYSTEM
dependency policy and is explicit source-capacity evidence, not a Spack install.
NumPy and Python build tools are installed as SDK prerequisites; rebuilding
their entire toolchain/graph is outside this named workload.

The C++ build enables the complete positive provider profile, keeps shared
libraries, and omits tests/examples and static duplicate libraries. PyArrow's
Arrow/Cython C++ bundle settings retain their upstream defaults, OFF. The
installed probe checks CSV parsing, compute, IPC, Parquet, dataset operations,
codec availability and mimalloc; native linkage must resolve the newly built
Arrow libraries inside the isolated install prefix. No wheel fallback supplies
Arrow/PyArrow. The experiment retains its actual CMake cache, dependency and
build logs, installed probe, linkage and per-stage guard/resource receipts.

Top-level CMake/Ninja concurrency is two. Bundled dependencies retain upstream
processor-count defaults: for example, BZip2 can explicitly invoke `make -j4`
on a four-CPU runner. The receipt distinguishes the outer job limit from nested
dependency concurrency, and the generated Ninja commands are retained. Each
stage retains a sampled aggregate memory limit
of 11 GiB and at least 2 GiB free disk, with explicit stage timeouts and a
70-minute job cap. Filesystem samples run every five seconds, so brief peaks
can be missed. The measured runner-wide allocation includes SDK/cache changes;
it is not a guaranteed footprint across runners. Sampler failures invalidate
the observation. No compiler source build, full scientific stack, either
EasyBuild generation or full native acceptance is inferred.

Execution outcome is pending. Read the actual retained stage exit codes and
measurements before selecting a revised resource budget or reporting success.

Read-only agent review found one issue: the initial receipt described two build
jobs without accounting for bundled dependency parallelism. The receipt and
scope above now describe actual enforcement, with generated commands retained.
Local checks passed: Ruff lint/format, ty, actionlint, script/input parsing,
repository harness and 18 focused provenance/harness tests. The frozen input
packet was compared byte-for-byte with the actual committed provider manifest.
