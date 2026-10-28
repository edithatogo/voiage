# Arrow source-build capacity experiment — 29 October 2026

## Lane ownership and handoff

Writing chat: `01a17093-d6f6-7302-a1fa-a868618ac648`
(`Voiage — Build capacity — Measure native stages`). Its isolated checkout is
`/Volumes/PortableSSD/GitHub/.codex-worktrees/voiage-arrow-capacity-20261029`,
branch `codex/arrow-build-capacity-20261029`. The lane owns only the new Arrow
capacity workflow, `scripts/observe_arrow_capacity.py`, its focused test file,
the frozen Arrow input JSON and this report. The first source-run commit is
`2233b838`, run `40704302172`. The corrected source-run commit is
`6e2249955b3e431ae0d0ce7dfe26bba786849800`, run `40709591308`.
Native scope is only those GitHub runs' private ephemeral builders.
The next operation is read-only result/artifact collection, followed by one
terminal evidence handoff. Current providers, private environments/bindings,
shared handoffs, runtime/dependency promotion and main integration remain with
the Voiage integrator chat `01a17ab7-148e-7042-be1e-03a9478643da`.

The source was committed and pushed before the run. The app's managed-worktree
attachment attempt returned `The checkout exists but is not a managed worktree`.
The clean isolated source checkpoint is preserved; no other checkout or native
operation was moved or interrupted. Its exact scoped commits are available for
the integrator's managed-worktree delivery at the terminal handoff. Local
result processing uses this worktree's own verified Python 3.15.1 environment.

## Experiment contract

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

The first run's source compilation and functional probes passed, but its
measurement outcome failed; see the retained first-run evidence below. A
corrected measurement run is required before reporting complete qualification.

Read-only agent review found one issue: the initial receipt described two build
jobs without accounting for bundled dependency parallelism. The receipt and
scope above now describe actual enforcement, with generated commands retained.
Local checks passed: Ruff lint/format, ty, actionlint, script/input parsing,
repository harness and 18 focused provenance/harness tests. The frozen input
packet was compared byte-for-byte with the actual committed provider manifest.

## Additional configuration finding for the provider owner

The pinned Spack builtin Arrow builder emits top-level `BUILD_SHARED` and
`BUILD_STATIC`, with CMake's generic `define` passing those names verbatim.
Authenticated Arrow 26 `DefineOptions.cmake` instead defines
`ARROW_BUILD_SHARED` and `ARROW_BUILD_STATIC`, both defaulting ON. Its
`BuildUtils.cmake` lines 255–264 set function-local `BUILD_SHARED/STATIC` from
per-target argument overrides or those prefixed global variables. The inherited
unprefixed cache flags therefore appear ineffective for this source version.
This can leave static duplicate output enabled in a nominal shared profile;
the actual Spack generated arguments/cache and intended policy still need
qualification. The finding was handed to the existing provider owner, without
changing that owner's source. This pilot's explicit prefixed shared-ON/static-OFF
settings remain a distinct measured profile, not proof of the inherited builder.

## First run and measurement corrections

Run `40704302172`, source `2233b83845f1bd1c528f3ea46deea8b535627b2a`,
compiled and installed Arrow C++ and PyArrow 26.0.0. The installed CSV, compute,
IPC, Parquet, dataset, six-codec and mimalloc probes all passed. C++ compilation
took 653 seconds, PyArrow 127 seconds, and total elapsed was 848 seconds. The
170 filesystem samples show 1.49 GiB maximum additional allocation. This is a
scoped measured disk observation, not the full matrix's required capacity.

The overall run correctly remained red because the final linkage assertion
failed. Its object scan selected `lib*.so*`, omitting `_parquet` and other
underscore-prefixed extension modules. The corrected scan selects all native
objects inside the actual installed PyArrow package and requires exact-prefix
Arrow/Parquet linkage. It does not loosen the provenance assertion.

The initial process-group-only guard also under-counted Ninja's compiler
children in separate process groups. Its 20 MiB C++ RSS figure is invalid as a
whole-build memory measurement. The capacity observer now additionally samples
the complete PPID descendant tree every two seconds and enforces the named
11 GiB aggregate limit across those descendants. Linux process handles bind
termination to that build's processes. The shared default resource guard is
unchanged. Dedicated regression cases cover underscore extensions and children
with different process groups. Original first-run receipts and logs remain
under ignored `.conductor/local/arrow-capacity/40704302172/`; they are not
rewritten as a passing measurement.

The second run `40709591308` again passed source compilation, installation and
all functional probes. The corrected descendant sampler recorded 1,892,212 KiB
(1.80 GiB) peak C++ process-tree RSS, and its 1.49 GiB sampled additional disk
use is consistent with the first run. Its overall workflow remained red because
`*.so*` additionally matched the non-ELF Parquet fixture
`v0.7.1.some-named-index.parquet`. Native Arrow/Parquet linkage itself resolved
to the fresh prefix. The scanner now requires both an actual shared-object
filename and ELF magic, with that precise fixture and a text linker-script
negative control added to the regression case. Original second-run artifacts
remain in `.conductor/local/arrow-capacity/40709591308/`.
