# Native build capacity assessment — 28 October 2026

## Findings from the existing build contract

The 250 GiB free-space floor and eight-CPU prerequisite were introduced by
`6e1351f7` (PR #1089) in `scripts/native_easybuild_qualification.py`. The original
PR and native qualification guide describe the floor but provide no disk-size
breakdown or measured peak. The receipt records only initial free space and CPU
count; it does not measure peak disk use or memory. Consequently, 250 GiB is an
enforced planning assumption, not an established minimum or measured footprint.

The driver is bound to source `9307d9ec`, catalogue `58e8b5a`, EasyBuild/easyblocks
5.4.0 and isolated foss-2023a/foss-2024a roots. Its recipes select Python 3.12.14,
Arrow 25.0.1 and Polars 1.42.1. These historical contracts differ from the current
Python 3.15.1, Arrow 26 and Polars 2.1 packaging work. Updating the current source
providers does not automatically update this immutable historical qualification.
`packaging/current/README.md` explicitly classifies the old smoke/native helpers
as historical replay tools. Applying their 250 GiB floor to the current source
lane without a current graph and sizing receipt is therefore a planning gap.

Each generation copies the entire prepared source cache with `copy2`, constructs
a fresh install/build/home tree, disables existing modules and downloads, and
uses EasyBuild robot dependency resolution. This tests a complete isolated
scientific stack rather than only the Voiage extension. Both generations must
pass sequentially, with distinct prefixes and linkage/isolation evidence. The
historical root-graph receipt resolves 108 modules for each generation; it is
dependency-resolution evidence, not evidence that those modules were built.
The graph includes GCC, MPI, numerical libraries, Python, Rust, Arrow and Polars.
The 250 GiB check runs again before the second generation, while the first run's
retained trees still occupy space; it is not a combined two-generation budget.
The driver itself has no intermediate-cleanup policy. EasyBuild's own cleanup
behavior must be verified before attributing retained build-tree size to it.

Large compiler builds are plausible: the upstream [Rust compiler build
guide](https://rustc-dev-guide.rust-lang.org/building/how-to-build-and-run)
warns that compiler builds, especially beyond stage 1, can consume around
100 GB. This is context for investigating compiler stages and debug/incremental
settings, not evidence that Voiage needs that amount or that 250 GiB is sufficient.

The current Spack Rust provider also deliberately sets
`llvm.download-ci-llvm=false`; its dated compiler additionally sets
`rust.download-rustc=false`, then invokes `x.py build` and `x.py install` without
an explicit stage. Thus the current native packaging work includes building the
compiler and LLVM backend, not merely compiling Voiage/Polars with a downloaded
compiler. Developer tools and docs default off, but the stage/build-tree budget
still needs measurement. Whether compiler SDKs or already-qualified site
toolchains satisfy the intended distribution contract is a useful design
decision; adopting either requires revising the explicit build policy rather
than representing the SDK pilot as source-compiler qualification.

## Decision and evidence needed

| Workload | What it establishes | Current sizing evidence |
| --- | --- | --- |
| Standard Voiage/Polars compatibility tests | Application behavior with installed dependencies | Focused GitHub job passed; no 250 GiB guard applies |
| Polars runtime compiled from source using its declared Rust SDK | Native dependency build and installed execution | GitHub pilot passed; 9.76 GiB sampled disk growth and 5.48 GiB peak sampled RSS |
| Rust/LLVM compiled from source | Compiler-provider installation | Current local declarations exist; measured native peak absent |
| Complete current Spack/EasyBuild scientific stack | Full graph, modules, numerical behavior and isolation | Current graph/provider completion and native measurements still required |

An eventual workspace budget must account for retained source/cache/install
bytes plus the largest concurrent build/stage/target footprint and documented
headroom. For generation two, include retained generation-one outputs. Measure
these terms rather than applying a fresh fixed floor without a total-capacity
model.

1. Resolve and record the exact current full dependency graph, required
   scientific consumers, compiler roles, toolchain generations and source-build
   policy. Keep standard wheel/consumer CI and full HPC qualification explicit.
2. Measure archive bytes, extracted sources, compiler build/stage/install trees,
   Cargo targets, scientific-library builds, installed modules and caches.
   Record peak physical allocation, free space, memory and elapsed time per
   stage, including the second generation and any retained first-generation data.
3. Review whether both older toolchain generations remain a maintained support
   promise. Do not silently remove an existing acceptance criterion.
4. Evaluate a single immutable source-download store, sequential ephemeral build
   roots and verified cleanup after retaining the required logs/inventories.
   Keep installed generations isolated and preserve necessary failure evidence.
5. Set any revised resource budget from observed stage peaks plus documented
   headroom. A failed bounded pilot provides a cause and a lower bound; it does
   not establish the full stack's minimum disk requirement.

## GitHub experiment

`native-build-capacity-observation.yml` runs a separate pilot on a standard
Ubuntu 24.04 GitHub-hosted runner. It downloads the exact checksum/size-bound
Polars 2.1.0 frontend and native runtime sources from the current provider
manifest, uses its declared dated Rust SDK, preserves upstream default features,
compiles the native runtime from source with two jobs, installs the frontend
from source and probes the installed native execution. Python is 3.15.1.

The pilot samples filesystem use every five seconds, retains compiler identity,
source hashes, build logs, process-group memory and resource-stop reasons. It
leaves at least 2 GiB free, limits the native build to 11 GiB sampled aggregate
RSS and 25 minutes, and caps the job at 35 minutes. Runner-wide disk deltas
include toolchain/cache changes and may include unrelated runner activity;
five-second samples can miss short peaks. Final directory sizes are retained
separately. No large binary/cache artifacts are uploaded.

This experiment uses a prebuilt Rust compiler SDK and only builds Polars. It
cannot establish Rust-from-source, Arrow C++/PyArrow, the entire scientific
stack, Spack installation, either EasyBuild generation or full matrix acceptance.
It leaves the historical driver's guard and criteria intact. The workflow is
manual, with a single experiment-branch push trigger to allow its initial run
without merging unmeasured tooling into the default branch.

### Executed result

[Run 40691333092](https://github.com/edithatogo/voiage/actions/runs/40691333092)
passed on tested commit `dc5e71525e120183e97611bb8c40d9c1c9b00e08`, based on
accepted source `e336cd9f`. Compiler setup, identity verification, runtime source
compilation, frontend source installation and the installed native sum probe
all returned zero. The wheel is `polars_runtime_32-2.1.0-cp311-abi3-linux_x86_64.whl`;
the installed probe reports Polars 2.1.0, `rt32`, UInt32 indexes and Python 3.15.1.
Upstream's optimized release profile with debug information was preserved.

| Measurement | Observed result |
| --- | --- |
| Runner | Standard Ubuntu 24.04 x86-64; 4 logical CPUs; 2 build jobs |
| Total experiment elapsed | 1,291.15 seconds (21 minutes 31 seconds) |
| Guarded runtime build elapsed | 1,275.63 seconds |
| Peak sampled additional filesystem allocation | 10,475,061,248 bytes (9.76 GiB) |
| Peak sampled process-group RSS | 5,746,572 KiB (5.48 GiB) |
| Initial / minimum free disk | 85.57 / 75.81 GiB |
| Filesystem samples | 259 at approximately five-second intervals |
| Final retained private workspace | Approximately 8.1 GiB |
| Cargo target / registry / uv cache | Approximately 5.7 GiB / 869 MiB / 1.5 GiB |
| Resource-stop result | `completed`, exit 0; no limit reached |

The runtime source archive was 3,878,004 bytes; the frontend was 811,508 bytes.
Source sizes therefore do not predict compilation footprint. The sampled
additional disk allocation includes SDK installation and workspace/cache growth,
while final retained sizes differ because successful temporary builds are
cleaned. This runner happened to expose substantially more free space than the
advertised standard-runner storage minimum; its observed capacity is not a
guarantee for future runners. The actual scoped build fits its measured budget,
but no fixed 10 GiB capacity guarantee follows from five-second peak samples.

The [uploaded evidence artifact](https://github.com/edithatogo/voiage/actions/runs/40691333092/artifacts/12818686617)
has archive digest
`sha256:cacbd2689f41395b1c28e585a71e7e55630cc1b60eb052162b3826d46915724a`.
Artifact retention is 14 days. The downloaded original logs, samples, receipts
and a checked digest inventory are retained locally under the experiment
worktree's ignored `.conductor/local/native-capacity/40691333092/` directory.
Readback independently checked every step exit, resource completion, the native
runtime marker, manifest digest and the minimum-free/maximum-growth arithmetic
against the retained samples.

Digests of the key original evidence:

- Provider/source manifest SHA-256:
  `614606208dbd605d5d48ffa699aa24eed15052a46bdc71ccdbd80a6eaf19223c`.
- Receipt SHA-256:
  `3b1fed9a22c0ed50f282a3f513d0423fdd629e32d76049976b5a17745d49ed2d`.
- Resource report SHA-256:
  `2a32a3f7b15b85f70c845ed0d3e2191f9bcd5c825fba820f375470866585d899`.
- Native build log SHA-256:
  `80fe71bed3b9180182918d3bcbf87d4ddb2f36a28b69135d6bed58c86c283194`.

### Resulting design recommendation

Use standard GitHub-hosted jobs for the current Polars source/SDK build and
ordinary compatibility checks. Prepare the current full dependency graph and
size Arrow, compiler bootstrap and complete HPC installation as explicit
subsequent workloads. Review whether every supported HPC run must reconstruct
Rust/LLVM, and evaluate verified compiler SDK/site-toolchain reuse where the
distribution contract permits it. Preserve both supported EasyBuild generations
and existing full-native criteria until a concrete support-policy revision is
accepted. The 250 GiB estimate is neither necessary for this successful Polars
build nor established as the correct budget for those remaining workloads.

The separate dependency preview's `polars-current` job
[122815426594](https://github.com/edithatogo/voiage/actions/runs/40688048158/job/122815426594)
passed on Python 3.15.1 with Polars 2.1.0. Its package metadata check reports all
80 installed packages compatible; all three focused interchange/schema/numerical
test files passed. This is installed-dependency application evidence, separate
from the pilot's native dependency compilation.

Local workflow validation: actionlint, YAML/embedded-Python parsing and the
repository harness passed; its 11 focused harness tests passed. These checks
qualify the experiment's structure, not its unexecuted native acceptance gates.
