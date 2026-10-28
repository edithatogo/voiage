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

Each generation copies the entire prepared source cache with `copy2`, constructs
a fresh install/build/home tree, disables existing modules and downloads, and
uses EasyBuild robot dependency resolution. This tests a complete isolated
scientific stack rather than only the Voiage extension. Both generations must
pass sequentially, with distinct prefixes and linkage/isolation evidence. The
250 GiB check runs again before the second generation, while the first run's
retained trees still occupy space; it is not a combined two-generation budget.
The driver itself has no intermediate-cleanup policy. EasyBuild's own cleanup
behavior must be verified before attributing retained build-tree size to it.

## Decision and evidence needed

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

Execution results are pending. The outcome must be read from the actual run,
its per-step exit codes, resource receipt and installed probe, not the workflow
name or a started GitHub job.
