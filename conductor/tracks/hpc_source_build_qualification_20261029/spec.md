# Rust/LLVM source builds and full HPC qualification

## Overview

Type: chore / build qualification. Create a reproducible Linux source-build
qualification path for the complete current Voiage HPC graph, with measured
capacity and retained native acceptance evidence. Planning was requested on
29 October 2026; this document does not claim implementation or build success.

This is the execution breakdown for the remaining HPC gate in
[remaining backlog delivery](../remaining_backlog_delivery_20260831/index.md),
including issue #1025. It does not supersede unrelated backlog work or duplicate
completed component pilots. Parent closure must consume this track's acceptance
receipt; neither record may declare the native gate complete independently.

## Requirements

1. Freeze the current source manifest, provider commits, package-manager versions,
   compiler roles, flags, target platform and complete graph before execution.
   Use Python 3.15.1 through a verified repo-local interpreter. Refresh dependency
   candidates only in the owning lane; historical graphs remain historical.
2. Resolve the combined Spack stable/nightly/bootstrap compiler-role constraints
   using supported graph representation. The successful isolated macOS Polars
   concretization is diagnostic evidence, not Linux full-root qualification.
3. Qualify EasyBuild fetching with supported pre-fetch network restrictions and
   verification of the actual resolved source path and hash. Reject shadowing,
   missing or altered sources. Do not simply remove unsupported
   `--disable-download` and allow uncontrolled network access.
4. Build the required Rust compilers and LLVM from authenticated source. Record
   stage0 bootstrap binaries explicitly, source-built stages, targets, vendoring,
   LLVM options and compiler identity. A binary SDK pilot cannot satisfy this
   requirement. Distinguish stable and dated-nightly compiler roles, build each
   distinct required configuration, and use the resulting compilers downstream.
5. Profile individual bootstrap stages and full graphs: wall/user/system time,
   descendant RSS, CPU utilization, I/O, disk growth and per-directory final size;
   record source/cache/build/install footprints separately and cold/warm state.
   Reuse the corrected descendant sampler, supplement with GNU time and Linux
   cgroup peak/OOM counters where available. Document interval, shared-page
   double-counting, missed peaks and unsupported counters. A process-group-only
   sample is insufficient for Ninja children in separate groups.
6. Use focused CPU profiling (`perf record`/`perf stat` or a supported equivalent)
   and compiler timing reports on a measured costly stage. Use allocation-stack
   profiling only when evidence identifies allocation pressure. Qualify permission,
   tool overhead and command support first; report unavailable profilers honestly.
   Sampled capacity evidence alone must not claim hotspot attribution.
7. Evaluate avoidable overhead from duplicate bootstrap stages, target sets,
   debug symbols, static/shared duplication, parallelism and cache/source copies.
   Verify Arrow's actual ARROW_BUILD_SHARED/ARROW_BUILD_STATIC cache and artifacts.
   Change only justified unnecessary work while retaining promised capabilities;
   compare an unchanged-input baseline and candidate with acceptance checks.
8. Run the full Linux Spack install and both maintained EasyBuild foss/2023a and
   foss/2024a graphs, sequentially where capacity requires. Include installed
   module imports, numerical fixtures, Arrow round-trips, Polars parity, actual
   library linkage and generation isolation. Preserve the supported feature set.
9. Prefer a bounded GitHub-hosted feasibility run, then use measured evidence to
   select execution capacity. Check current runner limits and available resources
   at dispatch. No assumed 250 GiB floor, silent support reduction, paid resource
   provisioning or local SSD exhaustion. A capacity stop remains an incomplete
   gate; propose a concrete alternative if a free runner cannot finish.

## Acceptance criteria

- AC1: Complete Linux graph resolves with exact sources and explicit compiler
  roles; unresolved solver constraints cannot be replaced by patched diagnostics.
- AC2: Fetch controls pass negative tests for network attempts, source shadowing,
  corruption and missing input, with retained actual source-resolution evidence.
- AC3: Every distinct required Rust/LLVM source configuration builds and installs,
  passes compiler/link probes, and is consumed by the downstream graph. Bootstrap
  SDK exceptions are enumerated and never counted as source-built output.
- AC4: Each stage has a terminal resource receipt. The capacity recommendation
  derives from observed peaks, concurrency and explicit headroom, and separates
  transient build size from retained installs. Any design changes have comparative
  evidence or a documented decision to preserve the existing design.
- AC5: Spack, EasyBuild 2023a and EasyBuild 2024a each have successful native
  installation and all required installed-consumer tests on their exact graph.
  A skipped, timed-out, OOM-killed or missing cell keeps the track incomplete.
- AC6: Correctness and security/automation agent reviews are complete, fixes and
  affected checks pass, required local/hosted gates pass for the integrated tree,
  and durable sanitized receipts survive hosted artifact expiration. The
  integrator reconciles the parent gate, merges and only then archives.

## Evidence and operational constraints

Each receipt records commit/tree, source digests, tool versions, actual command,
working directory, platform/runner, compiler identities, flags, concurrency,
cache state, timestamps, exit/signal/stop reason, resource method and artifacts.
Retain failed attempts alongside successful ones. Cache reuse requires exact
source/configuration keys and must not hide a missing clean-build witness.
Only clean owned temporary outputs may be removed after durable receipt capture.

Do not disturb active native jobs or change another lane's environment. Keep
normal hooks, repository security/coverage gates and third-party licences.
No human independent-owner review is required in this single-person repository;
use agents/subagents for technical review, with the existing integrator retaining
merge and shared-state ownership.

## Out of scope

Scientific algorithm changes, new hardware support, public package releases,
external registry submission, paid cloud activation, broad storage cleanup and
repeating unchanged completed Arrow/Polars pilots. Profilers are investigation
instruments, not a new general observability framework.
