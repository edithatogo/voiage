# Build capacity: implementation and scope decision

Status: instrumentation locally verified and pilot evidence retained; full HPC capacity qualification remains open. The obsolete fixed admission floor has been removed from the current replay candidate.

## Ownership and evidence

The capacity integrator owns this track extension, G04.4/G04.5, the verification
guard and its tests on `codex/build-capacity-conductor-20261028`. Arrow providers
remain with the primary checkout's existing owner. The independent pilot remains
owned by `codex/native-build-capacity-20261028` in the external native-capacity
worktree. Do not change its workflow, uncommitted assessment or running jobs.

Pilot workflow and committed assessment baseline:
`dc5e71525e120183e97611bb8c40d9c1c9b00e08`. The investigator's later uncommitted
assessment is provisional. The [terminal run](https://github.com/edithatogo/voiage/actions/runs/40691333092)
completed successfully; all build and installed-native-probe steps exited zero.
[Raw receipt](./capacity-evidence/receipt.json), samples, source inputs and
resource report are retained with [SHA-256 bindings](./capacity-evidence/provenance.json).
They are observations, never instructions or authority to change support.

The Polars 2.1.0 native runtime was compiled from its exact source with upstream
full/nightly defaults, two Cargo jobs and the declared nightly SDK. Python was
3.15.1. Observed filesystem growth peaked at 10,475,061,248 bytes (9.76 GiB);
starting free space was 85.57 GiB and minimum free space was 75.81 GiB. Overall
elapsed time was 1,291.15 seconds; the runtime guard sampled 5.48 GiB peak RSS.
The installed frontend/runtime version and native sum probe passed. Five-second
samples can miss spikes and include unrelated activity; rounded directory sizes
and shared blocks are not exclusive allocation. No compiler was built from
source. This is not Arrow, Spack, EasyBuild or full-matrix acceptance.

## B6-CAP1: support-scope policy accepted — 29 October 2026

The live [issue #1025](https://github.com/edithatogo/voiage/issues/1025) requires
current-security Spack and both EasyBuild generations, with installed numerical,
Arrow, Polars, linkage and module probes. Current `packaging/current/README.md`
keeps old native helpers historical. The current Rust provider explicitly
disables downloaded LLVM; the dated compiler disables downloaded rustc too.
The replay retains the first generation's root for the second generation.
Use workload-specific guarded measurements rather than the obsolete fixed
admission floor; the scoped Polars observation is not a combined build budget.

| Workload | What it proves | Capacity state |
| --- | --- | --- |
| Ordinary full tox, installed dependencies | Repository quality and application behaviour | Current instrumented calibration pending |
| Source Voiage/Polars with authenticated SDK | Native application/dependency build and execution | Polars-only pilot passed; Voiage source lane separate |
| Source Rust/LLVM compiler | Compiler provider build and installation | Pending current graph and bounded measurement |
| Complete Spack and two EasyBuild generations | Scientific stack, modules, linkage and isolation | Pending provider/solver and native qualification |

Accepted user policy: use authenticated compiler SDKs for ordinary development and
bounded dependency pilots, while retaining a separately labelled source
compiler/full-stack qualification lane. Do not force every ordinary test run to
rebuild toolchains. Retain both EasyBuild generations and full source-built compiler qualification.
No generation is retired. Site modules remain an assessed option requiring
separate qualification; they are not an accepted replacement for source builds.

| Option | Benefits | Costs and acceptance implications |
| --- | --- | --- |
| Retain complete source toolchains | Source provenance and explicit build portability | Largest storage/time burden; compiler/LLVM builds still need measurement |
| Reuse authenticated SDK | Smaller repeat builds; fixed upstream compiler identity | Platform availability, binary provenance and licences must be checked; cannot claim compiler-source qualification |
| Reuse qualified site modules | Matches an actual cluster and avoids duplicate toolchains | Site availability, ABI/linkage and configuration must be pinned; less portable evidence |

For every option retain licensing/notices, source and binary identities, CPU/ABI
and compiler feature configuration. Compare numerical and performance witnesses;
no performance equivalence follows from a smaller build. The explicit user decision in chat `01a1828e-c05a-7752-81a8-3a89b970678f`
on 29 October retains the existing issue #1025 criteria and compiler-source
obligations. No support migration or rollback is needed; any future reduction
requires a new decision and qualification.
No new purchase, infrastructure reconfiguration or upstream submission is implied.

## Execution sequence and acceptance

1. G04.4 adds disk observations to the existing guard. Test device deduplication,
   low-space stops, monitoring failures, short jobs, artifact hardlinks/symlinks,
   retained failure diagnostics and compatibility with existing report fields.
   Optional selected directories are scanned before/after only; per-interval
   samples use filesystem free space. Missing or failed evidence never passes.
2. G04.5 runs the final ordinary gate with selected artifact directories and
   declared cache state. Bind source, lock, coverage configuration, features,
   runtime and exact command. Preserve older observations; never purge shared
   caches to manufacture a cold result. A second warm observation uses unchanged
   inputs; repeat only to resolve this explicit calibration requirement.
3. B6-CAP2 retains the completed SDK pilot, then separately measures compiler,
   Arrow and full graph after their current providers and scope are qualified.
   Require both guest and host-backing evidence for VM capacity. The guest's
   virtual capacity cannot substitute for physical space.
4. B6-CAP3 qualifies a checksum-verified read-only source store using the selected
   builder's supported source-path mechanism. If unsupported, qualify filesystem
   copy-on-write staging; otherwise preserve existing copies and record the gap.
   Test changed checksums, attempted mutation, independent install/build roots
   and fallback before adoption. Do not edit historical replay recipes/helpers.
   Verify existing cleanup first; avoid duplicating EasyBuild's successful-build
   cleanup. Fresh consumer environments remain mandatory; use the existing
   explicit download-cache option only. No shared build roots or cache deletion.
5. B6-CAP4 publishes budgets only after representative successful measurements
   and B6-CAP1 decision. Include starting retained data, transient additional
   growth, final retained output and explicit reserve in GiB. For generation two,
   count generation-one retained outputs. Cold/warm and platform results remain
   separate. Failed pilots provide causes/lower bounds, never a full budget.

B8 compares optimisations on matched source/dependencies/compiler/features and
requires unchanged numerical, Arrow, linkage, installed-module and isolation
behaviour. B12 inventories stale targets, environments and failed scratch trees,
then applies its existing recovery and exact-target cleanup rules. Inventory
alone authorises no deletion. Compiler stages, debug features and supported
versions require separate decisions and qualification.

## Measurement limits and rollback

Filesystem free-space deltas include other processes. Directory snapshots count
allocated blocks, deduplicate hardlinks within each root, skip symlinks and other
mounted devices, and cannot establish physical-exclusive bytes for clones or
compressed storage. Overlapping roots must not be summed. Sampled peaks are lower
bounds; no hard capacity guarantee is inferred from them. Headroom must be stated
and justified independently of the observed peak.

Keep failed-run diagnostics, immutable source receipts and original installed
generations. Revert only this slice's instrumentation/configuration commits if
needed; preserve observations. Source-store adoption remains a future owned
native slice until the builder is available and its tests pass.

## Ordinary verification observation

The first instrumented full tox run on `09f3bc76` used a private download cache
seeded by the frozen bootstrap, fresh tox environments and the existing explicit
consumer cache reuse option. Cache state is mixed, not cold. Its sampled volume
free-space drop was 2.28 GiB, which includes concurrent activity and submodule
bootstrap; it is not exclusive job allocation. All raw directory snapshots remain
in [the resource receipt](./capacity-evidence/ordinary-full-tox-resource.json).

The run had 5,497 passing tests, 19 skips and one metadata serialization failure;
coverage was 94.73%. The docs lane also failed because the fresh worktree lacked
its pinned Astro submodule. These first failures are preserved. Focused repairs
are recorded separately in [verification](./capacity-verification.md); they do not
retroactively make the first run a passing capacity benchmark. A representative
successful cold/warm calibration and justified reserve remain G04.5 work. No
ordinary full-gate or HPC hard capacity threshold is introduced.

## Ordinary calibration continuation — 29 October 2026

The budget helper `scripts/build_capacity_budget.py` requires successful cold and
warm terminal receipts with identical clean source, interpreter, configuration,
command and filesystem identities. It rejects failed monitors, dirty/untracked
source, changed inputs, impossible space arithmetic and duplicate devices.
It preserves directory snapshots individually and adds an explicitly justified
reserve to the largest observed additional filesystem use. It never adopts a
hard admission limit or certifies exclusive physical capacity.

Cold scope for this calibration is a newly created private uv cache and fresh tox
environments in `.conductor/local/capacity/calibration-20261029/`. The warm run
reuses those exact paths. Existing documentation/node and Rust targets, root
bootstrap environment and OS caches are retained for both runs and declared in
the fixed identity packet. This is dependency/environment-cache calibration,
not a claim that all host caches or source toolchains start cold. The two full
runs gather this previously missing pair; neither is a repeat for a status poll.
Each uses the unchanged `tox run-parallel -p 2` gate, no selection/retry flags,
and an independent result transcript. Stop on the first failing run and preserve
its receipt; never launch the warm run after a failed cold run.

Both runs completed successfully on macOS ARM64 at clean source revision
`50b3ff67b22514a637baf78129dbba5a7ee21cbc`. The cold run observed a 4.282 GiB
sampled free-space drop and 4.273 GiB ending drop; the warm run observed 0.770
GiB and 0.726 GiB respectively. These are filesystem-wide changes, not exclusive
allocation or measured peak directory sizes. The warm run overlapped small
source-staging experiments on the same volume. Samples can miss spikes, and
existing tox, Rust and documentation artifacts were retained at the start.

The pair does **not** yet qualify a matched budget: the tox package inventories
refer to separately numbered temporary Voiage wheel paths. Those wheels were
removed by tox before their hashes could be compared. Other installed-package
entries match, but matching wheel filenames and source identity do not establish
identical built artifacts. The budget helper correctly rejects the pair. Future
calibration must retain or hash each built wheel during execution and compare
its identity before interpreting a pair. No new threshold is set.

Retained raw evidence (tox SHA-256 values are for the decompressed JSON): [cold resource](./capacity-evidence/ordinary-cold-20261029-resource.json)
(`7618cdcde7ea2e30237e56186b5357a0e1336c26338beefa1c100d2dfe88a12e`),
[cold tox](./capacity-evidence/ordinary-cold-20261029-tox.json.gz)
(`c0d67cb38de881ae56d42e8c4dde3266122962b53c81517ea7cc675788848dda`),
[warm resource](./capacity-evidence/ordinary-warm-20261029-resource.json)
(`919158fbb1b91c3fb1571ed21671e85b1f10b7acd680044a055f2ed4bf3734af`), and
[warm tox](./capacity-evidence/ordinary-warm-20261029-tox.json.gz)
(`17a634ad4a5512626fbcc9630f513e97a287c55b922d0ed55ca44d9d52fa289b`).

Source-staging controls separately confirmed macOS `clonefile(2)` isolation and
read-only source mounts in offline ARM64 Linux containers using the verified
Polars source archive. Linux `--reflink=always` was unsupported on the tested
bind mount, while its ordinary-copy fallback succeeded. These controls do not
prove EasyBuild support, host-side immutability, actual physical savings or HPC
acceptance. [Control receipts and checksum manifest](./capacity-evidence/source-staging-control/manifest.json)
are retained. The historical EasyBuild replay remains unchanged; source-path
resolution and offline enforcement require qualification in the current builder.
## Storage inventory for B12 review (29 October 2026)

This is a scoped inventory of the capacity worktree, not authority to delete.
Sizes below are allocated-directory snapshots from the retained warm receipt;
they can include shared blocks and do not prove recoverable physical bytes.
B12 must establish ownership, live use, reproducibility and preservation before
any exact-target cleanup. In particular, keep active operations and raw evidence.

| Target | Snapshot | Proposed review |
| --- | ---: | --- |
| `.tox` | 7.74 GiB | Compare with the private calibration tox environments after the final gate; preserve until its owner verifies no active use and recovery. |
| `.conductor/local/capacity/calibration-20261029/tox` | 7.73 GiB | Active final-gate workdir: preserve. Review only after the terminal receipt and handoff. |
| `.conductor/local/capacity/calibration-20261029/uv-cache` | 1.07 GiB | Active final-gate cache: preserve. Review later under cache preservation rules. |
| `rust/target` | 0.15 GiB | Review stale build outputs against active Rust work and reproducibility; no blanket Cargo cleanup. |
| `docs/astro-site/node_modules` | 0.28 GiB | Preserve as the active documentation dependency environment until its owner verifies replacement. |
| `.conductor/local/capacity/astro-bootstrap-before-init` | not measured | Inspect failed bootstrap scratch and retain diagnostic evidence before proposing exact targets. |
| `.conductor/local/capacity/native-arrow/40704302172` and `40709591308` | not measured | Failed pilot evidence; retain receipts and diagnostics. The native-capacity owner decides any future cleanup. |

Cold/warm snapshots and their checksums are linked above. The listing excludes
other checkouts and shared caches and is not a whole-machine inventory.
## Arrow/PyArrow pilot — 29 October 2026

The separately owned [Linux x64 source-build pilot](https://github.com/edithatogo/voiage/actions/runs/40713503189) at `1db97ce0c6cadd95cafa45a716f453c488cb9c7e` succeeded. Arrow C++ and PyArrow 26.0.0 were built using the declared compiler SDK and Python 3.15.1; the isolated installed probe exercised CSV, IPC, Parquet, datasets, compute and codecs, and the native-linkage check passed. Its receipt reports 1,597,198,336 bytes (1.49 GiB) maximum sampled free-space drop at five-second intervals. It records a two-job top-level build, while bundled dependencies may use their own processor-count defaults. Filesystem observations include unrelated activity and can miss short peaks; this is one profile, not an Arrow worst case, compiler-source build, Spack acceptance or full HPC capacity budget.

The investigator handed off [the frozen receipt](./capacity-evidence/arrow-owner-receipt-20261029.json), originally `docs/reviews/arrow-capacity-receipt-20261029.json` at commit `3cac43126e56fe254095eb11764a6466b3d019c8` (SHA-256 `36f02dc2c18220147202f0e1abc5cd0dbfaa243badc8fe2fe0ff24d8257d8908`). The run artifact `arrow-capacity-40713503189-1` contains stage logs, source/configuration inputs and resource observations; its ZIP SHA-256 is `e2da44e3ce78cf562b5e2aabe03acd8b9fa62fe5ae72ed3050d5eb379bf948a5`. Peak descendant RSS was 2.78 GiB in PyArrow and elapsed time was 1,165.23 seconds. Native-capacity ownership remains with the investigator. Previously failed runs and their linkage diagnostics remain historical evidence.

## Final-candidate gate observation

The full local gate on `6e6a77cd42dc03e67425086c24b21064d80e17d2` failed.
Lint and coverage found active-text reference-policy matches in the newly
retained raw tox transcripts and a test fixture. The minimum-version suite
also recorded a Hypothesis timing flake: 285.80 ms on the first call and
0.30 ms on replay against a 200 ms deadline. The original result is preserved
at `.conductor/local/capacity/final-candidate-6e6a77cd/` (log SHA-256
`ca387b27a44a50b65c644187d635c0f627a14ee09933f0b5a5dd91b0ec64e155`,
tox-result SHA-256 `fbf8d140f1a477ecf3887adafdd2910877f4420f40119b409066eaad398bdcef`).
The historical tox transcripts are now losslessly compressed, with their
decompressed checksums retained above; the fixture reads the actual tox matrix.
The subsequent candidate requires its own full gate and hosted checks.
