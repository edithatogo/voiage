# Planning ownership and retained evidence — 29 October 2026

Planning writer: chat 01a17093-d6f6-7302-a1fa-a868618ac648, standard Git worktree
`/Volumes/PortableSSD/GitHub/.codex-worktrees/voiage-hpc-source-plan-20261029`,
branch `codex/hpc-source-qualification-plan-20261029`, base
`761e7496daad85223e6e430bb9de67f6ef26cc62`.
Owned changes: this new track directory and an isolated proposed registry entry.
No native operation is owned or started. Next operation is specification/plan
review, then integrator registration. This is not an app-managed worktree.
The earlier exact app attach attempt on the measurement lane returned
"The checkout exists but is not a managed worktree"; use the supported isolated
Git fallback, without changing private app state or copying dirty source.

Repository integrator/provider owner remains
`01a17ab7-148e-7042-be1e-03a9478643da`; capacity/budget owner remains
`01a1828e-c05a-7752-81a8-3a89b970678f`. Their existing assignments and native
operations remain in force. The integrator alone updates primary shared records,
provider/runtime promotion and main. This packet proposes a registry entry in
isolation; it does not take over the primary registry. At dispatch, reconcile
current claims and assign bounded implementation packets. Do not duplicate active
provider or budget changes. The current primary handoff is owned by the integrator.

## Evidence to reuse after exact-input reconciliation

- Polars completed SDK-backed source pilot, PR #1216, run 40691333092:
  https://github.com/edithatogo/voiage/actions/runs/40691333092
  Approximately 9.76 GiB disk growth and 5.48 GiB sampled process-group RSS;
  this older method is not the later descendant-tree measurement.
- Arrow PR #1217 original successful run 40713503189 at 1db97ce:
  https://github.com/edithatogo/voiage/actions/runs/40713503189
  Approximately 1.49 GiB disk growth and 2.78 GiB sampled descendant RSS.
  Curated receipt is in that PR's docs/reviews/arrow-capacity-receipt-20261029.json.
- Integrator's newer Arrow source run 40718827238 is successful according to
  live chat readback on 29 October. Its exact-head artifact verification and
  local gate remain owned by the integrator; consume its final receipt.
- Current primary implementation-handoff.md records the unresolved combined
  compiler graph, isolated macOS solve, source-only Cython/pydantic-core probes,
  Arrow flag finding and EasyBuild source-resolution problem. These observations
  must be reconciled to committed owner packets before native dispatch.

Neither SDK-backed pilot proves Rust/LLVM compiler source builds, a full Spack
install, or either complete EasyBuild generation. The old 250 GiB assumption was
withdrawn; it is neither a measured requirement nor an acceptance prerequisite.

## Integration boundary

Only plan files are created here. No runtime test or HPC build is claimed.
Validate local Markdown links, JSON, unchecked task markers and diff scope.
The integrator should use the on-demand integration preflight with this exact
packet range and the current target before applying it. Reconcile the registry
entry once, preserving unrelated dirty source and previous ownership history.

## Compiler requirement review — 1 November 2026

The compiler requirement review was prepared on branch
`codex/hpc-compiler-requirements`. It covers the two Spack recipes, focused
tests and source records. The integrator owns provider promotion, the combined
Linux solve and the shared registry.

[Compiler input record](compiler-inputs-20261101.json). Voiage and its build
backend need stable Rust. The Polars native runtime requires the dated nightly
to retain its upstream feature set. A binary SDK can build downstream packages,
as the component pilots demonstrated. It cannot verify this track's requirement
that the selected rustc and its LLVM backend were compiled from source.
The source-build path is authenticated stage0 SDK, vendored compiler sources
and LLVM, source-built rustc and Cargo, then downstream installed probes.
Record the actual stage numbers, host/target and linkage from each execution.

The stable source identifies the 16 April Rust 1.95.0 stage0. Its channel
manifest, full Linux x86-64 SDK, rustc, cargo and rust-std component bytes
matched the authenticated hashes. The reviewed recipe patch pins this
bootstrap for Rust 1.96.0 and disables compiler downloads during configuration.
The retained September nightly instead names the August beta. Historical April
nightly EasyBuild overlays are separate. The current system-toolchain
candidate describes neither complete foss generation. All three matrix cells
remain required.

Next, the provider owner should use the patched overlay for the unmodified
Linux root solve, retain the selected compiler edges and concrete lock, and
bind the two EasyBuild generation-specific graphs to that same runtime.
Source resolution on Linux must reject missing, corrupt and shadowed inputs
before any bootstrap dispatch. The accepted Arrow and Polars component
measurements are insufficient to set a full compiler-build budget.

The correctness/security review found that removing the bootstrap checksum
table would escape the original tests. The added test checks five upstream
manifest entries and the exact stable archive URL on each declared platform.
On Python 3.15.1, 38 affected provider and historical-boundary tests passed.
Spack 1.2.2 loaded both current classes in a private macOS configuration and
reported both applicable Rust 1.96.0 bootstrap constraints; their intersection
is exactly 1.95.0. This class load did not concretize a Linux graph.
The stable channel manifest SHA-256 comes from the authenticated source's
`src/stage0`; the five published SDK hashes can be regenerated from its
`pkg.rust.target` entries. Only the Linux x86-64 bytes were checked here.

The private Spack probe with builtin catalogue
`fa170d11f60ea44045d9a02370002ed153e35f1a` applied Spack's
`Spec.constrain` to the active recipe edges. It returned
`rust-bootstrap@=1.95.0` and the registered class contained that version.
For replay, read `src/stage0` from the digest-verified stable source, verify
the dated channel manifest against `compiler_channel_manifest_hash`, then
compare the SDK and all three component archives against the manifest's
`xz_hash` values. The [stable input record](../../../packaging/current/history/stable-rust-1.96.0-inputs.json)
retains those identities. The second agent review confirmed the regression
fix. The review proposed no change to the supported matrix.

## Source/vendor lane — 2 November 2026

Current writer is chat `01a195fd-b970-7391-83a9-9796bbf98ae3` in
`/Users/doughnut/Library/Caches/innovate-arrow26-qual.Q7IJ07/voiage` on
`codex/linux-source-vendor-qualification`, PR #1261. Owned scope is the source
qualification workflow/helpers/tests, current Polars lock repair and Git source
checksums, and the source/vendor receipt packet. The prior graph lane is merged.
No native compiler build or installed HPC operation is active. Next operation is
final hosted qualification, review, receipt retention and protected merge.


## Native compiler execution lane — 2 November 2026

Writer remains chat `01a195fd-b970-7391-83a9-9796bbf98ae3` in
`/Users/doughnut/Library/Caches/innovate-arrow26-qual.Q7IJ07/voiage`, branch
`codex/native-rust-llvm-qualification`, based on merged PR #1261. Scope is the
native compiler workflow, stage observer, source/lock closure, controls and
receipts. Earlier ownership assignments above remain historical.

Both accepted cold runs used `24ae6cb6`: stable `41436471133` and dated-nightly
`41436475314`. All 12 stages per role passed, including installation, installed
compiler identity, Cargo compile/link `EVPI=5`, and resolved native linkage.
There is no active native build. The default LLVM backend sets were preserved.
Thirteen manifests map through Cargo to 12 original authenticated locks; the
upstream vendor contract and copyright preflight fail before costly compilation.

The [results](native-results-20261102.md), [review](review-native-rust-20261102.md)
and [manifest](evidence/native-rust-llvm-20261102/manifest.json) retain exact Git
objects, source/bootstrap hashes, artifact digests, all stage/resource logs and
51 passing Linux controls. Initial observer, unsupported-stage0, missing std
workspace, missing notice workspace and shared-lock failures remain retained.
One uv manifest-download timeout received one bounded read-only job rerun.

The full local tox gate and affected docs/coverage rerun passed before the later
qualification-only fixes. A subsequent full repeat stopped at the disk-space
preflight; current focused and hosted checks provide the later helper evidence.
Only completed owned temporary outputs were removed after receipt verification.
No paid compute was used. No source-built compiler binary archive is retained.

PR #1262 is the integration boundary. Final-head checks, protected merge and
main Actions are required delivery readbacks; consult that PR for live state.
The next native task is to bind measured compiler configurations to the actual
Spack/EasyBuild toolchains and execute downstream installed/generation-isolation
acceptance. Full P4/P5 profiling and matrix gates remain open.

## Compiler binding assessment lane — 3 November 2026

Chat `01a1a104-80d7-7501-86bd-85be63cb2cab` prepared the bounded
[compiler binding assessment](compiler-binding-assessment-20261103.md) in the
managed worktree `/Users/doughnut/.codex/worktrees/hpc-arrow-qualification/voiage`,
branch `codex/hpc-arrow-qualification-20261103`, based on `602f2102a`. Scope is
that assessment and this handoff entry; no native object or build is owned or
active. The next operation is source-resolution/offline qualification and capture
of package-manager effective settings. Earlier assignments remain historical.

## Source-cache control — 3 November 2026

The same lane owns `scripts/qualify_source_vendor.py`,
`tests/test_source_vendor_qualification.py` and its changelog entry for a
bounded source-cache guard. The new regression failed before the change and
passes after rejecting a symlink at the cache directory itself. A Python 3.15.1
macOS run of the four affected source/native-control test files returned 44
passed and 8 Linux-capability skips. This does not establish protection against
symlinked ancestors, concurrent replacement, EasyBuild source resolution or
network isolation. No native build was started. At verification the system
volume had approximately 2.8 GiB free and the local full gate was not run;
retain the 8 GiB reserve. Next: qualify exact package-manager resolved paths
and offline execution on Linux, then complete the full gate and hosted checks.

## Full source-resolution continuation — 3 November 2026

Chat `01a1a104-80d7-7501-86bd-85be63cb2cab` owns the bounded Linux source
preflight in `/Volumes/PortableSSD/GitHub/.codex-worktrees/voiage-scoped-delivery-20261103`,
branch `codex/hpc-native-sources-20261103`. Scope: the graph workflow's opt-in
source mode, private source/offline driver, affected controls and this handoff.
PR #1264 owns the preceding source-cache/evidence packet. No native operation
is running yet. Next: prove complete package-manager fetch/checksum resolution
and network-denied replay before any installed compiler/build dispatch. Both
EasyBuild generations and the Spack root remain required. Existing combined
tox receipts are retained; new automation receives focused and hosted checks.

## Installed qualification preparation — 3 November 2026

Chat `01a1a104-80d7-7501-86bd-85be63cb2cab` owns the next compiler/bootstrap
and installed-runner packet on `codex/hpc-installed-qualification-20261103` in
`/Volumes/PortableSSD/GitHub/.codex-worktrees/voiage-scoped-delivery-20261103`.
Scope: current bootstrap easyconfigs, compiler resource controls, native driver
and affected tests. The source-preflight packet remains separately in PR #1266.
Its 2024a replay passed; 2023a is still running. No installed build is active.
Next: make bootstrap SDK inputs explicit and review effective package-manager
compiler settings before guarded, network-denied installation. Earlier ownership
assignments remain historical; other checkouts and operations retain their owners.
