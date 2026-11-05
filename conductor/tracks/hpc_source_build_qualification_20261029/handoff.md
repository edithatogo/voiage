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

## Installed runner admission — 4 November 2026

The same chat and standard worktree own the installed pipeline on
`codex/hpc-installed-qualification-20261103`. PR #1266 merged as `6c94359aa`.
The new SDK source matrix passed all three cells in run 41648385434; the
stronger Spack cache replay passed run 41647637737. The installed driver first
repeats source admission in its own private workspace, then uses a root-owned
aggregate supervisor around an unprivileged driver and root-owned stage
monitors. Native build commands remain unprivileged and network-isolated.

Agent review corrected Spack unload scope and rejected broad system-library
acceptance. Forty-three focused controls pass locally, with the one Linux/root
cleanup capability test correctly unexecuted on macOS. The dedicated Linux
controls job must prove the root/runner/root-stage/detached-child cutoff before
full build dispatch; each installed job repeats that host-specific admission.
No installed build is running at this checkpoint. The accepted envelope stays
12 GiB sampled RSS, 8 GiB minimum free, four hours for the install stage and
5.5 hours for the whole driver. Effective compiler configurations and explicit
stage-2 commands are retained. Installed imports, fixed source identity,
numerical/Arrow/Parquet/Polars probes, selected-prefix linkage and unload checks
remain actual success conditions; no full matrix acceptance is claimed.

## Installed dispatch and first failure — 4 November 2026

The same chat/worktree owns PR #1267. Four preceding scoped packets merged as
PRs #1263–#1266. Root/runner/privileged-stage/detached-child cleanup admission
passed in run 41652588728 (9.06 seconds, cleanup_complete=true), allowing the
installed matrix dispatch. Source SDK replay passed all three cells in run
41648385434. The first installed run 41653232566 retains its independent Spack
operation. Both EasyBuild cells stopped before compilation because the fresh
runner lacked UCX's declared libibverbs development package; both aggregate
and stage cleanup completed. Commit 7b3d946a0 adds that OS prerequisite without
suppressing dependency checks. Run 41655526715 retries only EasyBuild 2023a/2024a.
Do not dispatch another Spack attempt while the existing operation is active.

Full tox run 41652588718 passed fourteen environments; coverage exposed three
AST-extraction fixture NameErrors, fixed in b2e91e45e. The subsequent focused
Python 3.15.1 provider/compiler/probe suite passed 43 tests, with one genuine
Linux/root capability skip on macOS. Reuse unchanged full-gate environments
and require current hosted full coverage. PR review also clarified probe imports
and the expected already-reaped cleanup outcome in 23b69cb18. The Python 3.15
lazy-import syntax is intentional and passes the qualified interpreter.

Small frozen receipts are retained in `evidence/installed-admission-20261104/`.
No installed cell or complete native matrix has passed at this checkpoint.
Next: inspect terminal native receipts, fix only demonstrated failures, then
merge PR #1267 after current-head required checks, signatures and threads pass.

## Colab network diagnosis ownership — 5 November 2026

Chat `01a1a104-80d7-7501-86bd-85be63cb2cab` continues in the standard worktree
`/Volumes/PortableSSD/GitHub/.codex-worktrees/voiage-scoped-delivery-20261103`,
branch `codex/hpc-native-install-results-20261104`, based on merged `308adce20`.
Scope is the isolated build network helper, affected controls, Colab diagnosis
and retained terminal receipts. The owned Colab CPU session is
`voiage-hpc-diagnosis-20261105`; no full matrix build is dispatched yet.
Earlier assignments remain history. Pleiades and NAS checks are read-only;
no workload, environment, storage or service changes are owned there.

The first installed Spack attempt and both corrected EasyBuild attempts stopped
at the four-hour install-stage budget, with cleanup_complete=true. Spack installed
the dated compiler then stopped in stable Rust; 2023a was in Tcl tests and 2024a
in Perl tests. Neither RSS nor free space triggered these stops. A real Colab
regression proves the existing fresh network namespace cannot complete a local
TCP exchange with loopback down. The new helper enables only loopback in a
namespace distinct from the parent, checks interfaces through the current
namespace socket API, clears supplementary groups and drops GID/UID before exec.
External ENETUNREACH and parent-network rejection remain required.

Agent review corrected an inherited-sysfs-view assumption. Colab's normal netns
mode denies loopback configuration (EPERM); a separate private user/net namespace
can configure loopback, but exact root/runner mappings are rejected by the
installed mapping policy. That platform remains unqualified; no full build was
dispatched there and no mapping policy was changed. The CPU diagnosis session
is being released after receipt retrieval. GitHub root controls must qualify
the new helper before another installed dispatch. Tests are retained rather than
skipped to conceal a platform gap. Private network-host inventory stays local.

## Two-backend admission — 5 November 2026

The user explicitly authorized trying Colab and Pleiades WSL, including a private
build workspace and prerequisite setup. The optional
`VOIAGE_NATIVE_USER_NAMESPACE=1` backend uses util-linux 2.39.3's supported
single-user root mapping plus one exact runner UID/GID range, with setgroups
allowed and exact kernel-map readback. Repeated range options replaced the root
mapping on that version and were rejected by the helper; they are not used.
The disposable Colab VM received only missing root-owned runner-ID entries in
its subordinate-ID files, preserving other entries and permissions. This is
runtime-local provisioning, not a grant on the shared WSL host. Two affected
Colab cases now pass: local TCP, external ENETUNREACH, effective non-root IDs,
zero effective capabilities, host-visible ownership and aggregate cleanup.
The ten unchanged controls retain their prior passing proof.

WSL admission exposed absent process I/O-counter support, which previously
interrupted sampling. The explicit capability-gap path retains CPU/RSS/disk/time
limits and marks I/O measurement incomplete. Its new regression failed before
the repair and passes afterward. New native admission and current-candidate
hosted/full-tox checks remain required before installed dispatch. Private host
addresses, hardware inventory, access files and connection receipts stay local.

## Privilege-context review repairs — 5 November 2026

Both backends passed all thirteen affected native admission checks at `61bf3434e`.
The first full attempts stopped during source preparation, before compilation.
WSL sudo-rs explicitly rejected preserve-entire-environment, and the unprivileged
driver inherited the root profile; Colab's source packaging lacked a Cargo
frontend. Attempts remain preserved in their private roots. A selected-user
build-environment allowlist now derives HOME/USER/LOGNAME from the real account
and excludes privileged authentication variables. The stage helper also uses an
absolute sibling path, fixing native Rust stages whose working directory lies
outside the checkout. The new runtime regression uses that consumer directory.
These review fixes require fresh admission and final-current-candidate hosted
checks; no earlier native attempt is relabeled as successful.

## Named sudo compatibility — 5 November 2026

PR #1268 merged as `ce98dcbe7` after all required hosted checks, signed heads,
resolved threads and a passing full tox run at `816da7577` (run41979612949).
The same writing chat/worktree continues on
`codex/hpc-sudo-compatibility-20261105`. Scope is named sudo environment
forwarding in source/install/native-compiler drivers and its actual nested
supervision regression. WSL's sudo-rs rejects preserve-entire-environment but
its operation-specific named preservation probe succeeds. The declared name set
includes receipt settings before their later assignments; it excludes credentials.
The aggregate cleanup control now also requires compiler-receipt-path propagation.

Both network hosts were authenticated read-only. Subsequent user direction
explicitly authorized separate Colab and WSL build workspaces/prerequisites.
Private paths, sessions, connection settings and native process ownership are
retained locally, not published here. The NAS remains read-only. WSL's requested
prerequisites installed, but apt exposed an existing OpenSSH configuration
failure; no unrelated service repair/restart was attempted. Colab's direct SDK
response failed checksum, and the bounded verified-archive relay stopped partway.
Those inputs were not accepted or used; only whole verified SDK bytes may be
used for the source-packaging frontend. Native compilers still build from source.

Both backends pass fifteen native admission controls at `9dcd1e741`. The complete
193 MiB packaging SDK relay matches the pinned digest; the earlier partial
response and interrupted relay remain unaccepted historical attempts. WSL's
terminal aggregate reports and changed boot identity establish host interruption
after detached launches; no compiler bootstrap failure is inferred from those
stops. Subsequent jobs retain attached real WSL command sessions without changing
global WSL idle settings. A real newer-Lmod initialization also exposed an unset
FPATH assumption under Bash nounset; the scoped module-path default preserves
the rest of strict shell behavior. Native matrix acceptance remains pending.

The next source retry also exposed the fixed `.graph-tools/source` candidate
checkout left by the earlier attempt. It is preserved. Native/source mode now
places the candidate checkout inside the already unique VOIAGE_SOURCE_WORK,
avoiding collisions without deleting or moving prior worktrees. The source-driver
failure regression verifies that this attempt-specific workspace reaches the
graph subprocess. Existing historical paths remain historical receipts.
