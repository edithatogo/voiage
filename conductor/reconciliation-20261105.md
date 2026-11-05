# Delivery reconciliation — 5 November 2026

## Current direction

The maintainer deferred HPC work on 5 November, except MLX/Metal extensions.
Spack, both EasyBuild generations, cluster/scheduler work and other accelerator
qualification are deferred, not accepted or complete. Preserve source, compiler,
resource and failure receipts. Do not restart native campaigns without renewed direction.
MLX/Metal remains eligible for bounded Apple Silicon implementation and device
qualification; this exception does not imply an existing MLX backend or measured speedup.
Scientific and consumer work independent of HPC remains eligible.

## Accepted implementation and remaining evidence

The execution packet guard, version identities, native structured logging,
logging privacy/resilience and VOP pilot tracks record completed implementation.
The parent engineering programme still requires the G03.4 provider-compatibility
acceptance, G04.4 artifact/measurement/replay evidence, G05.4 live-target
witness, and final review. Optional native telemetry is separately gated by demonstrated
need; its exporter is not implemented. Do not repeat the five completed tracks.

Polars container, pandas-free I/O and translation-reference delivery are accepted
through the retained PR #1209 and related receipts. Optional producer tests still
use pandas; Arrow metadata compatibility is not a pandas runtime dependency.
The remaining audit must distinguish runtime imports, test producers and
immutable historical records.

PR #1270 merged as 0d32ccbb3dbee64134d8b6e357a29ba796015335.
Full tox run 42006398652 passed for the unchanged runtime candidate; required
final-head PR checks passed separately. All three native graphs passed source
replay, but installed Spack/EasyBuild acceptance was not obtained. WSL Boost
segfault and binutils failures remain unresolved full-install gates. A focused
binutils 2.45 macro-patch compilation passed; that is not installed acceptance.
Pleiades subsequently became unreachable over SSH. Colab was intentionally
stopped through its owned supervisor on deferral: the driver is absent and
the terminal receipt records cleanup_complete=true and interruption, not success.
No WSL termination or cleanup is claimed while the host is unreachable.

## Remaining active work

1. Reconcile unique worktree commits and complete protected delivery. PR #1243
   merged; PR #1248 merged as `59c88e8ce` after 65 successful hosted checks,
   four workflow skips and the full local tox gate for
   `d706ae6a99af0d1772ee91e444bd35dbe9e3a172` (tree
   `d44d28702b746999ea7f8b6146733d011a0a00d0`) on repo-local Python 3.15.1.
   The guarded resource receipt is retained in the PR worktree at
   `.conductor/local/full-gate-1248-d706/resource-report.json`. Preserve
   recovery and retire integrated refs only after their disposition is established.
2. Close the bounded parent engineering live-target witness and review gaps.
3. One installed tobac workflow passed against published voiage 3.1.0; see
   `tracks/tobac_consumer_qualification_20261025/evidence/installed-20261105/`.
   The wider consumer track and its default version requirement remain open.
4. Finish evidence-based GitHub, Polars, security/privacy and quality audits.
   OpenSSF Scorecard is already configured; run 42008425486 passed. Assess its
   findings rather than claiming a perfect score or adding duplicate scanners.
5. Scope MLX/Metal numerical equivalence, CPU fallback and real device evidence.
6. Keep venue survey and human-written submission text pending. PR #1265
   delivered the refreshed packet; submission and venue acceptance remain external.
7. The integrated twenty-package methods/software expansion remains proposed;
   select and qualify a bounded package before treating it as implementation work.

## Recommended next implementation

Deliver the scoped critical PyPI action security fix in PR #1271 through
protected checks. Then close the parent engineering
witness and review gaps. Optional telemetry should
wait for demonstrated need. MLX/Metal starts with one measured kernel and CPU
fallback rather than a general acceleration framework. An isolated GPU probe
ran, but numerical parity, fallback and benchmarks remain unqualified.

## Branch reconciliation observation

The telemetry loopback lane at `0dec9cff1` retains seven commits relative to
main, but its complete tree `f36033999236f07a8a769e0640ec808d89b66dc1`
exactly matches squash merge #1251 (`48fe14771`), which is an ancestor of
main. Its source content is integrated; preserve its ref and worktree until
current ownership and cleanup disposition are confirmed. The later maintainer
direction defers further telemetry execution.

The primary checkout's local `main` retains five Conductor commits through
`a9ec5b1cf` that are not ancestors of remote main. Their completion labels
predate the later engineering acceptance audit and the opt-in OTLP merge. The
unique telemetry test in `9ee828d64` asserts the older no-network-export state,
so it is not suitable for application over the merged implementation. Retain
the checkout and commits; reconcile ownership before any cleanup, and do not
replace the later open engineering and telemetry gates with those labels.

The HPC Arrow qualification checkout at `0436e1bee` has 27 changed paths
relative to its merge base with the current observed remote main. Twenty-three
match main byte-for-byte, including the tox interpreter fix, venue packet and
source-cache guard. Four differ: the changelog and three HPC/backlog handoff
or plan files. The three handoff/plan diffs from the branch to main contain
only additions on main; its changelog entry is retained on main in a later
position. Their historical text is therefore preserved, with later main
receipts extending it. Preserve the checkout and its deferred HPC evidence
until ownership and cleanup disposition are confirmed.

The HPC compiler-requirement checkout at `ef535f31f` has 12 changed paths
relative to its main merge base; only one is byte-identical to current main.
The specific Rust 1.96.0 stage-zero dependency and 1.95.0 bootstrap hashes are
retained in main, which subsequently added newer compiler and installed-driver
changes. This limited inspection does not establish all remaining paths as
equivalent. Preserve the branch and its compiler evidence with HPC deferred.

The scientific ecosystem checkout at `b9e540267` has the exact tree
`924e81f0a0aad9d1824c85e951b644398bed989e` of merged PR #1253
(`5cc5e958f`), including the wheel shadow-package guard and its test. Its two
branch commits are therefore source-integrated. The checkout also has an
untracked `.tox-integration-5vXkiN/` directory; preserve it and the worktree
until its owner and cleanup disposition are established.
