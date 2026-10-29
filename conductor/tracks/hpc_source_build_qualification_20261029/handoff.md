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
