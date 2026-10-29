# Current continuation handoff — 28 October 2026

The active maintainer goal authorises implementation, review, fixes and archival
of all four named tracks: #1053 (including B6/B8/B12–B14), agent-safe
engineering #1110, optional native telemetry #1116 and tobac consumer
qualification #1193. Each requires implementation, review, fixes and actual
acceptance before archival.
Acceptance criteria and external authority remain
unchanged. Read `continuation-audit-20261028.json` before implementation.

## Current source and ownership

- Accepted source baseline: `3df3cb49e5c927075c0609a48a2535b4457424d4`.
- Integrator branch: `codex/hpc-arrow-providers-20261028`; primary checkout.
- Root owns shared governance, locks, workflows and packaging changes.
- Parallel investigations are read-only; no concurrent tox/cache writers.
- Use repo-local Python 3.15.1 and `scripts/guard_verification.py` for execution.
  Native compilation must retain job, memory, elapsed-time and physical-space
  limits. Do not restart a missing or terminal historical job from a status file.

## B6.1: qualify the exact current source archive

First freeze the archive/provenance/runtime negative controls. The implementation
slice owns only `scripts/qualify_hpc_source.py` and
`tests/test_hpc_source_qualification.py`. It must reject wrong archive digest,
wrong revision/tree, unsafe members, altered runtime source, retired dependency
requirements and a Python runtime mismatch. Qualification must use a real source
archive from the immutable baseline, not only synthetic fixtures.

The separate immutable Git exports under the ignored local goal directory keep
frontier lock observations away from the build source. The source-export receipt
verifies every exported regular tracked file. Unrelated latest tool updates are
not adopted into the main lock as part of this slice.

Source qualification is only one B6 requirement. It does not prove a native
solver, full source installation, installed numerical/Arrow/module behavior,
EasyBuild compiler-generation matrix or upstream packet completeness. Keep B6
open until those required witnesses pass. Historical recipes/receipts remain
byte-preserved and are not current Python installation defaults.

## Native capacity and false status findings

The earlier preflight observed a stopped Docker daemon. A fresh read now finds
OrbStack running with 10 configured CPUs, about 16 GiB configured memory and
three existing containers. None was started, stopped or reconfigured here;
availability is not HPC qualification. The historical internal HPC
VM path is absent; an old statement that its build is running is not a verified
wait. The only default Lima instance observed belongs to another project and
must not be reused or modified. Current physical free space is below the retained
250 GiB EasyBuild floor. Sparse virtual capacity is not physical free space;
do not reduce that floor or claim native acceptance to force completion.

## Subsequent work remains required

- Qualify current Spack/EasyBuild recipes and installed modules, then review B8.
- Audit all current PR/issue criteria, preserving human/scientific/venue gates.
- B12: verified independent restore of unique work, notes and stashes before
  exact-SHA branch pruning; keep active worktrees and maintained release lines.
- B13/B14: authoritative post-delivery inventories and final cross-reference
  validation. Archive only after actual criteria are met.

The engineering, telemetry and named consumer tracks have their own acceptance
records. They are authorised implementation scope under the expanded goal; their
acceptance cannot be inferred from the HPC provider work.

## Capacity correction and ownership — 29 October 2026

Chat `01a17ab7-148e-7042-be1e-03a9478643da` owns the capacity correction in
`/Users/doughnut/.codex/worktrees/easybuild-capacity-correction/voiage` on
`codex/easybuild-capacity-correction`. Its scope is capacity guidance and
track records; the historical driver and receipt schema are byte-preserved.
No native build is active in this lane. Next: qualify and integrate the correction.

User direction withdraws the prior 250 GiB estimate as a current prerequisite.
The byte-pinned historical replay driver retains its original check for receipt
validation. This supersedes the earlier capacity statements in this handoff and dated receipts. The
[GitHub Polars build](https://github.com/edithatogo/voiage/actions/runs/40691333092)
completed with approximately 10 GiB sampled disk growth. This is not evidence
of completion or sufficient capacity for the full EasyBuild matrix. Retain the
2 GiB free-space reserve and guard both guest and host backing volumes.

## Tobac shared-cost boundary — 29 October 2026

Chat `01a17ab7-148e-7042-be1e-03a9478643da` owns
`/Users/doughnut/.codex/worktrees/tobac-shared-cost-witness/voiage`, branch
`codex/tobac-shared-cost-witness`, for the shared-cost contract witness in
`tests/test_tobac_shared_cost_boundary.py`. There is no native service scope.
Next: freeze an extension contract against the named consumer and demonstrate
its independent red cases, preserving the existing v1 scalar-cost behavior.
Three guarded Python 3.15.1 boundary tests pass: v1 sums direct costs, rejects
undeclared shared assets, and charges 1.3 rather than the consumer's 0.8 when
the same 0.5 fixed cost is embedded in both scalar source prices. This witness
does not implement or qualify shared-cost accounting.

The candidate v2 input uses explicit asset IDs, fixed costs and owned-asset
IDs, with source `cost` retaining its direct-cost meaning. Union cost applies
to feasibility and results; the first acquisition pays the asset in the
ordered marginal ledger. Unknown or duplicate IDs and unit inconsistencies
must fail closed. Existing v1 input and result schema/behavior remain intact.
The two valid consumer-derived v2 examples have an observed red result: the
current v1 validator rejects the new `shared_costs` property. A new result
schema must identify the v2 contract; this red test is not implementation.
Frozen baseline: `b586f2f713133be5feac6f1de42d899e146c530f`; input schema
SHA-256 `dfdbda6e2e239815b9cada0bfd4665ca59bc87caf14ad6130724013554941b45`,
result schema `6fffac3024e148b37bc1f71c32ad9be286b6691baa17252ea835575961c4c416`,
contract module `36a5081d20fcf24ad475177b03e66924ea917e44fe360dc3e0f0e3d9f7c3c37e`,
and evaluator `042c466f3400297505247cfe273a2ac4458db1064941cdb021148b28d55042a5`.

The draft v2 implementation now passes 49 affected portfolio tests, including
v1 normative behavior and cost-basis/optimum tampering checks, on Python
3.15.1 under the resource guard. Focused Ruff and type checks pass. This is
not acceptance: verify source and wheel schema packaging, clean installation,
remaining edge cases, the full local gate, current hosted checks and the named
tobac consumer before completing the track or merging.
