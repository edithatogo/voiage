# Current continuation handoff — 28 October 2026

The active maintainer goal authorises implementation, review, fixes and archival
of this track, including B6/B8/B12–B14. The current goal is scoped to #1053;
other tracks are recorded only where they affect its acceptance criteria.
Acceptance criteria and external authority remain
unchanged. Read `continuation-audit-20261028.json` before implementation.

## Current source and ownership

- Accepted source baseline: `3df3cb49e5c927075c0609a48a2535b4457424d4`.
- Integrator branch: `codex/backlog-acceptance-20261028`; primary checkout.
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

The selected Docker daemon is stopped. Its shared VM configuration permits
16 GiB memory and is not changed by this handoff. The historical internal HPC
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
records. Do not implement or archive them solely under this narrowed goal.
