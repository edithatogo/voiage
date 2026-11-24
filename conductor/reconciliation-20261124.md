# Delivery reconciliation — 24 November 2026

The [backlog archive](./archive/remaining_backlog_delivery_20260831/index.md) records completed
repository closeout. Its inventory retains the scope of every remaining gate.
The review below records the earlier delivery snapshot.

## Repository closeout

Complete the repository-owned scope of [Remaining Backlog Delivery](./archive/remaining_backlog_delivery_20260831/index.md)
with the existing verification and recovery workflow. The audited `main` was
`9a304c95704dc9a6872137b6d34cb4d8e9d9c1d6`. On that revision, 66 checks succeeded
and four were skipped. The Spack evidence PR passed all 15 tox environments
and all 17 required checks. [PR #1403](https://github.com/edithatogo/voiage/pull/1403)
· [Full tox](https://github.com/edithatogo/voiage/actions/runs/45270206429)

The snapshot had 10 open PRs, 24 open issues, 28 local branches, 17 remote
branches and 16 registered worktrees before this reconciliation. The MLX
no-go is archived and its tracking issue is closed as complete.
[MLX archive](./archive/mlx_metal_evpi_backend_20261121/index.md)
· [#1369](https://github.com/edithatogo/voiage/issues/1369)

1. Resolve the disposition of the remaining repository repair PRs. The
   repository health report and serialization addendum are behind `main`.
   The security-template PR is a conflicting draft. The StepSecurity PR
   must respect the current Julia TagBot process-telemetry restriction.
   The offline CI profile awaits a dedicated runner and validated assets.
   [#1376](https://github.com/edithatogo/voiage/pull/1376)
   · [#1352](https://github.com/edithatogo/voiage/pull/1352)
   · [#1351](https://github.com/edithatogo/voiage/pull/1351)
   · [#1350](https://github.com/edithatogo/voiage/pull/1350)
   · [#1296](https://github.com/edithatogo/voiage/pull/1296)
2. Treat the five Renovate proposals as ongoing maintenance. Review any fix
   required for the closeout candidate under the existing security and delivery
   gates. At the snapshot, #1363 had 17 passing required checks and changed only
   the arXiv and JOSS requirements files. [Dependency dashboard](https://github.com/edithatogo/voiage/issues/1045)
   · [#1363](https://github.com/edithatogo/voiage/pull/1363)
3. Finish B12–B14. Reuse unchanged verification and recovery receipts, retire
   only verified integrated refs and worktrees eligible for removal, and record
   the retained work with its owner or gate. Validate Conductor and capture
   the post-merge PR, issue, branch and worktree inventory to close #1053.
   [Backlog plan](./archive/remaining_backlog_delivery_20260831/plan.md)
   · [#1053](https://github.com/edithatogo/voiage/issues/1053)

## Deferred HPC

The Colab Spack installation passed. The remaining EasyBuild foss/2023a and
foss/2024a installed cells, generation isolation, compiler and capacity
qualification, upstream recipe delivery, and hardware work move to the roadmap.
Resume the existing native plan when HPC is prioritized again. Its acceptance
requires all three installed cells and the remaining native gates.
[Spack receipt](./tracks/hpc_source_build_qualification_20261029/installed-spack-colab-20261124.md)
· [HPC roadmap](../roadmap.md#deferred-hpc-qualification)

## Independent follow-up

- Agent-safe engineering has delivered its repository slices. Julia General/JLL
  acceptance still depends on the upstream build and registry outcome.
  [#1110](https://github.com/edithatogo/voiage/issues/1110)
  · [Yggdrasil #14292](https://github.com/JuliaPackaging/Yggdrasil/pull/14292)
- The pyOpenSci survey has been submitted. The venue track awaits the
  maintainer's own submission text and disclosure assessment.
  [#1037](https://github.com/edithatogo/voiage/issues/1037)
- When selecting the next feature after closeout, finish one installed
  `tobac` consumer workflow under its existing acceptance criteria. The broader
  scientific, credal and simulation-outcome plans retain their own scope.
  Optional telemetry stays deferred pending demonstrated need.
  [#1193](https://github.com/edithatogo/voiage/issues/1193)
  · [#1116](https://github.com/edithatogo/voiage/issues/1116)
