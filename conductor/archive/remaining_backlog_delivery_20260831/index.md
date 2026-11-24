# Track: Remaining Backlog Delivery

- [Specification](./spec.md)
- [Implementation Plan](./plan.md)
- [Metadata](./metadata.json)
- [Final recovery and repository inventory (24 November)](./b13-inventory-20261124.json)
- [Engineering closeout checkpoint and recovery (22 November)](./engineering-closeout-20261122.md)
- [B8 verification and B13 inventory refresh](./b8-b13-checkpoint-20261114.md)
- [Evidence](./evidence.jsonl)
- [Baseline and recovery receipt](./baseline-and-recovery-20260831.json)
- [Remaining issue register](./remaining-issue-register-20260831.md)
- [Issue reconciliation receipt](./issue-reconciliation-20260831.json)
- [Interruption recovery receipt](./interruption-recovery-20260831.json)
- [Torch advisory triage](./torch-advisory-triage-20260831.json)
- [Local validation continuation](./local-validation-20260831.json)
- [Post-repair independent review](./post-repair-review-20260831.json)
- [Preparation review](./preparation-review-20260831.md)
- [Observed Codecov delivery](./codecov-observation-20260831.json)
- [Full R check with manuals](./r-manual-check-20260901.json)
- [Current R check receipt](./r-strict-check-20260831.json)
- [Recovery archive verification](./recovery-archive-verification-20261101.json)
- [GitHub issue #1053](https://github.com/edithatogo/voiage/issues/1053)
- [Related roadmap #296](https://github.com/edithatogo/voiage/issues/296)

Repository closeout completed on 24 November 2026. The final inventory records
the recovery verification, retired refs and worktrees, and every retained item.
Issue #1053 records the signed archive delivery and post-merge readback.
See the [HPC roadmap](../../../roadmap.md#deferred-hpc-qualification) for
deferred native acceptance. Scientific, venue, registry and maintainer gates
remain with their owners.

- [Delivery checkpoint (2026-09-01)](./delivery-checkpoint-20260901.md)
- [Delivery and cleanup observations (2026-09-01)](./delivery-checkpoint-20260901.json)
- [Pending pre-closeout receipt (2026-09-03)](./pre-closeout-20260903.json)

The dated pre-closeout receipt binds merged PRs #1087 and #1088, closed repository issues
#1024, #614, and #615, and the post-merge inventory. It remains fail-closed on
an exact native Spack receipt placeholder. Issue #1025 and all remaining human,
scientific, registry, identifier, and venue outcomes stay pending. The sole
unique cleanup branch was retired only after its exact commit and tree were
preserved under a nonbranch recovery ref and a checksum-bound complete-history
bundle. A separately checksum-bound restore record confirms that a mirror clone
resolved the exact commit and tree and passed full strict object verification.
The current repository closeout requires its own post-merge inventory after
eligible branch and worktree cleanup before issue #1053 may close.
The durable ref manifest excludes rotating `refs/codex/turn-diffs/*` app state,
labels that namespace transient, and separately binds every ordered stash
reflog entry by object ID, selector, and subject.

- [Prepared continuation handoff](./implementation-handoff.md)

The 29 October capacity correction in the handoff supersedes the 250 GiB
prerequisite cited in the dated records below; the historical observations are
retained as evidence.

- [Current continuation audit](./continuation-audit-20261028.json)
- [Current source qualification](./source-qualification-20261028.json)

The 2026-10-28 source receipt proves archive integrity only. Native solver,
build and installed-module evidence remain open; old running statements are
historical and do not identify a currently live build.

- [Source qualifier review and verification](./source-review-20261028.json)

- [Current source binding verification](./current-source-binding-20261028.json)
- [Recovery and partial local cleanup](./recovery-progress-20261028.json)
- [Native tool/catalogue preflight and remaining gaps](./native-preflight-20261028.json)
- [Polars provider source declarations](./polars-providers-20261028.json)
- [Polars provider review](./polars-provider-review-20261028.md)

- [Dated compiler source and bootstrap qualification](./rust-providers-20261028.json)
- [Dated compiler provider review](./rust-provider-review-20261028.md)

- [Arrow prerequisites, source-build evidence and capacity incident](./arrow-provider-readiness-20261028.json)
