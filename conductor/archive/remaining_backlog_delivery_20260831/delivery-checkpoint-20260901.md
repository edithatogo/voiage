# Delivery checkpoint — 2026-09-01

This checkpoint records delivered repository work through main
`15e514a53bec23b2a220ccfee7cf6c4d363a363e`. The full backlog remains open.
The [observation receipt](./delivery-checkpoint-20260901.json) preserves source
receipt hashes and original timestamps; earlier dated evidence is unchanged.

## Protected delivery

| Pull request | Merge commit | Delivered scope |
| --- | --- | --- |
| [#1041](https://github.com/edithatogo/voiage/pull/1041) | `5e062484` | Reviewed dependency update |
| [#1049](https://github.com/edithatogo/voiage/pull/1049) | `fd54b647` | Reviewed dependency update |
| [#1050](https://github.com/edithatogo/voiage/pull/1050) | `ffd569dc` | Reviewed dependency update |
| [#1060](https://github.com/edithatogo/voiage/pull/1060) | `66f51d21` | Spack source-build preparation and retained build limits |
| [#1061](https://github.com/edithatogo/voiage/pull/1061) | `15e514a5` | Local Julia v2.2.0 BinaryBuilder candidate |

The merge receipts for #1041, #1049, #1060, and #1061 bind reviewed and merged
trees. The #1050 receipt records its observed merged state. None establishes
an external package build or registration result.

## Controls and cleanup

B4 is complete: PR #1057 delivered the repaired dependency and historical
reproduction controls, and [#656](https://github.com/edithatogo/voiage/issues/656)
closed after the final drift audit. That dated audit found no harness findings,
verified the existing hosted ruleset and Codecov patch observation, and retained
the independently restored recovery bundle. No hosted setting mutation,
scientific approval, or venue acceptance follows from that result.

Three retired worktrees were removed after recovery verification. The guarded
local branch deletion covered merged #1057 and #1058 and closed, unmerged
#1056. Its unique Actions proposal remains at
`refs/codex/recovery/20260901-retired-actions-proposal`; it is not called merged.
The receipt preserves successful removal results and the recovery bundle
verification. This subset does not complete final cleanup while active work
remains.

## Julia registration staging

The fourth staging item in [#1023](https://github.com/edithatogo/voiage/issues/1023)
is prepared locally: `Project.toml` supplies the package identity, version and
compatibility metadata; the README gives the exact subdirectory registration
command and prerequisites; TagBot declares that subdirectory and the `julia`
tag prefix. Existing Julia readiness and release tests check those contracts.
This checkpoint corrects the README's historical version from v2.0.0 to v2.1.0.

The original upstream PR and its recorded platform builds target v2.1.0. The
merged v2.2.0 recipe remains an unbuilt, unsubmitted candidate. The staging
assessment does not mean registration is executable now: the real JLL UUID,
JLL dependency wiring, clean-depot installation without the manual override,
and supported-platform tests remain required before maintainer review and the
external Registrator command. JuliaHealth listing and a domain vignette also
remain separate unfinished work. No JLL or General acceptance is claimed.

## Issue reconciliation and remaining work

The posted corrections for #876, #853, and #850 distinguish implemented
contracts from unresolved scientific findings and independent review. All
three remain open. Updates to #1023, #555, and #1025 record their delivered
preparation and unmet build or registry criteria. The #1025 receipt predates
any later Linux VM experiment and is not a current execution-status report.

The public Scorecard observation is 8.0 for commit
`ab91035774a93fe8a7205f742bae7650af8088d8`, dated 2026-08-24. Its retrieval and
#620 update do not establish a scan of this checkpoint's main commit.

Other plan items remain in progress or pending. At the initial snapshot, test-acceleration delivery was still pending; the
later observation below supersedes that item. Actual HPC builds, R acceptance,
scientific review, identifiers,
venue submission, and final inventory reconciliation retain their own evidence
requirements. Personal declarations already confirmed by the maintainer are
not requested again. This checkpoint neither archives the track nor declares
the complete user goal satisfied.

## Later delivery observation

After the initial `15e514a5` snapshot,
[PR #1059](https://github.com/edithatogo/voiage/pull/1059) merged as
`790727317e6f154f54fdb1d037d7143752dc7d0b`. The separate receipt preserves that
merge observation. [#1028](https://github.com/edithatogo/voiage/issues/1028)
closed as completed at 2026-08-31T16:34:39Z after verification of the measured
profiles and acceptance criteria. B5 is therefore complete. This later result
does not change the initial snapshot's commit or the remaining open work.

[PR #1064](https://github.com/edithatogo/voiage/pull/1064) subsequently merged as
`25fc585c0bf9fea69d9bcc4220a2d667975ef3eb`. Its separate receipt records the
reviewed delivery of the dated Rust candidate. B6 remains unfinished: actual
native Linux builds and both requested EasyBuild foss stacks still require
their own successful evidence.

All eleven observed workflows on `25fc585c` completed successfully. The actual
[hosted Scorecard run](https://github.com/edithatogo/voiage/actions/runs/33415615230)
reported 8.2 at 2026-08-31T16:43:01Z for that commit. This later run is distinct
from the older public API snapshot above. Its Signed-Releases check remains
6; neither observation establishes full completion of #620.

## Local verification

All fifteen required tox environments passed across the retained initial run
and repaired coverage run. The initial run passed fourteen environments but
failed one metadata-normalization check: 4,666 tests passed, 16 skipped, and
coverage reached 95.16%. The existing normalizer repaired only this track's
metadata timestamp precision and Unicode serialization. Ten focused tests then
passed, followed by full coverage with 4,667 tests passed, 16 skipped, and
95.16% coverage. The failed run and both log hashes remain in the receipt.
No test, schema, or coverage threshold was relaxed. Final prose and Conductor
checks validate these additive receipt updates separately from runtime tests.

## Assurance delivery

[PR #1063](https://github.com/edithatogo/voiage/pull/1063) merged as
`b3f53c2b1ad797a947e066dd9ea48540e4e19455` at 2026-08-31T17:13:03Z. Its
reviewed and merged trees agree. The historical agent-assurance validator now
checks the existing semantic bindings and independently pins the assurance
record's narrative. The original record and schemas remain unchanged; this
repository repair does not supply independent scientific review or resolve
the nineteen pending findings. Issues #876, #853, and #850 remain open.

## Second cleanup wave

The reviewed cleanup completed at 2026-08-31T17:14:44Z for the merged #1060,
#1061, and #1059 worktrees and their three exact local branches. Independent
recovery, archive hashes, and absence of processes were checked before removal.
Their remote branch heads were already absent at the recorded observation;
this checkpoint does not claim to have deleted those remote heads. Three
redundant recovery branch aliases were also retired while preserving their
exact objects under nonbranch recovery refs and the verified bundle.

The receipt retains both pre-removal and completed observations. A fresh local
check confirms the three paths and branches are absent and the recovery refs
still identify their recorded objects. Active worktrees and unique work remain
protected; B12 is in progress, not complete. The later #853 update records
assurance delivery while leaving the issue open.
