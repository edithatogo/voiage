# Capacity slice verification — 28 October 2026

## Implemented and reviewed

Implementation: signed commit `09f3bc7643330201c5bab0b01feded30c1ffbb35`,
tree `140d95066bd6c868fe12753ef282fa832e86f22d`, based on the existing Arrow
owner's `aed8b4110` checkout. This branch adds only the capacity slice; its parent
provider work belongs to the existing owner. No PR, hosted delivery or merge of
this capacity slice is claimed. Review covered the guard, tests, track ownership,
pilot hashes, scope boundaries and current issue #1025 requirements.

The local Python interpreter actually reports 3.15.1. Dependencies were installed
from the unchanged frozen lock in an isolated external-SSD worktree. Historical
native helpers, runtime packages, locks, scientific features, compiler generations
and the existing capacity investigator's files were not changed.

The new guard retains legacy fields and command exits, deduplicates filesystem
samples, binds source/configuration/runtime identity and measures selected
artifact roots only at boundaries. It does not equate shared allocated blocks,
filesystem deltas or a guest's apparent capacity with attributable physical use.
The existing explicit consumer download-cache option was used; consumers still
create fresh installed environments. No shared source store or cleanup mechanism
was adopted without its native qualification.

## Executed checks and retained failures

- Red witness: three new tests failed before implementation (missing disk helper,
  missing artifact helper and unsupported CLI options).
- Green: all 14 guard tests passed, including low-space interruption, retained
  temporary evidence, terminal monitor failure and very short jobs. Ruff passed.
- Full required command: `tox run-parallel -p 2`, wrapped in the guard with
  selected artifact directories and a mixed-cache declaration. Thirteen lanes
  passed; docs and coverage lanes failed for the bounded causes below.
- Coverage executed the full test suite: 5,497 passed, 19 skipped, one failed.
  Branch-aware coverage was 94.73%, above the existing 90% gate.
- The sole failed test was Conductor metadata normalization: JSON escaping of
  existing Unicode differed from the canonical serializer. Two metadata files
  were corrected without changing their decoded values. A failure-only rerun
  selected no tests because its cache was empty; it is not verification. The
  explicit three-test normalization rerun passed, followed by all seven affected
  normalization/workflow-record tests after recording observations.
- Docs initially failed because the isolated worktree lacked the pinned
  `.repo-tools/astro-polyglot` submodule. The bootstrap's generated placeholder
  directory was preserved in local evidence, then the exact submodule commit
  `054d11e08d8b28fd50821ede8a97f8be3ad50447` was initialized. `tox run -e docs`
  passed: zero Astro diagnostics, 1,043 pages and valid internal links.
- Repository harness and reference policy passed. Pilot artifact SHA-256 bindings
  and capacity-plan links were checked. Historical G04.1–G04.3 remain accepted.

The initial full-run receipt remains failed. The focused repairs complete the
identified local verification failures; they do not fabricate a single passing
full-run timing or capacity measurement. Runtime, lock and test infrastructure
were unchanged by those metadata repairs. No duplicate full gate was launched
for documentation/status updates. Hosted checks remain required before delivery.

## Evidence locations and remaining gates

Raw full-run resource observations are retained in
[ordinary-full-tox-resource.json](./capacity-evidence/ordinary-full-tox-resource.json)
and indexed by G04's `ci-measurements.json`. The initial full log, tox result JSON,
red/green logs, failed-selection result, explicit reruns and docs repair receipt
are retained under `.conductor/local/capacity/` in the isolated capacity worktree;
the accompanying manifest binds their bytes. These local transcripts are not
published artifacts.

G04.4 implementation is locally verified. G04.5 remains in progress: the mixed
run observed 2.28 GiB of filesystem growth but failed, and its measurement also
includes unrelated activity and submodule bootstrap. No minimum-space budget or
cold/warm saving is claimed. The successful Linux SDK/Polars pilot remains scoped
to its own receipt. At this 28 October checkpoint B6-CAP1 awaited the support
decision; the 29 October continuation below records its acceptance. B6-CAP2 still
requires compiler/Arrow/full-stack evidence; B6-CAP3 requires a qualified builder
before source-store adoption; B6-CAP4 awaits those representative measurements.
B8 and B12 retain acceptance and preservation-first cleanup ownership.

## Continuation ownership — 29 October 2026

Chat `01a1828e-c05a-7752-81a8-3a89b970678f` (Voiage — Build capacity —
Calibration and qualification) retains the isolated source worktree
`/Volumes/PortableSSD/GitHub/.codex-worktrees/voiage-capacity-conductor-20261028`,
branch `codex/build-capacity-conductor-20261028`. Owned paths: capacity planning,
resource guard/budget helpers and their tests, lane evidence and G04 capacity
observations. Native scope: ordinary local verification only, plus bounded
readback of pinned native measurement receipts. No Arrow provider development,
native Arrow build, shared dependency/runtime promotion or main integration.
Provider chat `01a17ab7-148e-7042-be1e-03a9478643da` remains repository integrator;
native-capacity chat `01a17093-d6f6-7302-a1fa-a868618ac648` owns measurement PRs.
Earlier assignments above are historical. Next operation: qualify a matched
cold/warm ordinary-test budget validator, then measure that explicit calibration
pair with private caches. Unchanged earlier tests/receipts are reused; these runs
collect the missing calibration evidence rather than repeat a status check.

The user explicitly accepted the scope recommendation in this chat on 29 October:
retain both EasyBuild generations and source-built compiler qualification; use
authenticated compiler SDKs for routine tests and dependency pilots. This accepts
the support policy, not a native build result or resource threshold.
