# Review report for Optional MLX/Metal EVPI backend

## Summary

The feasibility study meets the track's no-go exit. Both candidates failed
the frozen common-offset p95 gate. Implementation leaves ML04–ML06 are
inapplicable.

## Verification checks

- **Plan compliance.** Verified ML01–ML03 and ML00 against the registered
  specification, the freeze commit, the installed-wheel receipt and the
  retained data. The four feasibility leaves are complete.
- **Style and correctness.** Checked both Python drivers with Ruff and the
  Rust driver with rustfmt. Reviewed the instrumented calls, bounds, fixture
  generation, dependencies and recorded route. The profile uses synthetic
  fixtures and retains only process measurements and scalar outputs.
- **Behavior.** Recomputed all eight cold-workflow equivalence witnesses,
  21 warm samples per lane, eligible fractions, mixed medians, p95
  failures, and process RSS thresholds. The two no-go outcomes reproduce.
  The follow-up ran 21 matched-result EVPI measurements and three
  import-only controls on the same Mac.
- **Tests.** Passed 26 Conductor cross-reference, normalization and
  follow-through tests on Python 3.15.2. All 15 tox environments and all
  17 required checks passed for the original evidence merge.
  [PR #1374](https://github.com/edithatogo/voiage/pull/1374) ·
  [Full tox for the evidence merge](https://github.com/edithatogo/voiage/actions/runs/44868670167)

## Findings and fixes

### Medium — MPS peak device memory interpretation

The frozen result's `gates.device` boolean compares the retained driver
allocation to the numerical budget. The README now explains why that
observation leaves peak device memory unverified. The independent p95 and
process RSS failures sustain the frozen no-go. The follow-up locates most
of the process RSS gap at PyTorch import and records the separate
first-reduction rise.
[Decision](./evidence/ml00/README.md) ·
[Memory profile](./evidence/ml00/memory-profile.md)

### Medium — Stale delivery trace

The Conductor mapping still listed only exploratory work after
the decision merged. The mapping and track metadata now record the merged
planning and evidence PRs. The index links the issue comment that records
the decision. Archive references retain access to the plan, source commits,
hashes and raw evidence.
[Planning PR #1372](https://github.com/edithatogo/voiage/pull/1372) ·
[Evidence PR #1374](https://github.com/edithatogo/voiage/pull/1374) ·
[Decision on #1369](https://github.com/edithatogo/voiage/issues/1369#issuecomment-7057775267)
