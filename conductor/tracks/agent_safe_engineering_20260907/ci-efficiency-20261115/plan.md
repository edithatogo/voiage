# Implementation plan

- [x] Inspect current main, open PR ownership, required checks and job timings.
  Baseline: `cee30c63bc8c4edfe8968544315d21065fb34142`.
  Existing focused workflow contracts: 52 passed under Python 3.15.1.
- [x] Remove unrelated dependency setup and add PR-only supersession to the two
  selected workflows. Implementation: `dba304a5a7b1b608cd59c2637f80cb9d49d975e6`.
  All 28 affected job definitions, triggers and retained action pins are preserved.
- [x] Verify the focused workflow and runtime changes.
  The 86 selected workflow/security/shard tests pass under Python 3.15.1;
  the standard-library-only harness reports zero findings across 37 workflows.
  The frozen tox bootstrap installs no project/scientific dependencies.
  The frozen C15 environment passes all 95 tests with zero skips, including
  four JAX/Polars oracle cases, with all three experimental distributions absent.
  The collector received one privacy-screened export and verified correlation.
  Exact source, configuration hashes and retained result files are in
  [validation.json](./validation.json).
- [!] Complete the required local `tox run-parallel -p 2` gate.
  The first attempt stopped before execution on the checkout volume. An isolated
  worktree then reached the unchanged 2 GiB free-space floor after 103.725 seconds
  (exit 125; minimum free bytes 1,966,112,768). Lint, base-runtime probes,
  21 producer tests, 19 JOSS tests and Vale over 1,053 files completed successfully.
  The full gate is not passed. Its remaining requirement is sufficient local
  storage for the isolated tox environments, with all resource limits retained.
- [x] Independently review the implementation and publish one focused draft PR.
  Independent review found no material regression. PR #1330 retains the source
  changes and evidence without altering existing dependency/security PRs.
- [ ] Verify current-revision hosted checks and record comparable execution and
  queue timings. The first candidate's 11 workflows remained queued when this
  receipt was prepared. No hosted speedup or whole-suite pass is claimed.

This follow-up commit records evidence and status only. The four workflow files,
all runtime code, dependency locks and tox configuration are byte-for-byte equal
to the tested implementation commit. Hosted checks remain required for the
current PR revision before integration.
